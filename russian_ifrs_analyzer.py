import json
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.schema import Document
import pandas as pd
from tenacity import retry, stop_after_attempt, wait_exponential
import openai
import time
import pdf2image
import pytesseract
import os
from datetime import datetime
import re
from typing import List, Dict, Optional, Any, Tuple


class RussianIFRSAnalyzer:
    def __init__(self, pdf_path: str, openai_api_key: str, max_retries: int = 5):
        self.pdf_path = pdf_path
        self.openai_api_key = openai_api_key
        self.max_retries = max_retries
        os.environ['OPENAI_API_KEY'] = openai_api_key
        
        self.llm = ChatOpenAI(
            temperature=0,
            model="gpt-4o",
            max_retries=max_retries,
            request_timeout=30
        )
        
        self.embeddings = OpenAIEmbeddings(
            max_retries=max_retries,
            request_timeout=30
        )
        
        self.pages_text = []
        
        self.last_request_time = time.time()
        self.min_request_interval = 1.25
        
        
        # Map metrics to their statements
        self.metric_locations = {
            'total_assets': 'balance_sheet',
            'total_equity': 'balance_sheet',
            'total_debt': 'balance_sheet',
            'cash': 'balance_sheet',
            'revenue': 'income_statement',
            'operating_profit': 'income_statement',
            'net_profit': 'income_statement',
            'ebitda': 'income_statement',
            'interest_expense': 'income_statement',
            'capex': 'cash_flow',
            'dividends': 'cash_flow'
        }
        
        # Map metrics to their likely page identifiers
        self.metric_identifiers = {
            'total_assets': ['отчет о финансовом положении', 'бухгалтерский баланс', 'активы', 'итого активы'],
            'total_equity': ['отчет о финансовом положении', 'бухгалтерский баланс', 'капитал', 'итого капитал'],
            'total_debt': ['отчет о финансовом положении', 'бухгалтерский баланс', 'заемные средства', 'кредиты'],
            'cash': ['отчет о финансовом положении', 'бухгалтерский баланс', 'денежные средства'],
            'revenue': ['отчет о прибыли', 'прибылях и убытках', 'выручка'],
            'operating_profit': ['отчет о прибыли', 'прибылях и убытках', 'операционная прибыль'],
            'net_profit': ['отчет о прибыли', 'прибылях и убытках', 'чистая прибыль', 'прибыль за период'],
            'ebitda': ['отчет о прибыли', 'прибылях и убытках', 'ebitda', 'прибыль до вычета'],
            'interest_expense': ['отчет о прибыли', 'прибылях и убытках', 'процентные расходы'],
            'capex': ['отчет о движении денежных средств', 'приобретение основных средств'],
            'dividends': ['отчет о движении денежных средств', 'дивиденды']
        }
        
        self.page_dates = {}  # Store dates found on each page


    def _wait_for_rate_limit(self):
        """Ensure minimum time between requests."""
        elapsed = time.time() - self.last_request_time
        if elapsed < self.min_request_interval:
            time.sleep(self.min_request_interval - elapsed)
        self.last_request_time = time.time()

    def _parse_llm_response(self, response: str, metric: str) -> Dict:
        """Safely parse LLM response to JSON with better cleaning."""
        print(f"\nDebug - Raw LLM response for {metric}:")
        print(response)
        
        try:
            # Clean the response string
            cleaned_response = response.strip()
            
            # Remove markdown code blocks if present
            if cleaned_response.startswith('```') and cleaned_response.endswith('```'):
                cleaned_response = cleaned_response[3:-3]
            if cleaned_response.startswith('```json') and cleaned_response.endswith('```'):
                cleaned_response = cleaned_response[7:-3]
            
            # Remove 'json' if it's at the start
            if cleaned_response.startswith('json'):
                cleaned_response = cleaned_response[4:]
            
            # Clean up any weird formatting
            cleaned_response = cleaned_response.strip()
            cleaned_response = re.sub(r'\s+', ' ', cleaned_response)  # Replace multiple spaces
            cleaned_response = re.sub(r',\s*}', '}', cleaned_response)  # Remove trailing commas
            cleaned_response = re.sub(r',\s*]', ']', cleaned_response)  # Remove trailing commas in arrays
            
            # Remove any non-JSON text before or after
            start_idx = cleaned_response.find('{')
            end_idx = cleaned_response.rfind('}') + 1
            if start_idx != -1 and end_idx != 0:
                cleaned_response = cleaned_response[start_idx:end_idx]
            
            print("\nDebug - Cleaned response:")
            print(cleaned_response)
            
            try:
                result = json.loads(cleaned_response)
            except json.JSONDecodeError:
                # Try replacing single quotes with double quotes
                cleaned_response = cleaned_response.replace("'", '"')
                result = json.loads(cleaned_response)
            
            print("\nDebug - Parsed JSON:")
            print(json.dumps(result, indent=2))
            
            return result
        except json.JSONDecodeError as e:
            print(f"\nDebug - JSON parse error: {e}")
            print(f"Failed response was: {response}")
            if metric == "date_extraction":
                return {
                    "statement_date": None,
                    "period_type": None,
                    "period_description": None,
                    "is_comparative": False
                }
            else:
                return {
                    'reported': {'value': None, 'date': None},
                    'comparative': {'value': None, 'date': None}
                }
        except Exception as e:
            print(f"\nDebug - Error parsing response: {e}")
            if metric == "date_extraction":
                return {
                    "statement_date": None,
                    "period_type": None,
                    "period_description": None,
                    "is_comparative": False
                }
            else:
                return {
                    'reported': {'value': None, 'date': None},
                    'comparative': {'value': None, 'date': None}
                }

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=4, max=60),
        reraise=True
    )
    def extract_pdf_text(self) -> List[str]:
        """Extract text from PDF using OCR."""
        print("\nDebug - Starting PDF text extraction")
        images = pdf2image.convert_from_path(self.pdf_path, first_page=1, last_page=10)
        
        for image in images:
            text = pytesseract.image_to_string(image, lang='rus')
            self.pages_text.append(text)
        
        print(f"\nDebug - Extracted {len(self.pages_text)} pages")
        for i, page in enumerate(self.pages_text, 1):
            print(f"\nPage {i}:")
            print(page[:500] + "...")  # Print first 500 chars of each page
        
        return self.pages_text

    def identify_statements(self):
        """Identify and extract full financial statements from the document."""
        print("\nDebug - Identifying financial statements")
        statement_markers = {
            'balance_sheet': [
                'отчет о финансовом положении',
                'бухгалтерский баланс'
            ],
            'income_statement': [
                'отчет о прибыли',
                'отчет о прибылях и убытках'
            ],
            'cash_flow': [
                'отчет о движении денежных средств'
            ]
        }
        
        for page_num, text in enumerate(self.pages_text):
            for statement_type, markers in statement_markers.items():
                if any(marker.lower() in text.lower() for marker in markers):
                    # Extract that statement's full text (current page plus next page)
                    statement_text = text
                    if page_num + 1 < len(self.pages_text):
                        statement_text += "\n" + self.pages_text[page_num + 1]
                    self.statements[statement_type] = statement_text
                    
                    # Try to extract statement date
                    date_match = re.search(r'на .*?(\d{1,2})\.(\d{1,2})\.(\d{4})', text)
                    if date_match:
                        self.statement_dates[statement_type] = f"{date_match.group(3)}-{date_match.group(2)}-{date_match.group(1)}"
                    
                    print(f"\nDebug - Found {statement_type} on page {page_num + 1}")
                    print(f"Debug - Date: {self.statement_dates.get(statement_type)}")



    def find_metric_page(self, metric: str) -> Tuple[int, Dict[str, Any]]:
        """Find the most relevant page for a given metric and extract its dates."""
        identifiers = self.metric_identifiers[metric]
        
        best_page = None
        max_score = 0
        date_info = None
        
        for page_num, text in enumerate(self.pages_text):
            text_lower = text.lower()
            
            # Skip table of contents pages
            if "содержание" in text_lower and page_num < 3:
                continue
                
            score = 0
            
            # Check for statement type indicators first
            if metric in ['total_assets', 'total_equity', 'cash'] and 'отчет о финансовом положении' in text_lower:
                score += 10
            elif metric in ['revenue', 'operating_profit', 'net_profit'] and 'отчет о прибыли' in text_lower:
                score += 10
            elif metric in ['capex', 'dividends'] and 'отчет о движении денежных средств' in text_lower:
                score += 10
            
            # Check for specific metric indicators
            for identifier in identifiers:
                if identifier.lower() in text_lower:
                    score += 5
                    if re.search(rf"{identifier.lower()}.{{0,30}}\d", text_lower):
                        score += 10
            
            if score > max_score:
                max_score = score
                best_page = page_num
                # Extract dates from the best matching page
                if score > 0:
                    date_info = self.extract_date_with_llm(text)
        
        print(f"\nDebug - For metric '{metric}' found best page: {best_page + 1 if best_page is not None else None} with score {max_score}")
        print(f"Debug - Date info: {date_info}")
        return best_page, date_info

    def extract_date_with_llm(self, page_text: str) -> Dict[str, Optional[str]]:
        """Extract reporting dates from page text using LLM."""
        print("\nDebug - Extracting dates with LLM")
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are extracting dates from Russian IFRS statements.
            Return EXACTLY this JSON format (copy exactly, only change values):
            {"statement_date":"2023-12-31","period_type":"12M","period_description":"Year ended 31 December 2023","is_comparative":false}

            Common Russian date patterns:
            - "по состоянию на 31 декабря 2023" → "2023-12-31"
            - "на 31 декабря 2023 года" → "2023-12-31"
            - "за 2023 год" → period_type "12M"
            - "за год, закончившийся 31 декабря 2023" → "2023-12-31" and "12M"

            Use null for missing values but keep the exact JSON structure."""),
            ("user", f"Extract dates from this statement page:\n{page_text[:500]}")
        ])

        try:
            response = self.llm(prompt.format_messages())
            print(f"\nDebug - Date extraction raw response: {response.content}")
            result = json.loads(response.content.strip())
            print(f"\nDebug - Date extraction parsed result: {result}")
            return result
        except Exception as e:
            print(f"\nDebug - Error extracting dates: {str(e)}")
            return {
                "statement_date": None,
                "period_type": None,
                "period_description": None,
                "is_comparative": False
            }

    def extract_figure_with_llm(self, metric: str) -> Dict[str, Optional[float]]:
        """Extract specific financial metric using LLM with enhanced Russian statement understanding."""
        page_num, date_info = self.find_metric_page(metric)
        if page_num is None:
            print(f"Debug - Could not find relevant page for {metric}")
            return {'reported': {'value': None, 'date': None},
                   'comparative': {'value': None, 'date': None}}

        context = self.pages_text[page_num]
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are extracting financial metrics from Russian IFRS statements.
            Return EXACTLY this JSON format (copy exactly, only change values):
            {"reported":{"value":60904.0,"date":"2023-12-31"},"comparative":{"value":null,"date":null}}

            Number format rules:
            - "60 904" → 60904.0
            - "(4 041)" → -4041.0
            - Numbers are in millions (млн руб.)
            - Ignore "Прим." or note numbers
            
            Real examples:
            - "Основные средства 16 60 904" → 60904.0
            - "Добавочный капитал (4 041)" → -4041.0
            
            Always keep the exact JSON structure, just change the values."""),
            ("user", f"""Find this exact metric: {metric}
Current context:
{context}""")
        ])

        try:
            response = self.llm(prompt.format_messages())
            print(f"\nDebug - Metric extraction raw response: {response.content}")
            result = json.loads(response.content.strip())
            print(f"\nDebug - Metric extraction parsed result: {result}")
            return result
        except Exception as e:
            print(f"Debug - Error in LLM extraction for {metric}: {str(e)}")
            return {
                'reported': {'value': None, 'date': None},
                'comparative': {'value': None, 'date': None}
            }

    def analyze_statements(self) -> pd.DataFrame:
        """Analyze IFRS statements with page-based context."""
        try:
            print("\nDebug - Starting analysis")
            self.extract_pdf_text()
            
            # Extract dates from each page
            for page_num, text in enumerate(self.pages_text):
                date = self.extract_date_with_llm(text)
                if date:
                    self.page_dates[page_num] = date
                    print(f"\nDebug - Found date on page {page_num + 1}: {date}")
            
            metrics = [
                'total_debt', 'revenue', 'interest_expense', 'total_equity',
                'total_assets', 'capex', 'dividends', 'operating_profit',
                'ebitda', 'cash', 'net_profit'
            ]

            data = {
                'Metric': [],
                'Date': [],
                'Value': [],
                'Comparative Date': [],
                'Comparative Value': []
            }

            for metric in metrics:
                result = self.extract_figure_with_llm(metric)
                reported = result.get('reported', {})
                comparative = result.get('comparative', {})
                
                data['Metric'].append(metric)
                data['Date'].append(reported.get('date'))
                data['Value'].append(reported.get('value'))
                data['Comparative Date'].append(comparative.get('date'))
                data['Comparative Value'].append(comparative.get('value'))

            return pd.DataFrame(data)
            
        except Exception as e:
            print(f"\nDebug - Error in analyze_statements: {str(e)}")
            raise

    @staticmethod
    def format_value(value: Optional[float]) -> str:
        """Format currency values with millions/billions notation."""
        if value is None:
            return "N/A"
        if abs(value) >= 1000:
            return f"{value/1000:.2f}B"
        return f"{value:.2f}M"