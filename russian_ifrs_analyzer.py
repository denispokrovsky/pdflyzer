import json
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.schema import Document
import pandas as pd
from typing import List, Dict, Optional, Any
from tenacity import retry, stop_after_attempt, wait_exponential
import openai
import time
import pdf2image
import pytesseract
import os
from datetime import datetime
import re

class RussianIFRSAnalyzer:
    def __init__(self, pdf_path: str, openai_api_key: str, max_retries: int = 5):
        self.pdf_path = pdf_path
        self.openai_api_key = openai_api_key
        self.max_retries = max_retries
        os.environ['OPENAI_API_KEY'] = openai_api_key
        
        self.llm = ChatOpenAI(
            temperature=0,
            model="gpt-4",
            max_retries=max_retries,
            request_timeout=30
        )
        
        self.embeddings = OpenAIEmbeddings(
            max_retries=max_retries,
            request_timeout=30
        )
        
        self.pages_text = []
        self.statements = {
            'balance_sheet': None,
            'income_statement': None,
            'cash_flow': None
        }
        self.statement_dates = {}
        
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
        
        self.pages_text = []
        self.page_dates = {}  # Store dates found on each page
    

    def _wait_for_rate_limit(self):
        """Ensure minimum time between requests."""
        elapsed = time.time() - self.last_request_time
        if elapsed < self.min_request_interval:
            time.sleep(self.min_request_interval - elapsed)
        self.last_request_time = time.time()

    def _parse_llm_response(self, response: str, metric: str) -> Dict:
        """Safely parse LLM response to JSON."""
        print(f"\nDebug - Raw LLM response for {metric}:")
        print(response)
        
        try:
            # Clean the response string
            cleaned_response = response.strip()
            if cleaned_response.startswith('```') and cleaned_response.endswith('```'):
                cleaned_response = cleaned_response[3:-3]
            if cleaned_response.startswith('json'):
                cleaned_response = cleaned_response[4:]
            
            print("\nDebug - Cleaned response:")
            print(cleaned_response)
            
            result = json.loads(cleaned_response)
            print("\nDebug - Parsed JSON:")
            print(json.dumps(result, indent=2))
            
            return result
        except json.JSONDecodeError as e:
            print(f"\nDebug - JSON parse error: {e}")
            print(f"Failed response was: {response}")
            return {}
        except Exception as e:
            print(f"\nDebug - Error parsing response: {e}")
            return {}

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

def extract_date_with_llm(self, page_text: str) -> Dict[str, Optional[str]]:
        """Extract reporting dates from page text using LLM."""
        print("\nDebug - Extracting dates with LLM")
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a financial expert analyzing Russian IFRS statements.
            Extract the reporting date and period type from the given page.
            Return ONLY a JSON object with this structure:
            {
                "statement_date": "YYYY-MM-DD",  # Main reporting date
                "period_type": "3M"/"6M"/"9M"/"12M",  # Reporting period
                "period_description": string,  # Description like "Year ended 31 December 2023"
                "is_comparative": boolean  # If this page shows comparative figures
            }
            
            Look for:
            - "на [date]" for balance sheet dates
            - "за [period] [year]" for income statement periods
            - Standard dates like "31 декабря 2023"
            - Period indicators like "за год", "за 6 месяцев"
            
            If date/period not found, use null for that field."""),
            ("user", f"Extract reporting date and period from this page:\n\n{page_text[:1000]}")
        ])

        try:
            self._wait_for_rate_limit()
            response = self.llm(prompt.format_messages())
            result = self._parse_llm_response(response.content, "date_extraction")
            print(f"\nDebug - Extracted dates: {result}")
            return result
        except Exception as e:
            print(f"\nDebug - Error extracting dates: {str(e)}")
            return {
                "statement_date": None,
                "period_type": None,
                "period_description": None,
                "is_comparative": False
            }

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

    def extract_figure_with_llm(self, metric: str) -> Dict[str, Optional[float]]:
        """Extract specific financial metric using LLM with date context."""
        print(f"\nDebug - Processing metric: {metric}")
        
        page_num, date_info = self.find_metric_page(metric)
        if page_num is None:
            print(f"Debug - Could not find relevant page for {metric}")
            return {'reported': {'value': None, 'date': None},
                   'comparative': {'value': None, 'date': None}}

        context = self.pages_text[page_num]
        
        print(f"\nDebug - Using page {page_num + 1} for {metric}")
        print(f"Debug - Date info: {date_info}")
        print(f"Debug - Context excerpt: {context[:500]}...")

        self._wait_for_rate_limit()

        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a financial analyst expert in IFRS statements.
            Find and extract numerical value for the EXACT specified metric from the given page.
            The text is in Russian. Return ONLY a JSON object in this format:
            {
                "reported": {"value": float, "date": "YYYY-MM-DD"},
                "comparative": {"value": float, "date": "YYYY-MM-DD"}
            }
            
            Rules:
            1. Numbers are in millions of rubles (млн руб.)
            2. If you see a number like 60 904, it means 60,904 million rubles
            3. Numbers in parentheses (123) are negative values
            4. Look for EXACT metric match, not similar ones
            5. If value isn't clearly stated, use null
            6. For values with 'Прим.' or 'Note', use the actual value, not the note number
            7. Use provided statement_date for the reporting period"""),
            ("user", f"""Metric to extract: {metric}
Statement date: {date_info.get('statement_date') if date_info else None}
Period type: {date_info.get('period_type') if date_info else None}
Period description: {date_info.get('period_description') if date_info else None}
Has comparative figures: {date_info.get('is_comparative') if date_info else False}

Look for exactly this value in the financial statement page.
For balance sheet items, look for direct line items.
For income statement items, look for the specific profit/revenue line.
For cash flow items, look in the cash flow section.

Context:
{context}""")
        ])

        try:
            response = self.llm(prompt.format_messages())
            result = self._parse_llm_response(response.content, metric)
            
            # Use date info if available
            if date_info and date_info.get('statement_date'):
                if 'reported' in result:
                    result['reported']['date'] = date_info['statement_date']
                
                # For comparative date, use previous year with same period end
                if 'comparative' in result and result['comparative'].get('value') is not None:
                    try:
                        current_date = datetime.strptime(date_info['statement_date'], '%Y-%m-%d')
                        comparative_date = current_date.replace(year=current_date.year - 1)
                        result['comparative']['date'] = comparative_date.strftime('%Y-%m-%d')
                    except:
                        pass
            
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
                date = self.extract_date_from_page(text)
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