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

class RussianIFRSAnalyzer:
    def __init__(self, pdf_path: str, openai_api_key: str, max_retries: int = 5):
        # [Previous initialization code remains the same]
        self.statements = {
            'balance_sheet': None,
            'income_statement': None,
            'cash_flow': None
        }
        self.statement_dates = {}
        
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

    def identify_statements(self):
        """Identify and extract full financial statements from the document."""
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
        
        # First, try to identify the pages containing each statement
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

    def extract_figure_with_llm(self, metric: str) -> Dict[str, Optional[float]]:
        """Extract specific financial metric using LLM with statement context."""
        print(f"\nDebug - Processing metric: {metric}")
        
        # Get the appropriate statement context
        statement_type = self.metric_locations.get(metric)
        if not statement_type or not self.statements.get(statement_type):
            print(f"Debug - Could not find appropriate statement for {metric}")
            return {'reported': {'value': None, 'date': None},
                   'comparative': {'value': None, 'date': None}}

        # Get the statement context
        context = self.statements[statement_type]
        statement_date = self.statement_dates.get(statement_type)
        
        print(f"\nDebug - Using {statement_type} statement context for {metric}")
        print(f"Debug - Statement date: {statement_date}")
        print(f"Debug - Context excerpt: {context[:500]}...")

        self._wait_for_rate_limit()

        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a financial analyst expert in IFRS statements.
            Extract the numerical value for the specified metric from the given financial statement.
            The text is in Russian. Return ONLY a JSON with two keys:
            'reported': {'value': float or null, 'date': 'YYYY-MM-DD'},
            'comparative': {'value': float or null, 'date': 'YYYY-MM-DD'}
            
            Important rules:
            - The statement date is provided - use it for the 'date' field
            - Look for the exact metric, not similar ones
            - Numbers are in millions of rubles
            - Convert any billions to millions (multiply by 1000)
            - Handle negative values (numbers in parentheses)
            - If exact value isn't found, return null"""),
            ("user", f"""Metric to extract: {metric}
Statement type: {statement_type}
Statement date: {statement_date}

Context:
{context}""")
        ])

        try:
            response = self.llm(prompt.format_messages())
            return self._parse_llm_response(response.content, metric)
        except Exception as e:
            print(f"Debug - Error in LLM extraction for {metric}: {str(e)}")
            return {
                'reported': {'value': None, 'date': None},
                'comparative': {'value': None, 'date': None}
            }

    def analyze_statements(self) -> pd.DataFrame:
        """Analyze IFRS statements with statement-based context."""
        try:
            print("\nDebug - Starting analysis")
            self.extract_pdf_text()
            
            # First identify and extract the statements
            print("\nDebug - Identifying financial statements")
            self.identify_statements()
            
            # Print found statements
            for statement_type, text in self.statements.items():
                if text:
                    print(f"\nDebug - Found {statement_type}")
                    print(f"Debug - Date: {self.statement_dates.get(statement_type)}")
                else:
                    print(f"\nDebug - Could not find {statement_type}")

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