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

class RussianFinancialExtractor:
    def __init__(self, pdf_path: str, openai_api_key: str, max_retries: int = 5):
        self.pdf_path = pdf_path
        self.openai_api_key = openai_api_key
        self.max_retries = max_retries
        os.environ['OPENAI_API_KEY'] = openai_api_key
        
        self.llm = ChatOpenAI(
            temperature=0,
            model="gpt-4o-mini",
            max_retries=max_retries,
            request_timeout=30
        )
        
        self.embeddings = OpenAIEmbeddings(
            max_retries=max_retries,
            request_timeout=30
        )
        
        self.pages_text = []
        self.vector_store = None
        self.extracted_figures = {}
        self.reporting_periods = None
        
        self.last_request_time = 0
        self.min_request_interval = 1.25
    
    def extract_pdf_text(self) -> List[str]:
        """Extract text from PDF using OCR."""
        images = pdf2image.convert_from_path(self.pdf_path, first_page=1, last_page=10)
        
        for image in images:
            text = pytesseract.image_to_string(image, lang='rus')
            self.pages_text.append(text)
        
        return self.pages_text
    

    def _wait_for_rate_limit(self):
        """Ensure minimum time between requests."""
        elapsed = time.time() - self.last_request_time
        if elapsed < self.min_request_interval:
            time.sleep(self.min_request_interval - elapsed)
        self.last_request_time = time.time()


    def create_vector_store(self):
        """Create FAISS vector store from PDF content."""
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=8000,
            chunk_overlap=100
        )
        
        documents = []
        for page_num, text in enumerate(self.pages_text, 1):
            # Add more metadata about the page content
            is_header = page_num <= 2  # First two pages usually contain header info
            is_financial_statement = any(marker in text.lower() for marker in [
                'отчет о финансовом положении',
                'бухгалтерский баланс',
                'отчет о прибылях',
                'отчет о движении денежных средств'
            ])
            
            chunks = text_splitter.split_text(text)
            for chunk in chunks:
                doc = Document(
                    page_content=chunk,
                    metadata={
                        "page": page_num,
                        "is_header": is_header,
                        "is_financial_statement": is_financial_statement
                    }
                )
                documents.append(doc)

        self.vector_store = FAISS.from_documents(documents, self.embeddings)

    def get_relevant_context(self, query: str, k: int = 2, filter_dict: Optional[Dict] = None) -> str:
        """Retrieve relevant context with optional metadata filtering."""
        self._wait_for_rate_limit()
        search_kwargs = {"k": k}
        if filter_dict:
            search_kwargs["filter"] = filter_dict
            
        docs = self.vector_store.similarity_search(query, **search_kwargs)
        return "\n".join([doc.page_content for doc in docs])

    def extract_reporting_periods(self) -> Dict[str, Any]:
        """Extract reporting period information using RAG."""
        # First, get context from header pages
        header_context = self.get_relevant_context(
            "дата отчет отчетность период",
            k=3,
            filter_dict={"is_header": True}
        )
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a financial analyst expert in IFRS statements.
            Analyze the given context and identify the reporting periods mentioned.
            The context is in Russian. Return ONLY a JSON with this structure:
            {
                "statement_date": "YYYY-MM-DD",
                "period_type": "3M"/"6M"/"9M"/"12M",
                "comparative_date": "YYYY-MM-DD" or null,
                "statement_type": "interim"/"annual",
                "year": YYYY
            }
            
            Look for:
            - The main reporting date
            - Whether it's an interim or annual report
            - The comparative period date if mentioned
            - The length of the reporting period (3,6,9,12 months)"""),
            ("user", f"Context:\n{header_context}")
        ])

        self._wait_for_rate_limit()
        response = self.llm(prompt.format_messages())
        return eval(response.content)

    def extract_figure_with_llm(self, metric: str) -> Dict[str, Any]:
        """Extract specific financial metric using LLM with rate limiting."""
        # First get the statement type context
        statement_type = "balance" if metric in [
            'total_debt', 'total_equity', 'total_assets', 'cash'
        ] else "income"
        
        filter_dict = {"is_financial_statement": True}
        
        metric_queries = {
            'total_debt': 'долгосрочные заемные средства OR краткосрочные заемные средства',
            'revenue': 'выручка от реализации OR доходы от реализации',
            'interest_expense': 'процентные расходы OR расходы в виде процентов',
            'total_equity': 'итого капитал',
            'total_assets': 'итого активы',
            'capex': 'капитальные затраты OR приобретение основных средств',
            'dividends': 'дивиденды выплаченные',
            'operating_profit': 'прибыль от операционной деятельности',
            'ebitda': 'EBITDA OR прибыль до вычета процентов налогов и амортизации',
            'cash': 'денежные средства и их эквиваленты',
            'net_profit': 'прибыль OR чистая прибыль'
        }

        query = metric_queries.get(metric, metric)
        context = self.get_relevant_context(query, k=2, filter_dict=filter_dict)

        # Include reporting period information in the context
        if self.reporting_periods:
            context = f"Report date: {self.reporting_periods['statement_date']}\nPeriod type: {self.reporting_periods['period_type']}\n\n{context}"

        self._wait_for_rate_limit()

        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a financial analyst expert in IFRS statements.
            Extract the numerical values for the specified metric from the given context.
            The context is in Russian. Consider the reporting date and period provided.
            Return ONLY a JSON with this structure:
            {
                "reported": {
                    "value": float or null,
                    "date": "YYYY-MM-DD",
                    "period_type": "3M"/"6M"/"9M"/"12M" or null
                },
                "comparative": {
                    "value": float or null,
                    "date": "YYYY-MM-DD",
                    "period_type": "3M"/"6M"/"9M"/"12M" or null
                }
            }"""),
            ("user", f"Metric to extract: {metric}\nStatement type: {statement_type}\n\nContext:\n{context}")
        ])

        try:
            response = self.llm(prompt.format_messages())
            return eval(response.content)
        except Exception as e:
            print(f"Error processing metric {metric}: {str(e)}")
            return {
                "reported": {"value": None, "date": None, "period_type": None},
                "comparative": {"value": None, "date": None, "period_type": None}
            }

    def analyze_statements(self) -> pd.DataFrame:
        """Analyze IFRS statements using RAG."""
        try:
            # Extract text and create vector store
            self.extract_pdf_text()
            self.create_vector_store()
            
            # First, extract reporting period information
            self.reporting_periods = self.extract_reporting_periods()

            metrics = [
                'total_debt', 'revenue', 'interest_expense', 'total_equity',
                'total_assets', 'capex', 'dividends', 'operating_profit',
                'ebitda', 'cash', 'net_profit'
            ]

            data = {
                'Metric': [],
                'Date': [],
                'Period Type': [],
                'Value': [],
                'Comparative Date': [],
                'Comparative Period Type': [],
                'Comparative Value': []
            }

            for metric in metrics:
                self._wait_for_rate_limit()
                result = self.extract_figure_with_llm(metric)
                
                data['Metric'].append(metric)
                data['Date'].append(result['reported']['date'])
                data['Period Type'].append(result['reported']['period_type'])
                data['Value'].append(result['reported']['value'])
                data['Comparative Date'].append(result['comparative']['date'])
                data['Comparative Period Type'].append(result['comparative']['period_type'])
                data['Comparative Value'].append(result['comparative']['value'])

            return pd.DataFrame(data)
            
        except Exception as e:
            print(f"Error in analyze_statements: {str(e)}")
            raise

    @staticmethod
    def format_value(value: Optional[float]) -> str:
        """Format currency values with millions/billions notation."""
        if value is None:
            return "N/A"
        if abs(value) >= 1000:
            return f"{value/1000:.2f}B"
        return f"{value:.2f}M"