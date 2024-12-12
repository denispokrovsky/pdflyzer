from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.schema import Document
import pandas as pd
from typing import List, Dict, Optional
from tenacity import retry, stop_after_attempt, wait_exponential
import openai
import time
import pdf2image
import pytesseract
import os

class RussianIFRSAnalyzer:
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
        
        self.last_request_time = 0
        self.min_request_interval = 1.25

    def _wait_for_rate_limit(self):
        """Ensure minimum time between requests."""
        elapsed = time.time() - self.last_request_time
        if elapsed < self.min_request_interval:
            time.sleep(self.min_request_interval - elapsed)
        self.last_request_time = time.time()

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=4, max=60),
        reraise=True
    )
    def extract_pdf_text(self) -> List[str]:
        """Extract text from PDF using OCR."""
        images = pdf2image.convert_from_path(self.pdf_path, first_page=1, last_page=10)
        
        for image in images:
            text = pytesseract.image_to_string(image, lang='rus')
            self.pages_text.append(text)
        
        return self.pages_text

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=4, max=60),
        reraise=True
    )
    def create_vector_store(self):
        """Create FAISS vector store from PDF content."""
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=8000,
            chunk_overlap=100
        )
        
        documents = []
        for page_num, text in enumerate(self.pages_text):
            chunks = text_splitter.split_text(text)
            for chunk in chunks:
                doc = Document(
                    page_content=chunk,
                    metadata={"page": page_num + 1}
                )
                documents.append(doc)

        self._wait_for_rate_limit()
        self.vector_store = FAISS.from_documents(documents, self.embeddings)

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=4, max=60),
        reraise=True
    )
    def get_relevant_context(self, query: str, k: int = 2) -> str:
        """Retrieve relevant context for a specific financial metric."""
        self._wait_for_rate_limit()
        docs = self.vector_store.similarity_search(query, k=k)
        return "\n".join([doc.page_content for doc in docs])

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=4, max=60),
        reraise=True
    )
    def extract_figure_with_llm(self, metric: str) -> Dict[str, Optional[float]]:
        """Extract specific financial metric using LLM with rate limiting."""
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
        context = self.get_relevant_context(query)

        self._wait_for_rate_limit()

        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a financial analyst expert in IFRS statements.
            Extract the numerical values for the specified metric from the given context.
            The context is in Russian. Return ONLY a JSON with two keys:
            'reported_value': [value as float or null],
            'comparative_value': [value as float or null]
            
            Important:
            - Convert all numbers to millions
            - If number is in billions, multiply by 1000
            - If number is in thousands, divide by 1000
            - Handle negative values appropriately (numbers in parentheses are negative)
            - If you can't find a value or are unsure, return null
            - For reported_value, use the main reported figure
            - For comparative_value, use the comparison figure if available
            - If only one value is present, set comparative_value to null
            - For balance sheet items, use the values as stated at the reporting date
            - For income statement and cash flow items, use the values for the reported period"""),
            ("user", f"Metric to extract: {metric}\n\nContext:\n{context}")
        ])

        try:
            response = self.llm(prompt.format_messages())
            return eval(response.content)
        except (openai.RateLimitError, openai.APIError) as e:
            print(f"Rate limit error for metric {metric}, retrying: {str(e)}")
            raise
        except Exception as e:
            print(f"Error processing metric {metric}: {str(e)}")
            return {'reported_value': None, 'comparative_value': None}

    def analyze_statements(self) -> pd.DataFrame:
        """Analyze IFRS statements and extract metrics."""
        try:
            self.extract_pdf_text()
            self.create_vector_store()

            metrics = [
                'total_debt',
                'revenue',
                'interest_expense',
                'total_equity',
                'total_assets',
                'capex',
                'dividends',
                'operating_profit',
                'ebitda',
                'cash',
                'net_profit'
            ]

            data = {
                'Metric': [],
                'Reported Value': [],
                'Comparative Value': []
            }

            for metric in metrics:
                self._wait_for_rate_limit()
                values = self.extract_figure_with_llm(metric)
                data['Metric'].append(metric)
                data['Reported Value'].append(values.get('reported_value'))
                data['Comparative Value'].append(values.get('comparative_value'))

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