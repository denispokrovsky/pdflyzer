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
        self._wait_for_rate_limit()
        docs = self.vector_store.similarity_search(query, k=k)
        return "\n".join([doc.page_content for doc in docs])

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=4, max=60),
        reraise=True
    )
    def extract_figure_with_llm(self, metric: str) -> Dict[str, Dict[str, Optional[float]]]:
        """Extract specific financial metric using LLM with rate limiting."""
        metric_queries = {
            'total_debt': 'долгосрочные заемные средства',
            'revenue': 'выручка от реализации',
            'interest_expense': 'процентные расходы',
            'total_equity': 'итого капитал',
            'total_assets': 'итого активы',
            'capex': 'капитальные затраты',
            'dividends': 'дивиденды',
            'operating_profit': 'прибыль от операционной деятельности',
            'ebitda': 'EBITDA',
            'cash': 'денежные средства и их эквиваленты',
            'net_profit': 'прибыль'
        }

        query = metric_queries.get(metric, metric)
        context = self.get_relevant_context(query)

        self._wait_for_rate_limit()

        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a financial analyst expert in IFRS statements.
            First, identify the periods in the statement (e.g., "31 December 2023", "3 months ended 31 March 2023", etc.).
            Then extract the numerical values for the specified metric for each period found.
            
            Return ONLY a JSON with the following structure:
            {
                "periods": [
                    {
                        "date": "YYYY-MM-DD",
                        "period_type": "3M/6M/9M/12M",
                        "value": float or null
                    },
                    ...
                ]
            }
            
            Important:
            - Convert all numbers to millions
            - If number is in billions, multiply by 1000
            - If number is in thousands, divide by 1000
            - Handle negative values appropriately (numbers in parentheses are negative)
            - If you can't find a value or are unsure for a period, use null
            - Include all periods mentioned in the document for this metric
            - For balance sheet items, include the date
            - For income statement items, include both the period type and end date"""),
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
            return {"periods": []}

    def analyze_statements(self) -> pd.DataFrame:
        """Analyze IFRS statements with flexible period handling."""
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

            # Extract each metric
            data = []
            for metric in metrics:
                self._wait_for_rate_limit()
                result = self.extract_figure_with_llm(metric)
                for period in result.get('periods', []):
                    data.append({
                        'Metric': metric,
                        'Date': period['date'],
                        'Period Type': period['period_type'],
                        'Value': period['value']
                    })

            return pd.DataFrame(data)
            
        except Exception as e:
            print(f"Error in analyze_statements: {str(e)}")
            raise

    @staticmethod
    def format_value(value: Optional[float]) -> str:
        if value is None:
            return "N/A"
        if abs(value) >= 1000:
            return f"{value/1000:.2f}B"
        return f"{value:.2f}M"