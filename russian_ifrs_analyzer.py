from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.schema import Document
import pandas as pd
from typing import List, Dict, Optional
import os

import pdf2image
import pytesseract
from pathlib import Path



class RussianIFRSAnalyzer:
    def __init__(self, pdf_path: str, openai_api_key: str):
        self.pdf_path = pdf_path
        self.openai_api_key = openai_api_key
        os.environ['OPENAI_API_KEY'] = openai_api_key
        
        # Initialize LLM and embeddings
        self.llm = ChatOpenAI(temperature=0, model="gpt-4")
        self.embeddings = OpenAIEmbeddings()
        
        # Initialize storage for extracted content
        self.pages_text = []
        self.vector_store = None
        self.extracted_figures = {}

    def extract_pdf_text(self) -> List[str]:
        """Extract text from PDF using OCR."""
        # Convert PDF to images
        images = pdf2image.convert_from_path(self.pdf_path)
        
        # Process each page with OCR
        for image in images:
            text = pytesseract.image_to_string(image, lang='rus')
            self.pages_text.append(text)
        
        return self.pages_text


    def create_vector_store(self):
        """Create FAISS vector store from PDF content."""
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=10000,
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

        self.vector_store = FAISS.from_documents(documents, self.embeddings)

    def get_relevant_context(self, query: str, k: int = 3) -> str:
        """Retrieve relevant context for a specific financial metric."""
        docs = self.vector_store.similarity_search(query, k=k)
        return "\n".join([doc.page_content for doc in docs])

    def extract_figure_with_llm(self, metric: str) -> Dict[str, Optional[float]]:
        """Extract specific financial metric using LLM."""
        # Prepare search queries in Russian
        metric_queries = {
            'total_debt': 'общий долг OR общая задолженность OR кредиты и займы',
            'revenue': 'выручка OR доходы от реализации',
            'interest_expense': 'процентные расходы OR финансовые расходы',
            'total_equity': 'собственный капитал OR капитал и резервы',
            'total_assets': 'всего активов OR итого активы',
            'capex': 'капитальные затраты OR инвестиции в основные средства',
            'dividends': 'дивиденды выплаченные',
            'operating_profit': 'операционная прибыль',
            'ebitda': 'EBITDA OR прибыль до вычета',
        }

        query = metric_queries.get(metric, metric)
        context = self.get_relevant_context(query)

        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a financial analyst expert in IFRS statements. 
            Extract the numerical values for the specified metric from the given context.
            The context is in Russian. Return ONLY a JSON with two keys:
            'current_year': [value as float or null],
            'previous_year': [value as float or null]
            
            Convert all numbers to millions. If number is in billions, multiply by 1000.
            If number is in thousands, divide by 1000.
            Handle negative values appropriately (numbers in parentheses are negative).
            If you can't find the value, return null."""),
            ("user", f"Metric to extract: {metric}\n\nContext:\n{context}")
        ])

        response = self.llm(prompt.format_messages())
        try:
            return eval(response.content)
        except:
            return {'current_year': None, 'previous_year': None}

    def calculate_net_debt(self) -> Dict[str, Optional[float]]:
        """Calculate Net Debt from Total Debt and Cash positions."""
        total_debt = self.extracted_figures.get('total_debt', {})
        cash = self.extract_figure_with_llm('денежные средства и эквиваленты')
        
        return {
            'current_year': (
                total_debt.get('current_year') - cash.get('current_year')
                if total_debt.get('current_year') is not None and cash.get('current_year') is not None
                else None
            ),
            'previous_year': (
                total_debt.get('previous_year') - cash.get('previous_year')
                if total_debt.get('previous_year') is not None and cash.get('previous_year') is not None
                else None
            )
        }

    def analyze_statements(self) -> pd.DataFrame:
        """Main method to analyze the IFRS statements and extract all required figures."""
        # Extract text and create vector store
        self.extract_pdf_text()
        self.create_vector_store()

        # List of metrics to extract
        metrics = [
            'total_debt',
            'revenue',
            'interest_expense',
            'total_equity',
            'total_assets',
            'capex',
            'dividends',
            'operating_profit',
            'ebitda'
        ]

        # Extract each metric
        for metric in metrics:
            self.extracted_figures[metric] = self.extract_figure_with_llm(metric)

        # Calculate Net Debt
        self.extracted_figures['net_debt'] = self.calculate_net_debt()

        # Create DataFrame for reporting
        data = {
            'Metric': [],
            'Current Year': [],
            'Previous Year': []
        }

        for metric, values in self.extracted_figures.items():
            data['Metric'].append(metric)
            data['Current Year'].append(values['current_year'])
            data['Previous Year'].append(values['previous_year'])

        return pd.DataFrame(data)

def format_currency(value: Optional[float]) -> str:
    """Format currency values with millions/billions notation."""
    if value is None:
        return "N/A"
    if abs(value) >= 1000:
        return f"{value/1000:.2f}B"
    return f"{value:.2f}M"