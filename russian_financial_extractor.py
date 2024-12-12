import pandas as pd
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import re
from datetime import datetime

@dataclass
class FinancialPeriod:
    end_date: datetime
    period_type: str  # '3M', '6M', '9M', '12M'
    year: int

class RussianFinancialExtractor:
    """Extract financial data from Russian financial statements with flexible period handling."""
    
    def __init__(self):
        self.balance_sheet_metrics = {
            'total_assets': r'Итого активы',
            'current_assets': r'Итого текущие активы',
            'non_current_assets': r'Итого долгосрочные активы',
            'total_liabilities': r'Итого обязательства',
            'current_liabilities': r'Итого текущие обязательства',
            'non_current_liabilities': r'Итого долгосрочные обязательства',
            'total_equity': r'Итого капитал',
            'cash_equivalents': r'Денежные средства и их эквиваленты'
        }
        
        self.income_statement_metrics = {
            'revenue': r'Итого выручка от реализации',
            'operating_profit': r'Прибыль от операционной деятельности',
            'profit_before_tax': r'Прибыль до налога на прибыль',
            'net_profit': r'Прибыль',
            'ebitda': r'EBITDA'  # May need custom calculation
        }

    def detect_periods(self, text: str) -> List[FinancialPeriod]:
        """Detect reporting periods from the document text."""
        periods = []
        
        # Look for date patterns in header and near financial tables
        date_patterns = [
            r'за (?:год|период).+?(\d{2}).(\d{2}).(\d{4})',
            r'на (\d{2}).(\d{2}).(\d{4})',
            r'за (\d+) месяц.+?(\d{4})'
        ]
        
        for pattern in date_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                if len(match.groups()) == 3:  # Full date pattern
                    date = datetime(int(match.group(3)), int(match.group(2)), int(match.group(1)))
                    # Determine period type based on context
                    period_type = self._determine_period_type(text, match.start())
                    periods.append(FinancialPeriod(date, period_type, date.year))
                elif len(match.groups()) == 2:  # Month count + year pattern
                    months = int(match.group(1))
                    year = int(match.group(2))
                    period_type = f"{months}M"
                    # Construct end date based on month count
                    date = datetime(year, (months % 12) or 12, 31)
                    periods.append(FinancialPeriod(date, period_type, year))
        
        return periods

    def _determine_period_type(self, text: str, position: int) -> str:
        """Determine the period type based on surrounding context."""
        context = text[max(0, position-100):position+100]
        if re.search(r'(три|3|первый).+?(месяц|квартал)', context, re.IGNORECASE):
            return '3M'
        elif re.search(r'(шесть|6|полугод)', context, re.IGNORECASE):
            return '6M'
        elif re.search(r'(девять|9)', context, re.IGNORECASE):
            return '9M'
        elif re.search(r'(двенадцать|12|год)', context, re.IGNORECASE):
            return '12M'
        return '12M'  # Default to annual if unclear

    def extract_value(self, text: str, metric_pattern: str) -> Optional[float]:
        """Extract numerical value for a given metric pattern."""
        # Look for the metric pattern followed by numbers
        value_pattern = f"{metric_pattern}.*?(\d+'?\d*'?\d*)"
        match = re.search(value_pattern, text, re.MULTILINE)
        if match:
            # Clean up the number string and convert to float
            value_str = match.group(1).replace("'", "")
            try:
                return float(value_str)
            except ValueError:
                return None
        return None

    def extract_metrics(self, text: str, metrics_dict: Dict[str, str]) -> Dict[str, float]:
        """Extract all metrics from the given dictionary."""
        results = {}
        for metric_name, pattern in metrics_dict.items():
            value = self.extract_value(text, pattern)
            if value is not None:
                results[metric_name] = value
        return results

    def process_document(self, text: str) -> Dict[str, Dict]:
        """Process the document and extract all financial data with period information."""
        periods = self.detect_periods(text)
        results = {}
        
        for period in periods:
            period_key = f"{period.period_type}_{period.year}"
            results[period_key] = {
                'period': period,
                'balance_sheet': self.extract_metrics(text, self.balance_sheet_metrics),
                'income_statement': self.extract_metrics(text, self.income_statement_metrics)
            }
            
            # Calculate EBITDA if not directly available
            if 'ebitda' not in results[period_key]['income_statement']:
                operating_profit = results[period_key]['income_statement'].get('operating_profit')
                depreciation = self.extract_value(text, r'знос, истощение и амортизация')
                if operating_profit is not None and depreciation is not None:
                    results[period_key]['income_statement']['ebitda'] = operating_profit + depreciation
        
        return results

    def format_results(self, results: Dict[str, Dict]) -> pd.DataFrame:
        """Format results into a DataFrame with proper multi-index."""
        data = []
        for period_key, period_data in results.items():
            period = period_data['period']
            
            # Combine all metrics
            metrics = {}
            metrics.update({f"bs_{k}": v for k, v in period_data['balance_sheet'].items()})
            metrics.update({f"is_{k}": v for k, v in period_data['income_statement'].items()})
            
            data.append({
                'period_type': period.period_type,
                'year': period.year,
                'end_date': period.end_date,
                **metrics
            })
        
        return pd.DataFrame(data)