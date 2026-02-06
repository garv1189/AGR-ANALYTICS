"""
Data Preprocessor for 10-K Filings
Cleans, structures, and converts data into JSON format
"""
import re
import json
from typing import Dict, List, Optional, Tuple
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class DataPreprocessor:
    """Preprocess and structure 10-K filing data"""
    
    def __init__(self):
        self.financial_keywords = [
            'revenue', 'income', 'assets', 'liabilities', 'equity', 
            'cash', 'debt', 'earnings', 'expenses', 'profit', 'loss'
        ]
    
    def clean_html(self, content: str) -> str:
        """Remove HTML tags and clean text"""
        soup = BeautifulSoup(content, 'html.parser')
        
        # Remove script and style elements
        for script in soup(["script", "style", "meta", "link"]):
            script.decompose()
        
        # Get text
        text = soup.get_text()
        
        # Clean whitespace
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = ' '.join(chunk for chunk in chunks if chunk)
        
        return text
    
    def extract_tables(self, content: str) -> List[pd.DataFrame]:
        """Extract financial tables from HTML content"""
        soup = BeautifulSoup(content, 'html.parser')
        tables = []
        
        for table in soup.find_all('table'):
            try:
                # Convert to pandas dataframe
                df = pd.read_html(str(table))[0]
                
                # Filter tables that look like financial statements
                if self._is_financial_table(df):
                    df = self._clean_financial_table(df)
                    tables.append(df)
            except Exception as e:
                logger.warning(f"Failed to parse table: {e}")
                continue
        
        logger.info(f"Extracted {len(tables)} financial tables")
        return tables
    
    def _is_financial_table(self, df: pd.DataFrame) -> bool:
        """Check if table contains financial data"""
        if df.empty or len(df) < 2:
            return False
        
        # Check for financial keywords in first column
        first_col = df.iloc[:, 0].astype(str).str.lower()
        keyword_matches = sum(
            any(keyword in cell for keyword in self.financial_keywords)
            for cell in first_col
        )
        
        # Check for numeric columns (at least 2)
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        has_numbers = len(numeric_cols) >= 2
        
        return keyword_matches >= 2 and has_numbers
    
    def _clean_financial_table(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and standardize financial table"""
        # Remove completely empty rows and columns
        df = df.dropna(how='all').dropna(axis=1, how='all')
        
        # Set first column as index if it looks like labels
        if df.iloc[:, 0].dtype == 'object':
            df = df.set_index(df.columns[0])
        
        # Clean numeric columns
        for col in df.columns:
            if df[col].dtype == 'object':
                # Try to convert to numeric
                df[col] = df[col].astype(str).str.replace('[^0-9.-]', '', regex=True)
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        return df
    
    def extract_numerical_data(self, tables: List[pd.DataFrame], metadata: Dict) -> Dict:
        """Extract and structure numerical financial data"""
        numerical_data = {
            'metadata': metadata,
            'extraction_date': datetime.now().isoformat(),
            'financial_statements': [],
            'key_metrics': {},
            'ratios': {}
        }
        
        # Process each table
        for idx, table in enumerate(tables):
            table_dict = {
                'table_id': f"table_{idx}",
                'rows': len(table),
                'columns': len(table.columns),
                'data': table.to_dict('records'),
                'column_names': table.columns.tolist(),
                'index_names': table.index.tolist() if hasattr(table, 'index') else []
            }
            numerical_data['financial_statements'].append(table_dict)
        
        # Extract key metrics from tables
        numerical_data['key_metrics'] = self._extract_key_metrics(tables)
        
        # Calculate financial ratios
        numerical_data['ratios'] = self._calculate_ratios(numerical_data['key_metrics'])
        
        return numerical_data
    
    def _extract_key_metrics(self, tables: List[pd.DataFrame]) -> Dict:
        """Extract key financial metrics from tables"""
        metrics = {}
        
        for table in tables:
            # Convert to string for pattern matching
            table_str = table.to_string().lower()
            
            # Common metric patterns
            metric_patterns = {
                'total_revenue': r'(?:total\s+)?revenue',
                'net_income': r'net\s+(?:income|earnings)',
                'total_assets': r'total\s+assets',
                'total_liabilities': r'total\s+liabilities',
                'shareholders_equity': r'(?:shareholders?|stockholders?)\s+equity',
                'cash_and_equivalents': r'cash\s+and\s+(?:cash\s+)?equivalents',
                'total_debt': r'(?:total\s+)?(?:long[- ]term\s+)?debt',
                'operating_income': r'(?:income|loss)\s+from\s+operations',
                'gross_profit': r'gross\s+profit'
            }
            
            for metric_name, pattern in metric_patterns.items():
                match = re.search(pattern, table_str, re.IGNORECASE)
                if match and metric_name not in metrics:
                    # Try to extract the numeric value
                    try:
                        # Get the row
                        matching_rows = table[table.astype(str).apply(
                            lambda row: any(re.search(pattern, str(val), re.IGNORECASE) 
                                          for val in row), axis=1
                        )]
                        
                        if not matching_rows.empty:
                            # Get the most recent value (usually last numeric column)
                            numeric_cols = matching_rows.select_dtypes(include=[np.number]).columns
                            if len(numeric_cols) > 0:
                                value = matching_rows[numeric_cols[-1]].values[0]
                                if not pd.isna(value):
                                    metrics[metric_name] = float(value)
                    except Exception as e:
                        logger.debug(f"Could not extract {metric_name}: {e}")
                        continue
        
        return metrics
    
    def _calculate_ratios(self, metrics: Dict) -> Dict:
        """Calculate financial ratios from key metrics"""
        ratios = {}
        
        try:
            # Profitability ratios
            if 'net_income' in metrics and 'total_revenue' in metrics:
                ratios['profit_margin'] = metrics['net_income'] / metrics['total_revenue']
            
            if 'net_income' in metrics and 'shareholders_equity' in metrics:
                ratios['return_on_equity'] = metrics['net_income'] / metrics['shareholders_equity']
            
            if 'net_income' in metrics and 'total_assets' in metrics:
                ratios['return_on_assets'] = metrics['net_income'] / metrics['total_assets']
            
            # Leverage ratios
            if 'total_debt' in metrics and 'shareholders_equity' in metrics:
                ratios['debt_to_equity'] = metrics['total_debt'] / metrics['shareholders_equity']
            
            if 'total_liabilities' in metrics and 'total_assets' in metrics:
                ratios['debt_ratio'] = metrics['total_liabilities'] / metrics['total_assets']
            
            # Liquidity (placeholder - needs current assets/liabilities)
            if 'cash_and_equivalents' in metrics and 'total_assets' in metrics:
                ratios['cash_ratio'] = metrics['cash_and_equivalents'] / metrics['total_assets']
                
        except ZeroDivisionError:
            logger.warning("Division by zero in ratio calculation")
        except Exception as e:
            logger.error(f"Error calculating ratios: {e}")
        
        return ratios
    
    def extract_textual_data(self, content: str, sections: Dict[str, str], metadata: Dict) -> Dict:
        """Extract and structure textual data"""
        # Clean HTML from each section
        cleaned_sections = {}
        for section_name, section_content in sections.items():
            cleaned_sections[section_name] = self.clean_html(section_content)
        
        textual_data = {
            'metadata': metadata,
            'extraction_date': datetime.now().isoformat(),
            'full_text': self.clean_html(content),
            'sections': cleaned_sections,
            'text_statistics': {}
        }
        
        # Calculate text statistics
        for section_name, section_text in cleaned_sections.items():
            textual_data['text_statistics'][section_name] = {
                'character_count': len(section_text),
                'word_count': len(section_text.split()),
                'sentence_count': len(re.split(r'[.!?]+', section_text)),
                'paragraph_count': len(re.split(r'\n\s*\n', section_text))
            }
        
        return textual_data
    
    def process_10k_filing(self, content: str, metadata: Dict, sections: Dict[str, str]) -> Dict:
        """Complete preprocessing pipeline for 10-K filing"""
        logger.info(f"Processing 10-K filing: {metadata.get('accession_number')}")
        
        # Extract tables
        tables = self.extract_tables(content)
        
        # Process numerical data
        numerical_data = self.extract_numerical_data(tables, metadata)
        
        # Process textual data
        textual_data = self.extract_textual_data(content, sections, metadata)
        
        # Combine into structured format
        processed_data = {
            'filing_id': metadata.get('accession_number'),
            'company_info': {
                'name': metadata.get('company_name'),
                'cik': metadata.get('cik'),
                'ticker': metadata.get('ticker'),
                'sic': metadata.get('sic')
            },
            'filing_info': {
                'filing_date': metadata.get('filing_date'),
                'report_date': metadata.get('report_date'),
                'form_type': metadata.get('form_type'),
                'accession_number': metadata.get('accession_number')
            },
            'processing_info': {
                'processed_date': datetime.now().isoformat(),
                'tables_extracted': len(tables),
                'sections_extracted': len(sections)
            },
            'numerical_data': numerical_data,
            'textual_data': textual_data
        }
        
        logger.info("Processing complete")
        return processed_data
    
    def save_to_json(self, data: Dict, filepath: str):
        """Save processed data to JSON file"""
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False, default=str)
        logger.info(f"Saved data to {filepath}")


# Singleton instance
data_preprocessor = DataPreprocessor()
