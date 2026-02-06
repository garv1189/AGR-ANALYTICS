"""
SEC EDGAR Data Extractor
Extracts 10-K filings from SEC EDGAR database
"""
import requests
import time
import re
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from bs4 import BeautifulSoup
import logging
from dataclasses import dataclass
import json

from config import settings

logger = logging.getLogger(__name__)


@dataclass
class Company:
    """Company information"""
    cik: str
    name: str
    ticker: Optional[str] = None
    sic: Optional[str] = None
    

@dataclass
class Filing:
    """SEC Filing information"""
    accession_number: str
    filing_date: str
    report_date: str
    company: Company
    form_type: str
    file_url: str
    document_url: str


class SECEdgarExtractor:
    """Extract SEC 10-K filings from EDGAR database"""
    
    BASE_URL = "https://www.sec.gov"
    SEARCH_URL = f"{BASE_URL}/cgi-bin/browse-edgar"
    COMPANY_SEARCH_URL = f"{BASE_URL}/cgi-bin/cik_lookup"
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": settings.SEC_USER_AGENT,
            "Accept-Encoding": "gzip, deflate",
            "Host": "www.sec.gov"
        })
        self.rate_limit_delay = 1.0 / settings.SEC_API_RATE_LIMIT
        self.last_request_time = 0
    
    def _rate_limit(self):
        """Enforce rate limiting"""
        current_time = time.time()
        time_since_last_request = current_time - self.last_request_time
        if time_since_last_request < self.rate_limit_delay:
            time.sleep(self.rate_limit_delay - time_since_last_request)
        self.last_request_time = time.time()
    
    def _make_request(self, url: str, params: Optional[Dict] = None) -> requests.Response:
        """Make rate-limited request to SEC"""
        self._rate_limit()
        try:
            response = self.session.get(url, params=params, timeout=30)
            response.raise_for_status()
            return response
        except requests.RequestException as e:
            logger.error(f"Request failed: {e}")
            raise
    
    def search_company_by_name(self, company_name: str) -> List[Company]:
        """Search for company by name to get CIK"""
        logger.info(f"Searching for company: {company_name}")
        
        # Try company tickers API first
        url = f"{self.BASE_URL}/files/company_tickers.json"
        response = self._make_request(url)
        
        companies = []
        if response.status_code == 200:
            data = response.json()
            for item in data.values():
                if company_name.lower() in item.get('title', '').lower():
                    companies.append(Company(
                        cik=str(item['cik_str']).zfill(10),
                        name=item['title'],
                        ticker=item.get('ticker', ''),
                    ))
        
        if not companies:
            # Fallback to EDGAR search
            params = {
                'company': company_name,
                'output': 'xml'
            }
            response = self._make_request(self.COMPANY_SEARCH_URL, params=params)
            soup = BeautifulSoup(response.content, 'xml')
            
            for company_tag in soup.find_all('company'):
                cik = company_tag.find('CIK').text.zfill(10)
                name = company_tag.find('conformed-name').text
                companies.append(Company(cik=cik, name=name))
        
        logger.info(f"Found {len(companies)} companies")
        return companies
    
    def get_10k_filings(
        self, 
        cik: str, 
        count: int = 5,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> List[Filing]:
        """Get 10-K filings for a company"""
        logger.info(f"Fetching 10-K filings for CIK: {cik}")
        
        params = {
            'action': 'getcompany',
            'CIK': cik,
            'type': '10-K',
            'dateb': '',
            'owner': 'exclude',
            'count': count,
            'output': 'xml'
        }
        
        response = self._make_request(self.SEARCH_URL, params=params)
        soup = BeautifulSoup(response.content, 'xml')
        
        # Get company info
        company_info = soup.find('company-info')
        if not company_info:
            logger.error("No company info found")
            return []
        
        company = Company(
            cik=company_info.find('cik').text.zfill(10),
            name=company_info.find('conformed-name').text,
            sic=company_info.find('assigned-sic').text if company_info.find('assigned-sic') else None
        )
        
        # Parse filings
        filings = []
        for filing_tag in soup.find_all('filing'):
            filing_date = filing_tag.find('filing-date').text
            
            # Filter by date if specified
            if start_date or end_date:
                file_date = datetime.strptime(filing_date, '%Y-%m-%d')
                if start_date and file_date < start_date:
                    continue
                if end_date and file_date > end_date:
                    continue
            
            accession_number = filing_tag.find('accession-number').text
            file_number = accession_number.replace('-', '')
            
            # Construct document URL
            document_url = f"{self.BASE_URL}/cgi-bin/viewer?action=view&cik={cik}&accession_number={accession_number}&xbrl_type=v"
            
            filing = Filing(
                accession_number=accession_number,
                filing_date=filing_date,
                report_date=filing_tag.find('period').text if filing_tag.find('period') else filing_date,
                company=company,
                form_type=filing_tag.find('type').text,
                file_url=f"{self.BASE_URL}/cgi-bin/viewer?action=view&cik={cik}&accession_number={accession_number}",
                document_url=document_url
            )
            filings.append(filing)
        
        logger.info(f"Found {len(filings)} 10-K filings")
        return filings
    
    def download_10k_document(self, filing: Filing) -> Tuple[str, Dict]:
        """Download full 10-K document with metadata"""
        logger.info(f"Downloading 10-K document: {filing.accession_number}")
        
        # Get the filing page
        accession_number_formatted = filing.accession_number.replace('-', '')
        archives_url = f"{self.BASE_URL}/cgi-bin/viewer?action=view&cik={filing.company.cik}&accession_number={filing.accession_number}&xbrl_type=v"
        
        response = self._make_request(archives_url)
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Find the actual 10-K document link
        document_links = []
        for link in soup.find_all('a', href=True):
            href = link['href']
            if '.htm' in href and '10-k' in link.text.lower():
                document_links.append(href)
        
        # Get first document or use alternative method
        if document_links:
            doc_url = f"{self.BASE_URL}{document_links[0]}" if not document_links[0].startswith('http') else document_links[0]
        else:
            # Alternative: construct direct URL
            doc_url = f"{self.BASE_URL}/Archives/edgar/data/{filing.company.cik}/{accession_number_formatted}/{filing.accession_number}.txt"
        
        # Download document
        response = self._make_request(doc_url)
        content = response.text
        
        # Extract metadata
        metadata = {
            'company_name': filing.company.name,
            'cik': filing.company.cik,
            'ticker': filing.company.ticker,
            'sic': filing.company.sic,
            'accession_number': filing.accession_number,
            'filing_date': filing.filing_date,
            'report_date': filing.report_date,
            'form_type': filing.form_type,
            'document_url': doc_url,
            'download_date': datetime.now().isoformat()
        }
        
        logger.info(f"Successfully downloaded 10-K document ({len(content)} bytes)")
        return content, metadata
    
    def extract_sections(self, content: str) -> Dict[str, str]:
        """Extract specific sections from 10-K document"""
        sections = {}
        
        # Common section patterns
        section_patterns = {
            'item_1': r'(?:ITEM\s+1[\.\s]+|ITEM\s+1[\s\-]+)(?:BUSINESS|Description of Business)',
            'item_1a': r'(?:ITEM\s+1A[\.\s]+|ITEM\s+1A[\s\-]+)(?:RISK FACTORS|Risk Factors)',
            'item_7': r'(?:ITEM\s+7[\.\s]+|ITEM\s+7[\s\-]+)(?:MANAGEMENT|Management)(?:\'s|s)?(?:\s+DISCUSSION|Discussion)',
            'item_7a': r'(?:ITEM\s+7A[\.\s]+|ITEM\s+7A[\s\-]+)(?:QUANTITATIVE|Quantitative)',
            'item_8': r'(?:ITEM\s+8[\.\s]+|ITEM\s+8[\s\-]+)(?:FINANCIAL STATEMENTS|Financial Statements)',
        }
        
        # Try to extract each section
        for section_name, pattern in section_patterns.items():
            match = re.search(pattern, content, re.IGNORECASE | re.MULTILINE)
            if match:
                start_pos = match.start()
                # Find next section or end of document
                remaining_content = content[start_pos:]
                next_section_match = re.search(r'\n\s*ITEM\s+\d+[A-Z]?[\.\s\-]', remaining_content[100:], re.IGNORECASE)
                
                if next_section_match:
                    end_pos = start_pos + 100 + next_section_match.start()
                    sections[section_name] = content[start_pos:end_pos]
                else:
                    sections[section_name] = remaining_content[:50000]  # Limit to 50k chars
        
        return sections


# Singleton instance
edgar_extractor = SECEdgarExtractor()
