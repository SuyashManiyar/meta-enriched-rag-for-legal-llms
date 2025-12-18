#!/usr/bin/env python3
"""
Australian Legal Document Web Scraper
Scrapes legal documents from various Australian legal databases and courts.

Supported Sources:
- Federal Court of Australia (judgments.fedcourt.gov.au)
- High Court of Australia (eresources.hcourt.gov.au)
- NSW Caselaw (caselaw.nsw.gov.au)
- Australian Legislation (legislation.gov.au)
- Queensland Legislation (legislation.qld.gov.au)
- Western Australia Legislation (legislation.wa.gov.au)
- Tasmania Legislation (legislation.tas.gov.au)
"""

import requests
import time
import os
import json
import logging
from urllib.parse import urljoin, urlparse
from pathlib import Path
import re
from typing import List, Dict, Optional
import PyPDF2
from bs4 import BeautifulSoup
import random

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('legal_scraper.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class AustralianLegalScraper:
    def __init__(self, output_dir: str = "australian_legal_documents_final"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Request headers to avoid blocking
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
        }
        
        # Rate limiting
        self.delay_range = (2, 5)  # Random delay between requests
        
        # Document counter
        self.doc_counter = 1
        
        # Metadata storage
        self.metadata = {}
    
    def random_delay(self):
        """Add random delay between requests to be respectful"""
        delay = random.uniform(*self.delay_range)
        time.sleep(delay)
    
    def safe_request(self, url: str, timeout: int = 30) -> Optional[requests.Response]:
        """Make a safe HTTP request with error handling"""
        try:
            self.random_delay()
            response = requests.get(url, headers=self.headers, timeout=timeout)
            response.raise_for_status()
            return response
        except requests.RequestException as e:
            logger.error(f"Request failed for {url}: {e}")
            return None
    
    def extract_text_from_pdf(self, pdf_content: bytes) -> str:
        """Extract text from PDF content"""
        try:
            from io import BytesIO
            pdf_reader = PyPDF2.PdfReader(BytesIO(pdf_content))
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
            return text.strip()
        except Exception as e:
            logger.error(f"PDF text extraction failed: {e}")
            return ""
    
    def save_document(self, content: bytes, url: str, doc_type: str = "pdf") -> str:
        """Save document and return filename"""
        # Generate filename
        parsed_url = urlparse(url)
        domain = parsed_url.netloc.replace('.', '_')
        path_parts = parsed_url.path.strip('/').replace('/', '_')
        
        filename = f"{self.doc_counter:03d}_{domain}_{path_parts}"
        if not filename.endswith(f".{doc_type}"):
            filename += f".{doc_type}"
        
        filepath = self.output_dir / filename
        
        # Save file
        with open(filepath, 'wb') as f:
            f.write(content)
        
        # Store metadata
        self.metadata[filename] = {
            'url': url,
            'doc_type': doc_type,
            'domain': parsed_url.netloc,
            'file_size': len(content),
            'doc_number': self.doc_counter
        }
        
        self.doc_counter += 1
        logger.info(f"Saved: {filename}")
        return filename
    
    def scrape_federal_court_judgments(self, max_docs: int = 50) -> List[str]:
        """Scrape Federal Court of Australia judgments"""
        logger.info("Scraping Federal Court judgments...")
        saved_files = []
        
        # Federal Court search URLs (example patterns)
        base_urls = [
            "https://judgments.fedcourt.gov.au/judgments/Judgments/fca/single/2023/",
            "https://judgments.fedcourt.gov.au/judgments/Judgments/fca/single/2022/",
            "https://judgments.fedcourt.gov.au/judgments/Judgments/fca/single/2021/",
            "https://judgments.fedcourt.gov.au/judgments/Judgments/fca/single/2020/",
        ]
        
        for base_url in base_urls:
            if len(saved_files) >= max_docs:
                break
                
            response = self.safe_request(base_url)
            if not response:
                continue
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Find PDF links (adjust selector based on actual site structure)
            pdf_links = soup.find_all('a', href=re.compile(r'\.pdf$', re.I))
            
            for link in pdf_links[:max_docs - len(saved_files)]:
                pdf_url = urljoin(base_url, link.get('href'))
                pdf_response = self.safe_request(pdf_url)
                
                if pdf_response and pdf_response.content:
                    filename = self.save_document(pdf_response.content, pdf_url, 'pdf')
                    saved_files.append(filename)
                    
                    if len(saved_files) >= max_docs:
                        break
        
        return saved_files
    
    def scrape_high_court_decisions(self, max_docs: int = 20) -> List[str]:
        """Scrape High Court of Australia decisions"""
        logger.info("Scraping High Court decisions...")
        saved_files = []
        
        # High Court URLs (example patterns)
        base_urls = [
            "https://eresources.hcourt.gov.au/downloadPdf/",
        ]
        
        # Example HCA case numbers (you would need to build this list)
        case_numbers = [
            "2023/HCA/1", "2023/HCA/2", "2022/HCA/45", "2022/HCA/44",
            "2021/HCA/50", "2021/HCA/49", "2020/HCA/55", "2020/HCA/54"
        ]
        
        for case_num in case_numbers[:max_docs]:
            url = f"https://eresources.hcourt.gov.au/downloadPdf/{case_num}"
            response = self.safe_request(url)
            
            if response and response.content:
                filename = self.save_document(response.content, url, 'pdf')
                saved_files.append(filename)
        
        return saved_files
    
    def scrape_nsw_caselaw(self, max_docs: int = 30) -> List[str]:
        """Scrape NSW Caselaw decisions"""
        logger.info("Scraping NSW Caselaw...")
        saved_files = []
        
        # NSW Caselaw search API or direct URLs
        base_url = "https://caselaw.nsw.gov.au/decision/"
        
        # Example decision IDs (you would need to collect these)
        decision_ids = [
            "549fa9f13004262463b5de89", "549fa24c3004262463b38f36",
            "549fdd743004262463c0e3b6", "5a8e1e8be4b087b8baa86611",
            "549f9fd93004262463b2be58", "549fa48c3004262463b44364"
        ]
        
        for decision_id in decision_ids[:max_docs]:
            url = f"{base_url}{decision_id}"
            response = self.safe_request(url)
            
            if response:
                # NSW Caselaw might serve HTML, convert to PDF or save as HTML
                filename = self.save_document(response.content, url, 'html')
                saved_files.append(filename)
        
        return saved_files
    
    def scrape_legislation_gov_au(self, max_docs: int = 25) -> List[str]:
        """Scrape Australian Government legislation"""
        logger.info("Scraping legislation.gov.au...")
        saved_files = []
        
        # Legislation URLs (example patterns)
        legislation_ids = [
            "F2023L00486", "F2021N00097", "F2008L03415", "F2016N00036",
            "C2023A00002", "F2006B11509", "F2006B04469", "F2005C00417",
            "C2023A00059", "F2006B03701", "F2017L01604", "C2016A00007"
        ]
        
        for leg_id in legislation_ids[:max_docs]:
            url = f"https://legislation.gov.au/Details/{leg_id}"
            response = self.safe_request(url)
            
            if response:
                # Look for PDF download link
                soup = BeautifulSoup(response.content, 'html.parser')
                pdf_link = soup.find('a', href=re.compile(r'\.pdf$', re.I))
                
                if pdf_link:
                    pdf_url = urljoin(url, pdf_link.get('href'))
                    pdf_response = self.safe_request(pdf_url)
                    
                    if pdf_response and pdf_response.content:
                        filename = self.save_document(pdf_response.content, pdf_url, 'pdf')
                        saved_files.append(filename)
        
        return saved_files
    
    def scrape_state_legislation(self, max_docs: int = 25) -> List[str]:
        """Scrape state and territory legislation"""
        logger.info("Scraping state legislation...")
        saved_files = []
        
        # Queensland legislation
        qld_acts = [
            "act-1981-074", "act-1855-awlai", "act-1962-069"
        ]
        
        for act in qld_acts:
            if len(saved_files) >= max_docs:
                break
            url = f"https://legislation.qld.gov.au/view/whole/html/inforce/current/{act}"
            response = self.safe_request(url)
            
            if response:
                filename = self.save_document(response.content, url, 'html')
                saved_files.append(filename)
        
        # Western Australia legislation
        wa_docs = ["mrdoc_3678.docx", "mrdoc_46257.docx", "mrdoc_2942.docx"]
        
        for doc in wa_docs:
            if len(saved_files) >= max_docs:
                break
            url = f"https://legislation.wa.gov.au/legislation/statutes.nsf/RedirectURL/{doc}"
            response = self.safe_request(url)
            
            if response:
                filename = self.save_document(response.content, url, 'docx')
                saved_files.append(filename)
        
        # Tasmania legislation
        tas_regs = ["sr-2022-110", "sr-1999-091"]
        
        for reg in tas_regs:
            if len(saved_files) >= max_docs:
                break
            url = f"https://legislation.tas.gov.au/view/whole/html/inforce/current/{reg}"
            response = self.safe_request(url)
            
            if response:
                filename = self.save_document(response.content, url, 'html')
                saved_files.append(filename)
        
        return saved_files
    
    def save_metadata(self):
        """Save metadata to JSON file"""
        metadata_file = self.output_dir / "scraping_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(self.metadata, f, indent=2)
        logger.info(f"Metadata saved to {metadata_file}")
    
    def run_full_scrape(self, target_docs: int = 200):
        """Run complete scraping process"""
        logger.info(f"Starting Australian legal document scraping (target: {target_docs} documents)")
        
        all_files = []
        
        # Distribute documents across sources
        federal_court_docs = min(80, target_docs // 4)
        high_court_docs = min(30, target_docs // 6)
        nsw_caselaw_docs = min(50, target_docs // 4)
        legislation_docs = min(40, target_docs // 5)
        
        try:
            # Scrape from different sources
            all_files.extend(self.scrape_federal_court_judgments(federal_court_docs))
            all_files.extend(self.scrape_high_court_decisions(high_court_docs))
            all_files.extend(self.scrape_nsw_caselaw(nsw_caselaw_docs))
            all_files.extend(self.scrape_legislation_gov_au(legislation_docs))
            all_files.extend(self.scrape_state_legislation(target_docs - len(all_files)))
            
        except KeyboardInterrupt:
            logger.info("Scraping interrupted by user")
        except Exception as e:
            logger.error(f"Scraping error: {e}")
        
        # Save metadata
        self.save_metadata()
        
        logger.info(f"Scraping completed. Downloaded {len(all_files)} documents.")
        logger.info(f"Documents saved to: {self.output_dir}")
        
        return all_files

def main():
    """Main function to run the scraper"""
    scraper = AustralianLegalScraper()
    
    # Run scraping
    files = scraper.run_full_scrape(target_docs=200)
    
    print(f"\n✅ Scraping completed!")
    print(f"📁 Documents saved to: {scraper.output_dir}")
    print(f"📊 Total documents: {len(files)}")
    print(f"📋 Metadata saved to: {scraper.output_dir}/scraping_metadata.json")
    
    # Print summary by source
    sources = {}
    for filename, metadata in scraper.metadata.items():
        domain = metadata['domain']
        sources[domain] = sources.get(domain, 0) + 1
    
    print(f"\n📈 Documents by source:")
    for domain, count in sources.items():
        print(f"  {domain}: {count} documents")

if __name__ == "__main__":
    main()