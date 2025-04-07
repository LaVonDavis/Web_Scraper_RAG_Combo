import logging
import redis
import requests
from bs4 import BeautifulSoup
from typing import List, Tuple
from urllib.parse import urljoin, urlparse
from config import Config

logger = logging.getLogger(__name__)

class NewsScraper:
    def __init__(self):
        self.redis = redis.Redis(**Config.REDIS)
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'}

    def fetch_url(self, url: str) -> str:
        """Fetch HTML content with error handling and rate limiting."""
        try:
            response = requests.get(url, headers=self.headers, timeout=10)
            response.raise_for_status()
            return response.text
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to fetch {url}: {str(e)}")
            return ""

    def extract_content(self, html: str, base_url: str) -> Tuple[List[str], List[Tuple[str, str]]]:
        """Extract and clean text content with spam filtering."""
        soup = BeautifulSoup(html, 'html.parser')
        
        # Remove unwanted elements
        for tag in ['script', 'style', 'footer', 'nav', 'header']:
            for element in soup.find_all(tag):
                element.decompose()

        # Extract main content
        text = soup.get_text(separator=' ', strip=True)
        links = []
        for a in soup.find_all('a', href=True):
            href = a['href']
            if not href.startswith('http'):
                href = urljoin(base_url, href)
            if self._is_valid_link(href):
                links.append((href, a.get_text(strip=True)))

        return self._clean_text(text), links

    def _is_valid_link(self, url: str) -> bool:
        """Filter out non-news links."""
        parsed = urlparse(url)
        return all(invalid not in parsed.path for invalid in ['wp-login', 'cdn-cgi', 'tag'])

    def _clean_text(self, text: str) -> List[str]:
        """Basic text cleaning and chunking."""
        # Remove excessive whitespace
        cleaned = ' '.join(text.split())
        # Split into paragraphs
        return [p for p in cleaned.split('  ') if len(p) > 100]

    def process_site(self, url: str):
        """Full processing pipeline for a news site."""
        if self.redis.exists(url):
            logger.info(f"Skipping already processed URL: {url}")
            return

        html = self.fetch_url(url)
        if not html:
            return

        text_chunks, links = self.extract_content(html, url)
        if text_chunks:
            # Store in Redis pipeline
            pipe = self.redis.pipeline()
            for chunk in text_chunks:
                pipe.rpush(url, chunk)
            pipe.execute()
            logger.info(f"Stored {len(text_chunks)} chunks from {url}")
