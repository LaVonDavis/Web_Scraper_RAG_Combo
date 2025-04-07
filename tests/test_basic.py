import unittest
from src.scraper import NewsScraper
from src.nlp_processing import NLPProcessor

class TestScraper(unittest.TestCase):
    def test_fetch_url(self):
        scraper = NewsScraper()
        html = scraper.fetch_url("https://example.com")
        self.assertGreater(len(html), 0)

    def test_content_extraction(self):
        scraper = NewsScraper()
        test_html = "<html><body><p>Test content</p></body></html>"
        text, links = scraper.extract_content(test_html, "https://example.com")
        self.assertIn("Test content", text[0])

class TestNLP(unittest.TestCase):
    def test_chunking(self):
        long_text = " ".join(["sentence"] * 1000)
        chunks = NLPProcessor.chunk_text(long_text)
        self.assertTrue(2 <= len(chunks) <= 4)

    def test_preprocessing(self):
        processed = NLPProcessor.preprocess(["This is a test sentence!"])
        self.assertEqual(processed[0], "test sentence")
