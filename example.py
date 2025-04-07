from src.scraper import NewsScraper
from src.rag_system import RAGSystem
from src.nlp_processing import NLPProcessor
from src.config import Config

def main():
    # Scrape and store content
    scraper = NewsScraper()
    for url in Config.URLS:
        scraper.process_site(url)

    # Process documents
    processor = NLPProcessor()
    documents = []
    for url in Config.URLS:
        chunks = scraper.redis.lrange(url, 0, -1)
        documents.extend(processor.preprocess([chunk.decode() for chunk in chunks]))

    # Initialize RAG
    rag = RAGSystem()
    rag.build_index(documents)
    
    # Example query
    response = rag.query("What are the latest developments in blockchain scalability?")
    print("Generated Response:", response)

if __name__ == "__main__":
    main()
