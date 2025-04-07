import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    REDIS = {
        'host': os.getenv('REDIS_HOST', 'localhost'),
        'port': int(os.getenv('REDIS_PORT', 6379)),
        'db': int(os.getenv('REDIS_DB', 0)),
        'ssl': os.getenv('REDIS_SSL', 'false').lower() == 'true'
    }

    NLP = {
        'spacy_model': 'en_core_web_lg',
        'tokenizer': 'cl100k_base',
        'chunk_size': 512,
        'overlap': 64
    }

    RAG = {
        'context_length': 4096,
        'max_tokens': 3500,
        'temperature': 0.7,
        'top_p': 0.9,
        'threads': max(os.cpu_count() // 2, 4)
    }

    MODELS = {
        'embedder': 'all-MiniLM-L6-v2',
        'llm': os.path.expanduser('~/models/zephyr-7b-alpha.Q4_K_M.gguf')
    }

    URLS = [
        'https://www.coindesk.com',
        'https://cointelegraph.com',
        'https://www.newsbtc.com',
        'https://cryptoslate.com'
    ]
