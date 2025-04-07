import spacy
import tiktoken
from nltk.corpus import stopwords
from typing import List
from config import Config

nlp = spacy.load(Config.NLP['spacy_model'])
tokenizer = tiktoken.get_encoding(Config.NLP['tokenizer'])

class NLPProcessor:
    @staticmethod
    def chunk_text(text: str) -> List[str]:
        """Token-aware text chunking with overlap."""
        tokens = tokenizer.encode(text)
        chunk_size = Config.NLP['chunk_size']
        overlap = Config.NLP['overlap']
        
        chunks = []
        start = 0
        while start < len(tokens):
            end = min(start + chunk_size, len(tokens))
            chunks.append(tokenizer.decode(tokens[start:end]))
            start = end - overlap
            
        return chunks

    @staticmethod
    def preprocess(texts: List[str]) -> List[str]:
        """Full text normalization pipeline."""
        stop_words = set(stopwords.words('english'))
        processed = []
        
        for text in texts:
            doc = nlp(text)
            cleaned = []
            for token in doc:
                if token.is_alpha and not token.is_stop and token.lemma_.lower() not in stop_words:
                    cleaned.append(token.lemma_.lower())
            processed.append(' '.join(cleaned))
            
        return processed

    @staticmethod
    def semantic_search(target: str, candidates: List[str], top_n: int = 10) -> List[Tuple[str, float]]:
        """Find semantically similar terms using spaCy vectors."""
        target_doc = nlp(target)
        if not target_doc.has_vector:
            return []
            
        similarities = []
        for term in candidates:
            term_doc = nlp(term)
            if term_doc.has_vector:
                similarities.append((term, target_doc.similarity(term_doc)))
                
        return sorted(similarities, key=lambda x: x[1], reverse=True)[:top_n]
