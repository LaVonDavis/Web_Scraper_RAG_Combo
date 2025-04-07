import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List
from config import Config
from llama_cpp import Llama
import logging

logger = logging.getLogger(__name__)

class RAGSystem:
    def __init__(self):
        self.embedder = SentenceTransformer(Config.MODELS['embedder'])
        self.index = None
        self.documents = []
        
        # Initialize LLM
        self.llm = Llama(
            model_path=Config.MODELS['llm'],
            n_ctx=Config.RAG['context_length'],
            n_threads=Config.RAG['threads']
        )

    def build_index(self, documents: List[str]):
        """Create FAISS index with validation."""
        if not documents:
            raise ValueError("No documents provided for indexing")
            
        self.documents = documents
        embeddings = self.embedder.encode(documents, show_progress_bar=True)
        self.index = faiss.IndexFlatL2(embeddings.shape[1])
        self.index.add(embeddings.astype(np.float32))

    def query(self, question: str, top_k: int = 3) -> str:
        """End-to-end RAG pipeline with safety checks."""
        if not self.index:
            raise RuntimeError("Index not initialized. Call build_index() first")
            
        # Embed question
        query_embedding = self.embedder.encode([question])
        
        # Search index
        distances, indices = self.index.search(query_embedding, top_k)
        
        # Retrieve relevant documents
        context = [self.documents[i] for i in indices[0]]
        
        # Generate prompt
        prompt = self._format_prompt(question, context)
        
        # Generate response
        response = self.llm(
            prompt,
            max_tokens=Config.RAG['max_tokens'],
            temperature=Config.RAG['temperature'],
            top_p=Config.RAG['top_p']
        )
        
        return response['choices'][0]['text']

    def _format_prompt(self, question: str, context: List[str]) -> str:
        """Create structured prompt with token limits."""
        prompt = f"""Answer the question based on the context below. Be detailed but concise.

Question: {question}

Context:
"""
        token_buffer = Config.RAG['max_tokens'] - len(tokenizer.encode(prompt))
        
        for doc in context:
            doc_tokens = tokenizer.encode(doc)
            if len(doc_tokens) > token_buffer:
                doc = tokenizer.decode(doc_tokens[:token_buffer])
                token_buffer = 0
            else:
                token_buffer -= len(doc_tokens)
            prompt += f"- {doc}\n"
            
            if token_buffer <= 0:
                break
                
        return prompt
