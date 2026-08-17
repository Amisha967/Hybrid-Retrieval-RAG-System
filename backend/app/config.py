import os
from typing import Optional

try:
    from pydantic_settings import BaseSettings
    class Settings(BaseSettings):
        PROJECT_NAME: str = "Hybrid Retrieval RAG System"
        API_V1_STR: str = "/api/v1"
        
        # ChromaDB & Vector Store
        CHROMA_PERSIST_DIRECTORY: str = os.getenv("CHROMA_PERSIST_DIRECTORY", "./data/chroma_db")
        CHROMA_COLLECTION_NAME: str = os.getenv("CHROMA_COLLECTION_NAME", "hybrid_rag_documents")
        
        # Embedding Model (HuggingFace / PyTorch)
        EMBEDDING_MODEL_NAME: str = os.getenv("EMBEDDING_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")
        EMBEDDING_DEVICE: str = os.getenv("EMBEDDING_DEVICE", "cpu")
        
        # Cross-Encoder Reranker Model
        RERANKER_MODEL_NAME: str = os.getenv("RERANKER_MODEL_NAME", "cross-encoder/ms-marco-MiniLM-L-6-v2")
        RERANKER_DEVICE: str = os.getenv("RERANKER_DEVICE", "cpu")
        
        # Sparse BM25 Persistence
        BM25_PERSIST_PATH: str = os.getenv("BM25_PERSIST_PATH", "./data/bm25_index.pkl")
        
        # Caching
        CACHE_MAX_SIZE: int = int(os.getenv("CACHE_MAX_SIZE", "1000"))
        CACHE_TTL_SECONDS: int = int(os.getenv("CACHE_TTL_SECONDS", "3600"))
        
        # LLM Settings (Pluggable: local_extractive, openai, gemini, huggingface)
        LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "local_extractive")
        GEMINI_API_KEY: Optional[str] = os.getenv("GEMINI_API_KEY", None)
        OPENAI_API_KEY: Optional[str] = os.getenv("OPENAI_API_KEY", None)
        HF_MODEL_NAME: str = os.getenv("HF_MODEL_NAME", "google/flan-t5-base")
        
        # Default Hybrid Retrieval Hyperparameters
        DEFAULT_DENSE_TOP_K: int = 10
        DEFAULT_SPARSE_TOP_K: int = 10
        DEFAULT_RERANK_TOP_K: int = 5
        DEFAULT_HYBRID_ALPHA: float = 0.5
        
        class Config:
            case_sensitive = True
            env_file = ".env"

except ImportError:
    class Settings:
        PROJECT_NAME: str = "Hybrid Retrieval RAG System"
        API_V1_STR: str = "/api/v1"
        CHROMA_PERSIST_DIRECTORY: str = os.getenv("CHROMA_PERSIST_DIRECTORY", "./data/chroma_db")
        CHROMA_COLLECTION_NAME: str = os.getenv("CHROMA_COLLECTION_NAME", "hybrid_rag_documents")
        EMBEDDING_MODEL_NAME: str = os.getenv("EMBEDDING_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")
        EMBEDDING_DEVICE: str = os.getenv("EMBEDDING_DEVICE", "cpu")
        RERANKER_MODEL_NAME: str = os.getenv("RERANKER_MODEL_NAME", "cross-encoder/ms-marco-MiniLM-L-6-v2")
        RERANKER_DEVICE: str = os.getenv("RERANKER_DEVICE", "cpu")
        BM25_PERSIST_PATH: str = os.getenv("BM25_PERSIST_PATH", "./data/bm25_index.pkl")
        CACHE_MAX_SIZE: int = int(os.getenv("CACHE_MAX_SIZE", "1000"))
        CACHE_TTL_SECONDS: int = int(os.getenv("CACHE_TTL_SECONDS", "3600"))
        LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "local_extractive")
        GEMINI_API_KEY: Optional[str] = os.getenv("GEMINI_API_KEY", None)
        OPENAI_API_KEY: Optional[str] = os.getenv("OPENAI_API_KEY", None)
        HF_MODEL_NAME: str = os.getenv("HF_MODEL_NAME", "google/flan-t5-base")
        DEFAULT_DENSE_TOP_K: int = 10
        DEFAULT_SPARSE_TOP_K: int = 10
        DEFAULT_RERANK_TOP_K: int = 5
        DEFAULT_HYBRID_ALPHA: float = 0.5

settings = Settings()
