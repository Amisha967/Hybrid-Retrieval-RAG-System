import os
import sys
import requests
from typing import Dict, Any, Optional, List

# Ensure parent and current directory are on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.dirname(__file__))

try:
    from frontend.config import API_V1_URL, BACKEND_URL
except ImportError:
    from config import API_V1_URL, BACKEND_URL

def check_backend_health() -> bool:
    try:
        res = requests.get(f"{BACKEND_URL}/health", timeout=2)
        return res.status_code == 200
    except Exception:
        return False

def upload_document(
    file_bytes: bytes,
    filename: str,
    strategy: str = "recursive",
    chunk_size: int = 500,
    chunk_overlap: int = 100
) -> Dict[str, Any]:
    files = {"file": (filename, file_bytes)}
    data = {
        "strategy": strategy,
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap
    }
    response = requests.post(f"{API_V1_URL}/documents/upload", files=files, data=data, timeout=60)
    response.raise_for_status()
    return response.json()

def execute_query(
    query: str,
    top_k_dense: int = 10,
    top_k_sparse: int = 10,
    top_k_rerank: int = 5,
    hybrid_alpha: float = 0.5,
    enable_reranker: bool = True,
    use_cache: bool = True,
    llm_provider: str = "local_extractive",
    custom_api_key: Optional[str] = None
) -> Dict[str, Any]:
    payload = {
        "query": query,
        "top_k_dense": top_k_dense,
        "top_k_sparse": top_k_sparse,
        "top_k_rerank": top_k_rerank,
        "hybrid_alpha": hybrid_alpha,
        "enable_reranker": enable_reranker,
        "use_cache": use_cache,
        "llm_provider": llm_provider,
        "custom_api_key": custom_api_key or None
    }
    response = requests.post(f"{API_V1_URL}/query", json=payload, timeout=60)
    response.raise_for_status()
    return response.json()

def fetch_metrics() -> Dict[str, Any]:
    response = requests.get(f"{API_V1_URL}/metrics", timeout=10)
    response.raise_for_status()
    return response.json()

def fetch_documents() -> Dict[str, Any]:
    response = requests.get(f"{API_V1_URL}/documents", timeout=10)
    response.raise_for_status()
    return response.json()

def delete_document(doc_id: str) -> Dict[str, Any]:
    response = requests.delete(f"{API_V1_URL}/documents/{doc_id}", timeout=10)
    response.raise_for_status()
    return response.json()

def reset_database() -> Dict[str, Any]:
    response = requests.post(f"{API_V1_URL}/documents/reset", timeout=10)
    response.raise_for_status()
    return response.json()
