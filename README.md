# ⚡ Hybrid Retrieval RAG System

[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-009688.svg?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector%20Store-orange.svg)](https://www.trychroma.com)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2+-EE4C2C.svg?logo=pytorch&logoColor=white)](https://pytorch.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.32+-FF4B4B.svg?logo=streamlit&logoColor=white)](https://streamlit.io)
[![Docker](https://img.shields.io/badge/Docker-Compose-2496ED.svg?logo=docker&logoColor=white)](https://www.docker.com)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

A high-performance, domain-specific **Retrieval-Augmented Generation (RAG)** system engineered with **FastAPI**, **ChromaDB**, **BM25**, **PyTorch Cross-Encoder Re-ranking**, and **Streamlit**. Designed to enable low-latency, citation-grounded question answering across complex unstructured document files with sub-second p50/p95 pipeline latency guarantees.

---

## 🌟 Key Features

1. **Domain-Specific Multi-Format Ingestion**:
   - Automated text and metadata extraction from `.pdf`, `.docx`, `.txt`, `.md`, `.json`, `.csv`.
   - Dynamic chunking strategies: **Recursive Character Splitting**, **Sliding Window Token Splitting**, and **Fixed-Size Chunking** with configurable chunk sizes and boundary-preserving overlaps.

2. **Hybrid Search Pipeline (Dense + Sparse + Re-ranking)**:
   - **Dense Vector Search**: ChromaDB vector store backed by HuggingFace `sentence-transformers/all-MiniLM-L6-v2` (384-dimensional cosine similarity embeddings).
   - **Sparse Lexical Search**: Custom Okapi BM25 engine with tokenization, stemming, stopword filtering, and persistent disk indexing.
   - **Hybrid Fusion**: Reciprocal Rank Fusion (RRF) and Convex Combination ($\alpha \cdot \text{Dense} + (1-\alpha) \cdot \text{BM25}$) with dynamic weighting.
   - **Cross-Encoder Re-ranking**: HuggingFace `cross-encoder/ms-marco-MiniLM-L-6-v2` applying joint cross-attention over query-chunk pairs for maximum top-$k$ context precision.

3. **Citation-Backed Generation & Latency Tracking**:
   - Fact-grounded generation output with numbered inline citations `[1]`, `[2]` linking directly to source document chunks.
   - Built-in **p50, p90, p95, p99** rolling latency percentiles and stage-by-stage timing (`embedding_ms`, `dense_search_ms`, `sparse_search_ms`, `fusion_ms`, `rerank_ms`, `generation_ms`).
   - Thread-safe **LRU Key-Value Query Cache** with SHA-256 parameter hashing and TTL expiration to consistently maintain sub-second response times.

4. **Containerized Microservices & Modern Web Interface**:
   - Responsive **Streamlit UI** featuring an interactive chat interface, deep-dive inspection drawer for retrieved chunks/scores, and real-time telemetry dashboard.
   - **Docker Compose** orchestration for reproducible local and remote cloud deployments.

---

## 🏛 Architecture Diagram

```
                                  ┌────────────────────────┐
                                  │      Streamlit UI      │
                                  │  (Port 8501 / Docker)  │
                                  └───────────┬────────────┘
                                              │ REST API
                                              ▼
                                  ┌────────────────────────┐
                                  │     FastAPI Backend    │
                                  │  (Port 8000 / Docker)  │
                                  └───────────┬────────────┘
                                              │
               ┌──────────────────────────────┼──────────────────────────────┐
               ▼                              ▼                              ▼
    ┌────────────────────┐         ┌────────────────────┐         ┌────────────────────┐
    │ Document Ingestion │         │  KV Query Cache    │         │  Latency Tracking  │
    │ & Chunking Engine  │         │ (LRU + TTL Store)  │         │ (p50/p95 Telemetry)│
    └──────────┬─────────┘         └────────────────────┘         └────────────────────┘
               │
               ├──────────────────────────────┐
               ▼                              ▼
    ┌────────────────────┐         ┌────────────────────┐
    │  Sparse Index      │         │ Dense Vector Store │
    │     (BM25)         │         │    (ChromaDB)      │
    └──────────┬─────────┘         └──────────┬─────────┘
               │                              │
               └──────────────┬───────────────┘
                              │ Hybrid Fusion (RRF / Convex Score Combination)
                              ▼
                   ┌──────────────────────┐
                   │ Cross-Encoder        │
                   │ Reranker (PyTorch/HF)│
                   └──────────┬───────────┘
                              │ Top-K Reranked Chunks
                              ▼
                   ┌──────────────────────┐
                   │ Citation-Backed      │
                   │ Grounded Generator   │
                   └──────────────────────┘
```

---

## 🚀 Quick Start with Docker

The fastest way to deploy the entire stack:

```bash
# Clone the repository
git clone https://github.com/yourusername/hybrid-retrieval-rag.git
cd hybrid-retrieval-rag

# Start full-stack microservices (FastAPI + Streamlit + ChromaDB)
docker-compose up --build
```

- **Streamlit Web UI**: [http://localhost:8501](http://localhost:8501)
- **FastAPI Interactive Swagger Docs**: [http://localhost:8000/docs](http://localhost:8000/docs)
- **API ReDoc**: [http://localhost:8000/redoc](http://localhost:8000/redoc)

---

## 💻 Local Setup & Development

### 1. Environment Configuration

```bash
# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Copy environment variables
cp .env.example .env
```

### 2. Start the Backend Service

```bash
uvicorn backend.app.main:app --host 0.0.0.0 --port 8000 --reload
```

### 3. Start the Frontend UI

```bash
streamlit run frontend/app.py --server.port 8501
```

---

## 📡 REST API Reference

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/api/v1/documents/upload` | Uploads and indexes documents with custom chunking strategy (`recursive`, `sliding_window`, `fixed`). |
| `POST` | `/api/v1/query` | Executes hybrid search, cross-encoder reranking, and citation-backed answer generation. |
| `GET` | `/api/v1/metrics` | Returns p50/p90/p95/p99 latency percentiles, stage breakdowns, and cache telemetry. |
| `GET` | `/api/v1/documents` | Lists all indexed documents and chunk statistics. |
| `DELETE` | `/api/v1/documents/{doc_id}` | Deletes a document from dense and sparse stores. |
| `POST` | `/api/v1/documents/reset` | Resets all indices, caches, and telemetry metrics. |
| `GET` | `/health` | Healthcheck probe. |

### Example Query Request (`POST /api/v1/query`)

```json
{
  "query": "How does Raft consensus handle leader election and split votes?",
  "top_k_dense": 10,
  "top_k_sparse": 10,
  "top_k_rerank": 5,
  "hybrid_alpha": 0.5,
  "enable_reranker": true,
  "use_cache": true,
  "llm_provider": "local_extractive"
}
```

### Example Query Response

```json
{
  "query": "How does Raft consensus handle leader election and split votes?",
  "answer": "According to the indexed documents, regarding 'How does Raft consensus handle leader election and split votes?':\n\n• Raft uses randomized election timers to avoid split votes. [1]\n\n• A new leader must be chosen when an existing leader fails. [1]",
  "citations": [
    {
      "citation_id": 1,
      "label": "[1]",
      "chunk_id": "doc_123_chunk_0",
      "doc_id": "doc_123",
      "filename": "distributed_systems_guide.md",
      "snippet": "Leader Election: A new leader must be chosen when an existing leader fails. Raft uses randomized election timers to avoid split votes...",
      "relevance_score": 0.9421
    }
  ],
  "retrieved_chunks": [
    {
      "chunk_id": "doc_123_chunk_0",
      "doc_id": "doc_123",
      "filename": "distributed_systems_guide.md",
      "chunk_index": 0,
      "text": "Leader Election: A new leader must be chosen when an existing leader fails. Raft uses randomized election timers to avoid split votes.",
      "dense_score": 0.8912,
      "sparse_score": 0.9124,
      "fused_score": 0.9018,
      "rerank_score": 0.9421,
      "final_rank": 1,
      "metadata": {}
    }
  ],
  "latencies": {
    "embedding_ms": 1.2,
    "dense_search_ms": 4.5,
    "sparse_search_ms": 1.1,
    "fusion_ms": 0.3,
    "rerank_ms": 12.8,
    "generation_ms": 1.5,
    "total_pipeline_ms": 21.4
  },
  "cache_hit": false,
  "groundedness_score": 0.96
}
```

---

## 🧪 Running Automated Tests

A comprehensive unit and integration test suite covers chunking algorithms, BM25 indexing, ChromaDB embeddings, RRF fusion, Cross-Encoder reranker, LRU caching, and API endpoints.

```bash
# Run pytest with verbose logging
PYTHONPATH=. pytest backend/tests -v
```

---

## 📁 Repository Structure

```
RAG/
├── backend/
│   ├── app/
│   │   ├── __init__.py
│   │   ├── main.py                  # FastAPI entrypoint & middleware
│   │   ├── config.py                # Pydantic environment configuration
│   │   ├── api/
│   │   │   ├── __init__.py
│   │   │   ├── routes.py            # API endpoints & controller logic
│   │   │   └── schemas.py           # Pydantic request/response schemas
│   │   ├── core/
│   │   │   ├── __init__.py
│   │   │   ├── parsers.py           # PDF, DOCX, TXT, MD parsers
│   │   │   ├── chunking.py          # Recursive, Sliding Window, Fixed splitters
│   │   │   ├── cache.py             # Thread-safe LRU KV Cache with TTL
│   │   │   └── metrics.py           # p50/p95 latency timer & telemetry
│   │   ├── retrieval/
│   │   │   ├── __init__.py
│   │   │   ├── dense.py             # ChromaDB vector store & HF embeddings
│   │   │   ├── sparse.py            # BM25Okapi sparse lexical engine
│   │   │   ├── hybrid.py            # RRF & Convex hybrid fusion pipeline
│   │   │   └── reranker.py          # PyTorch Cross-Encoder Re-ranker
│   │   └── generation/
│   │       ├── __init__.py
│   │       ├── generator.py         # Citation generator & LLM backends
│   │       └── prompts.py           # Prompt templates & system instructions
│   ├── tests/
│   │   ├── __init__.py
│   │   ├── test_chunking.py         # Chunking unit tests
│   │   ├── test_retrieval.py        # Dense/Sparse/Hybrid/Reranker tests
│   │   ├── test_cache_metrics.py    # LRU cache & percentile tests
│   │   └── test_api.py              # FastAPI end-to-end integration tests
│   ├── requirements.txt             # Backend dependencies
│   └── Dockerfile                   # Backend Docker container specification
├── frontend/
│   ├── app.py                       # Streamlit multi-tab application
│   ├── config.py                    # Frontend configuration
│   ├── utils.py                     # API client wrapper
│   ├── requirements.txt             # Frontend dependencies
│   └── Dockerfile                   # Frontend Docker container specification
├── sample_data/                     # Sample unstructured context files
│   ├── cloud_computing_architecture.txt
│   ├── distributed_systems_guide.md
│   └── mlops_best_practices.txt
├── docker-compose.yml               # Multi-service container orchestration
├── .env.example                     # Environment template
├── .gitignore                       # Git ignore configuration
├── requirements.txt                 # Unified requirements file
└── README.md                        # Documentation
```

---

## 📜 License
MIT License. Free for commercial and research use.
# Hybrid-Retrieval-RAG-System
