from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional

class ChunkingConfig(BaseModel):
    strategy: str = Field(default="recursive", description="Chunking strategy: 'recursive', 'fixed', or 'sliding_window'")
    chunk_size: int = Field(default=500, description="Target size of each chunk in characters")
    chunk_overlap: int = Field(default=100, description="Character overlap between consecutive chunks")

class DocumentUploadResponse(BaseModel):
    filename: str
    doc_id: str
    num_chunks: int
    strategy: str
    status: str
    message: str

class RetrievedChunk(BaseModel):
    chunk_id: str
    doc_id: str
    filename: str
    chunk_index: int
    text: str
    dense_score: Optional[float] = None
    sparse_score: Optional[float] = None
    fused_score: Optional[float] = None
    rerank_score: Optional[float] = None
    final_rank: int
    metadata: Dict[str, Any] = Field(default_factory=dict)

class Citation(BaseModel):
    citation_id: int
    label: str
    chunk_id: str
    doc_id: str
    filename: str
    snippet: str
    relevance_score: float

class LatencyBreakdown(BaseModel):
    embedding_ms: float = 0.0
    dense_search_ms: float = 0.0
    sparse_search_ms: float = 0.0
    fusion_ms: float = 0.0
    rerank_ms: float = 0.0
    generation_ms: float = 0.0
    total_pipeline_ms: float = 0.0

class QueryRequest(BaseModel):
    query: str = Field(..., description="The user query / question")
    top_k_dense: int = Field(default=10, description="Number of dense vector matches to retrieve")
    top_k_sparse: int = Field(default=10, description="Number of BM25 lexical matches to retrieve")
    top_k_rerank: int = Field(default=5, description="Final number of top chunks after reranking")
    hybrid_alpha: float = Field(default=0.5, ge=0.0, le=1.0, description="Weight for Dense vs BM25: alpha*Dense + (1-alpha)*BM25")
    enable_reranker: bool = Field(default=True, description="Whether to apply Cross-Encoder reranking")
    use_cache: bool = Field(default=True, description="Whether to utilize KV Query Caching")
    llm_provider: Optional[str] = Field(default=None, description="LLM provider: 'local_extractive', 'gemini', 'openai', 'huggingface'")
    custom_api_key: Optional[str] = Field(default=None, description="Optional custom API key for external LLM provider")

class QueryResponse(BaseModel):
    query: str
    answer: str
    citations: List[Citation]
    retrieved_chunks: List[RetrievedChunk]
    latencies: LatencyBreakdown
    cache_hit: bool
    groundedness_score: float

class CacheStats(BaseModel):
    hits: int
    misses: int
    total_entries: int
    max_size: int
    hit_rate_pct: float

class LatencyPercentiles(BaseModel):
    p50_ms: float
    p90_ms: float
    p95_ms: float
    p99_ms: float
    avg_ms: float
    min_ms: float
    max_ms: float

class SystemMetricsResponse(BaseModel):
    total_queries: int
    latency_percentiles: LatencyPercentiles
    stage_breakdown_avg: Dict[str, float]
    cache_metrics: CacheStats
    total_indexed_documents: int
    total_indexed_chunks: int

class DocumentInfo(BaseModel):
    doc_id: str
    filename: str
    num_chunks: int
    created_at: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

class DocumentListResponse(BaseModel):
    documents: List[DocumentInfo]
    total_documents: int
    total_chunks: int
