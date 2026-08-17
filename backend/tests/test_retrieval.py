import pytest
from backend.app.core.chunking import TextChunk
from backend.app.retrieval.sparse import SparseBM25Store, PureBM25Okapi, tokenize_text
from backend.app.retrieval.dense import DenseVectorStore
from backend.app.retrieval.reranker import CrossEncoderReranker
from backend.app.retrieval.hybrid import HybridRetrievalPipeline

def test_bm25_tokenization_and_scoring():
    corpus = [
        "Distributed consensus protocols like Raft and Paxos ensure state machine replication.",
        "Cloud storage models include object storage, block storage, and file systems.",
        "Deep learning models require GPU acceleration and continuous feature monitoring."
    ]
    tokens = [tokenize_text(doc) for doc in corpus]
    bm25 = PureBM25Okapi(tokens)
    
    q_tokens = tokenize_text("Raft consensus leader election")
    scores = bm25.get_scores(q_tokens)
    
    assert len(scores) == 3
    # First document should have highest BM25 score
    assert scores[0] > scores[1]
    assert scores[0] > scores[2]

def test_sparse_bm25_store():
    store = SparseBM25Store()
    store.reset()
    
    chunks = [
        TextChunk(chunk_id="c1", doc_id="d1", text="Machine learning model monitoring detects data drift.", chunk_index=0, start_char=0, end_char=50),
        TextChunk(chunk_id="c2", doc_id="d1", text="Zero trust security requires continuous verification.", chunk_index=1, start_char=51, end_char=100)
    ]
    store.add_chunks(chunks)
    assert store.count() == 2
    
    results = store.search("data drift monitoring", top_k=2)
    assert len(results) > 0
    assert results[0]["chunk_id"] == "c1"
    
    store.delete_document("d1")
    assert store.count() == 0

def test_dense_vector_store():
    store = DenseVectorStore()
    store.reset()
    
    chunks = [
        TextChunk(chunk_id="c1", doc_id="d1", text="Kubernetes automates container orchestration and scaling.", chunk_index=0, start_char=0, end_char=50),
        TextChunk(chunk_id="c2", doc_id="d2", text="PostgreSQL relational database ACID transaction guarantees.", chunk_index=0, start_char=0, end_char=50)
    ]
    store.add_chunks(chunks)
    assert store.count() == 2
    
    results = store.search("container orchestration", top_k=2)
    assert len(results) > 0
    assert "dense_score" in results[0]

def test_hybrid_fusion():
    dense_res = [
        {"chunk_id": "c1", "dense_score": 0.9, "text": "Doc A"},
        {"chunk_id": "c2", "dense_score": 0.6, "text": "Doc B"}
    ]
    sparse_res = [
        {"chunk_id": "c2", "sparse_score": 0.95, "text": "Doc B"},
        {"chunk_id": "c3", "sparse_score": 0.7, "text": "Doc C"}
    ]
    
    # RRF test
    fused_rrf = HybridRetrievalPipeline.reciprocal_rank_fusion(dense_res, sparse_res, k=60)
    assert len(fused_rrf) == 3
    assert all("fused_score" in x for x in fused_rrf)
    
    # Convex combination test
    fused_convex = HybridRetrievalPipeline.convex_combination(dense_res, sparse_res, alpha=0.5)
    assert len(fused_convex) == 3
    # c2 is present in both with high scores so should be top
    assert fused_convex[0]["chunk_id"] == "c2"

def test_cross_encoder_reranker():
    reranker = CrossEncoderReranker()
    query = "consensus in distributed systems"
    candidates = [
        {"chunk_id": "c1", "text": "Python web frameworks include FastAPI and Django.", "fused_score": 0.8},
        {"chunk_id": "c2", "text": "Raft achieves consensus in distributed systems via leader election.", "fused_score": 0.7}
    ]
    
    reranked = reranker.rerank(query, candidates, top_k=2)
    assert len(reranked) == 2
    # c2 contains exact query terms and distributed consensus facts
    assert reranked[0]["chunk_id"] == "c2"
    assert reranked[0]["final_rank"] == 1
