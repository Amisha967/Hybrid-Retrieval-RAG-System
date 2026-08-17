import pytest

try:
    from fastapi.testclient import TestClient
    from backend.app.main import app
    from backend.app.retrieval.dense import dense_store
    from backend.app.retrieval.sparse import sparse_store
    HAS_FASTAPI = True
except ImportError:
    HAS_FASTAPI = False

pytestmark = pytest.mark.skipif(not HAS_FASTAPI, reason="FastAPI or dependencies not installed in global environment")

if HAS_FASTAPI:
    client = TestClient(app)

    @pytest.fixture(autouse=True)
    def setup_teardown():
        dense_store.reset()
        sparse_store.reset()
        yield
        dense_store.reset()
        sparse_store.reset()

    def test_health_endpoint():
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"

    def test_document_upload_and_query_flow():
        sample_text = (
            "Enterprise cloud computing architecture provides high reliability, security, and scalability. "
            "Consensus protocols like Raft ensure fault-tolerant distributed replication across nodes."
        )
        files = {"file": ("cloud_arch.txt", sample_text.encode("utf-8"), "text/plain")}
        data = {"strategy": "recursive", "chunk_size": 100, "chunk_overlap": 20}
        
        upload_res = client.post("/api/v1/documents/upload", files=files, data=data)
        assert upload_res.status_code == 200
        upload_data = upload_res.json()
        assert upload_data["status"] == "success"
        assert upload_data["num_chunks"] > 0
        doc_id = upload_data["doc_id"]

        docs_res = client.get("/api/v1/documents")
        assert docs_res.status_code == 200
        docs_data = docs_res.json()
        assert docs_data["total_documents"] == 1
        assert docs_data["documents"][0]["doc_id"] == doc_id

        query_payload = {
            "query": "What protocols ensure distributed consensus?",
            "top_k_dense": 5,
            "top_k_sparse": 5,
            "top_k_rerank": 3,
            "hybrid_alpha": 0.5,
            "enable_reranker": True,
            "use_cache": True,
            "llm_provider": "local_extractive"
        }
        
        query_res = client.post("/api/v1/query", json=query_payload)
        assert query_res.status_code == 200
        q_data = query_res.json()
        assert "answer" in q_data
        assert len(q_data["retrieved_chunks"]) > 0
        assert len(q_data["citations"]) > 0
        assert q_data["latencies"]["total_pipeline_ms"] > 0
        assert q_data["cache_hit"] is False

        # Test cache hit
        query_res_cached = client.post("/api/v1/query", json=query_payload)
        assert query_res_cached.status_code == 200
        q_cached_data = query_res_cached.json()
        assert q_cached_data["cache_hit"] is True

        # Check metrics
        metrics_res = client.get("/api/v1/metrics")
        assert metrics_res.status_code == 200
        m_data = metrics_res.json()
        assert m_data["total_queries"] >= 2
        assert m_data["cache_metrics"]["hits"] >= 1

        # Delete document
        del_res = client.delete(f"/api/v1/documents/{doc_id}")
        assert del_res.status_code == 200
else:
    def test_placeholder():
        pass
