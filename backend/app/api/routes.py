import uuid
import datetime
from typing import List, Optional
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Query
from backend.app.api.schemas import (
    DocumentUploadResponse,
    QueryRequest,
    QueryResponse,
    RetrievedChunk,
    Citation,
    LatencyBreakdown,
    SystemMetricsResponse,
    DocumentListResponse,
    DocumentInfo,
    CacheStats,
    LatencyPercentiles
)
from backend.app.core.parsers import extract_text_from_file
from backend.app.core.chunking import DocumentChunker
from backend.app.core.cache import global_cache
from backend.app.core.metrics import global_metrics, LatencyTimer
from backend.app.retrieval.dense import dense_store
from backend.app.retrieval.sparse import sparse_store
from backend.app.retrieval.hybrid import hybrid_pipeline
from backend.app.generation.generator import generator

router = APIRouter()

# In-memory document registry to track ingested document metadata
_doc_registry = {}

@router.post("/documents/upload", response_model=DocumentUploadResponse)
async def upload_document(
    file: UploadFile = File(...),
    strategy: str = Form("recursive"),
    chunk_size: int = Form(500),
    chunk_overlap: int = Form(100)
):
    try:
        content = await file.read()
        if not content:
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")

        filename = file.filename or "uploaded_doc.txt"
        doc_id = str(uuid.uuid4())[:8]

        # 1. Parse text & extract metadata
        text, file_meta = extract_text_from_file(content, filename)
        if not text.strip():
            raise HTTPException(status_code=400, detail=f"No text could be extracted from {filename}")

        # 2. Chunk document
        chunks = DocumentChunker.chunk_document(
            text=text,
            doc_id=doc_id,
            filename=filename,
            strategy=strategy,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            base_metadata=file_meta
        )

        if not chunks:
            raise HTTPException(status_code=400, detail="Document chunking produced 0 chunks.")

        # 3. Ingest into Dense Vector Store (ChromaDB)
        dense_store.add_chunks(chunks)

        # 4. Ingest into Sparse Index (BM25)
        sparse_store.add_chunks(chunks)

        # 5. Record in registry
        _doc_registry[doc_id] = {
            "doc_id": doc_id,
            "filename": filename,
            "num_chunks": len(chunks),
            "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "metadata": file_meta
        }

        # Clear query cache on new document ingestion
        global_cache.clear()

        return DocumentUploadResponse(
            filename=filename,
            doc_id=doc_id,
            num_chunks=len(chunks),
            strategy=strategy,
            status="success",
            message=f"Successfully indexed '{filename}' into {len(chunks)} chunks."
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process document: {str(e)}")

@router.post("/query", response_model=QueryResponse)
async def query_hybrid_rag(request: QueryRequest):
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    total_timer = LatencyTimer()
    total_timer.__enter__()

    cache_key = global_cache.generate_key("query", request.model_dump(exclude={"custom_api_key"}))
    
    # 1. Check KV Cache
    if request.use_cache:
        cached_result = global_cache.get(cache_key)
        if cached_result is not None:
            total_timer.__exit__(None, None, None)
            cached_result["cache_hit"] = True
            cached_result["latencies"]["total_pipeline_ms"] = round(total_timer.elapsed_ms, 2)
            # Record cached query metric
            global_metrics.record_query(cached_result["latencies"])
            return QueryResponse(**cached_result)

    # 2. Hybrid Retrieval (Dense + Sparse + RRF + Reranker)
    retrieved_raw, stage_latencies = hybrid_pipeline.retrieve(
        query=request.query,
        top_k_dense=request.top_k_dense,
        top_k_sparse=request.top_k_sparse,
        top_k_rerank=request.top_k_rerank,
        hybrid_alpha=request.hybrid_alpha,
        enable_reranker=request.enable_reranker
    )

    # Format RetrievedChunk models
    retrieved_chunks: List[RetrievedChunk] = []
    for idx, c in enumerate(retrieved_raw):
        retrieved_chunks.append(
            RetrievedChunk(
                chunk_id=c.get("chunk_id", ""),
                doc_id=c.get("doc_id", ""),
                filename=c.get("filename", ""),
                chunk_index=c.get("chunk_index", 0),
                text=c.get("text", ""),
                dense_score=c.get("dense_score"),
                sparse_score=c.get("sparse_score"),
                fused_score=c.get("fused_score"),
                rerank_score=c.get("rerank_score"),
                final_rank=c.get("final_rank", idx + 1),
                metadata=c.get("metadata", {})
            )
        )

    # 3. Citation-Backed Answer Generation
    answer, citations, groundedness, gen_latency_ms = generator.generate_answer(
        query=request.query,
        retrieved_chunks=retrieved_raw,
        provider=request.llm_provider,
        custom_api_key=request.custom_api_key
    )
    stage_latencies["generation_ms"] = gen_latency_ms

    total_timer.__exit__(None, None, None)
    stage_latencies["total_pipeline_ms"] = round(total_timer.elapsed_ms, 2)

    # 4. Record Metrics
    global_metrics.record_query(stage_latencies)

    latencies_obj = LatencyBreakdown(**stage_latencies)
    
    response_payload = {
        "query": request.query,
        "answer": answer,
        "citations": [c.model_dump() for c in citations],
        "retrieved_chunks": [r.model_dump() for r in retrieved_chunks],
        "latencies": latencies_obj.model_dump(),
        "cache_hit": False,
        "groundedness_score": groundedness
    }

    # 5. Store in KV Cache
    if request.use_cache:
        global_cache.set(cache_key, response_payload)

    return QueryResponse(
        query=request.query,
        answer=answer,
        citations=citations,
        retrieved_chunks=retrieved_chunks,
        latencies=latencies_obj,
        cache_hit=False,
        groundedness_score=groundedness
    )

@router.get("/metrics", response_model=SystemMetricsResponse)
async def get_system_metrics():
    report = global_metrics.get_full_report()
    cache_stats = global_cache.get_stats()
    
    total_docs = len(_doc_registry)
    total_chunks = dense_store.count()

    return SystemMetricsResponse(
        total_queries=report["total_queries"],
        latency_percentiles=LatencyPercentiles(**report["latency_percentiles"]),
        stage_breakdown_avg=report["stage_breakdown_avg"],
        cache_metrics=CacheStats(**cache_stats),
        total_indexed_documents=total_docs,
        total_indexed_chunks=total_chunks
    )

@router.get("/documents", response_model=DocumentListResponse)
async def list_documents():
    docs = [
        DocumentInfo(
            doc_id=v["doc_id"],
            filename=v["filename"],
            num_chunks=v["num_chunks"],
            created_at=v["created_at"],
            metadata=v["metadata"]
        )
        for v in _doc_registry.values()
    ]
    return DocumentListResponse(
        documents=docs,
        total_documents=len(docs),
        total_chunks=dense_store.count()
    )

@router.delete("/documents/{doc_id}")
async def delete_document(doc_id: str):
    if doc_id not in _doc_registry:
        raise HTTPException(status_code=404, detail="Document ID not found.")

    dense_store.delete_document(doc_id)
    sparse_store.delete_document(doc_id)
    del _doc_registry[doc_id]
    global_cache.clear()

    return {"status": "success", "message": f"Deleted document {doc_id}."}

@router.post("/documents/reset")
async def reset_database():
    dense_store.reset()
    sparse_store.reset()
    _doc_registry.clear()
    global_cache.clear()
    global_metrics.reset()
    return {"status": "success", "message": "Database and telemetry reset."}
