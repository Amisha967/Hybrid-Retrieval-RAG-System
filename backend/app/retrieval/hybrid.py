from typing import List, Dict, Any, Tuple
from backend.app.retrieval.dense import dense_store
from backend.app.retrieval.sparse import sparse_store
from backend.app.retrieval.reranker import reranker
from backend.app.core.metrics import LatencyTimer

class HybridRetrievalPipeline:
    """
    Orchestrates Dense (ChromaDB) + Sparse (BM25) search with Reciprocal Rank Fusion
    and Cross-Encoder Re-ranking.
    """
    
    @staticmethod
    def reciprocal_rank_fusion(
        dense_results: List[Dict[str, Any]],
        sparse_results: List[Dict[str, Any]],
        k: int = 60
    ) -> List[Dict[str, Any]]:
        fused_scores: Dict[str, float] = {}
        chunks_map: Dict[str, Dict[str, Any]] = {}

        # Dense ranks
        for rank, item in enumerate(dense_results):
            chunk_id = item["chunk_id"]
            chunks_map[chunk_id] = item
            fused_scores[chunk_id] = fused_scores.get(chunk_id, 0.0) + (1.0 / (k + rank + 1))

        # Sparse ranks
        for rank, item in enumerate(sparse_results):
            chunk_id = item["chunk_id"]
            if chunk_id not in chunks_map:
                chunks_map[chunk_id] = item
            else:
                # Merge sparse_score into the chunk entry
                chunks_map[chunk_id]["sparse_score"] = item.get("sparse_score", 0.0)
            fused_scores[chunk_id] = fused_scores.get(chunk_id, 0.0) + (1.0 / (k + rank + 1))

        # Normalize fused scores
        max_fused = max(fused_scores.values()) if fused_scores else 1.0
        
        merged_results = []
        for chunk_id, score in fused_scores.items():
            entry = dict(chunks_map[chunk_id])
            entry["fused_score"] = round(score / max_fused, 4)
            if "dense_score" not in entry:
                entry["dense_score"] = 0.0
            if "sparse_score" not in entry:
                entry["sparse_score"] = 0.0
            merged_results.append(entry)

        merged_results.sort(key=lambda x: x["fused_score"], reverse=True)
        return merged_results

    @staticmethod
    def convex_combination(
        dense_results: List[Dict[str, Any]],
        sparse_results: List[Dict[str, Any]],
        alpha: float = 0.5
    ) -> List[Dict[str, Any]]:
        chunks_map: Dict[str, Dict[str, Any]] = {}

        for item in dense_results:
            chunk_id = item["chunk_id"]
            chunks_map[chunk_id] = dict(item)
            chunks_map[chunk_id]["sparse_score"] = 0.0

        for item in sparse_results:
            chunk_id = item["chunk_id"]
            if chunk_id in chunks_map:
                chunks_map[chunk_id]["sparse_score"] = item.get("sparse_score", 0.0)
            else:
                chunks_map[chunk_id] = dict(item)
                chunks_map[chunk_id]["dense_score"] = 0.0

        merged_results = []
        for chunk_id, entry in chunks_map.items():
            d_score = entry.get("dense_score", 0.0)
            s_score = entry.get("sparse_score", 0.0)
            fused = (alpha * d_score) + ((1.0 - alpha) * s_score)
            entry["fused_score"] = round(fused, 4)
            merged_results.append(entry)

        merged_results.sort(key=lambda x: x["fused_score"], reverse=True)
        return merged_results

    def retrieve(
        self,
        query: str,
        top_k_dense: int = 10,
        top_k_sparse: int = 10,
        top_k_rerank: int = 5,
        hybrid_alpha: float = 0.5,
        enable_reranker: bool = True
    ) -> Tuple[List[Dict[str, Any]], Dict[str, float]]:
        latencies: Dict[str, float] = {}

        # 1. Dense Search
        with LatencyTimer() as dense_timer:
            dense_matches = dense_store.search(query=query, top_k=top_k_dense)
        latencies["dense_search_ms"] = round(dense_timer.elapsed_ms, 2)

        # 2. Sparse Search (BM25)
        with LatencyTimer() as sparse_timer:
            sparse_matches = sparse_store.search(query=query, top_k=top_k_sparse)
        latencies["sparse_search_ms"] = round(sparse_timer.elapsed_ms, 2)

        # 3. Hybrid Fusion
        with LatencyTimer() as fusion_timer:
            # Combine via convex combination using alpha
            fused_candidates = self.convex_combination(
                dense_results=dense_matches,
                sparse_results=sparse_matches,
                alpha=hybrid_alpha
            )
        latencies["fusion_ms"] = round(fusion_timer.elapsed_ms, 2)

        # 4. Cross-Encoder Re-Ranking
        if enable_reranker and fused_candidates:
            with LatencyTimer() as rerank_timer:
                final_chunks = reranker.rerank(
                    query=query,
                    candidates=fused_candidates,
                    top_k=top_k_rerank
                )
            latencies["rerank_ms"] = round(rerank_timer.elapsed_ms, 2)
        else:
            final_chunks = fused_candidates[:top_k_rerank]
            for idx, c in enumerate(final_chunks):
                c["rerank_score"] = c.get("fused_score", 0.0)
                c["final_rank"] = idx + 1
            latencies["rerank_ms"] = 0.0

        return final_chunks, latencies

hybrid_pipeline = HybridRetrievalPipeline()
