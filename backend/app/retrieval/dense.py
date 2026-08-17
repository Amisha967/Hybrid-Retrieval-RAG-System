import os
import threading
from typing import List, Dict, Any, Optional, Tuple
from backend.app.config import settings
from backend.app.core.chunking import TextChunk
from backend.app.core.cache import global_cache

class DenseVectorStore:
    """
    Manages dense vector indexing and retrieval using ChromaDB and SentenceTransformer embeddings,
    with an in-memory fallback if ChromaDB is not installed in the runtime environment.
    """
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(DenseVectorStore, cls).__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self):
        if self._initialized:
            return
        
        self.persist_dir = settings.CHROMA_PERSIST_DIRECTORY
        os.makedirs(self.persist_dir, exist_ok=True)
        self.collection_name = settings.CHROMA_COLLECTION_NAME
        
        self.chroma_client = None
        self.collection = None
        self._memory_store: List[Dict[str, Any]] = []

        try:
            import chromadb
            from chromadb.config import Settings as ChromaSettings
            self.chroma_client = chromadb.PersistentClient(
                path=self.persist_dir,
                settings=ChromaSettings(anonymized_telemetry=False, allow_reset=True)
            )
            self.collection = self.chroma_client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"}
            )
        except Exception as e:
            print(f"[DenseStore] ChromaDB initialized in lightweight memory fallback mode: {e}")

        self._embedder = None
        self._initialized = True

    def _get_embedder(self):
        if self._embedder is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._embedder = SentenceTransformer(
                    settings.EMBEDDING_MODEL_NAME,
                    device=settings.EMBEDDING_DEVICE
                )
            except Exception as e:
                self._embedder = self._FallbackEmbedder()
        return self._embedder

    class _FallbackEmbedder:
        """Lightweight 384-dimensional deterministic feature hashing fallback"""
        def encode(self, texts: List[str], **kwargs) -> List[List[float]]:
            import hashlib
            import numpy as np
            embeddings = []
            for t in texts:
                vec = np.zeros(384, dtype=np.float32)
                words = t.lower().split()
                for i, word in enumerate(words):
                    h = int(hashlib.md5(word.encode('utf-8')).hexdigest(), 16)
                    idx = h % 384
                    vec[idx] += 1.0 / (i + 1)
                norm = np.linalg.norm(vec)
                if norm > 0:
                    vec = vec / norm
                embeddings.append(vec.tolist())
            return embeddings

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        embedder = self._get_embedder()
        if hasattr(embedder, "encode"):
            embeddings = embedder.encode(texts, show_progress_bar=False, normalize_embeddings=True)
            if hasattr(embeddings, "tolist"):
                return embeddings.tolist()
            return [list(e) for e in embeddings]
        return [[0.0] * 384 for _ in texts]

    def embed_query(self, query: str) -> List[float]:
        cache_key = global_cache.generate_key("embed", query)
        cached_embedding = global_cache.get(cache_key)
        if cached_embedding is not None:
            return cached_embedding
        
        emb = self.embed_texts([query])[0]
        global_cache.set(cache_key, emb)
        return emb

    def add_chunks(self, chunks: List[TextChunk]) -> None:
        if not chunks:
            return
        
        ids = [c.chunk_id for c in chunks]
        texts = [c.text for c in chunks]
        metadatas = [
            {
                "doc_id": c.doc_id,
                "filename": c.metadata.get("filename", ""),
                "chunk_index": c.chunk_index,
                "start_char": c.start_char,
                "end_char": c.end_char,
                "timestamp": c.metadata.get("timestamp", "")
            }
            for c in chunks
        ]
        embeddings = self.embed_texts(texts)

        if self.collection is not None:
            self.collection.upsert(
                ids=ids,
                embeddings=embeddings,
                documents=texts,
                metadatas=metadatas
            )
        else:
            for idx, c in enumerate(chunks):
                self._memory_store.append({
                    "id": ids[idx],
                    "embedding": embeddings[idx],
                    "document": texts[idx],
                    "metadata": metadatas[idx]
                })

    def search(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        query_embedding = self.embed_query(query)
        
        if self.collection is not None:
            count = self.collection.count()
            if count == 0:
                return []
            k = min(top_k, count)
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=k,
                include=["documents", "metadatas", "distances"]
            )
            formatted_results = []
            if results and "ids" in results and results["ids"]:
                ids_list = results["ids"][0]
                docs_list = results["documents"][0] if "documents" in results else []
                meta_list = results["metadatas"][0] if "metadatas" in results else []
                dist_list = results["distances"][0] if "distances" in results else []

                for i, chunk_id in enumerate(ids_list):
                    distance = dist_list[i] if i < len(dist_list) else 0.0
                    similarity = max(0.0, min(1.0, 1.0 - distance))
                    formatted_results.append({
                        "chunk_id": chunk_id,
                        "text": docs_list[i] if i < len(docs_list) else "",
                        "dense_score": round(float(similarity), 4),
                        "metadata": meta_list[i] if i < len(meta_list) else {},
                        "doc_id": meta_list[i].get("doc_id", "") if i < len(meta_list) else "",
                        "filename": meta_list[i].get("filename", "") if i < len(meta_list) else "",
                        "chunk_index": meta_list[i].get("chunk_index", 0) if i < len(meta_list) else 0
                    })
            return formatted_results

        # In-memory cosine similarity fallback
        import numpy as np
        if not self._memory_store:
            return []
        
        scored = []
        q_vec = np.array(query_embedding)
        for item in self._memory_store:
            i_vec = np.array(item["embedding"])
            denom = (np.linalg.norm(q_vec) * np.linalg.norm(i_vec))
            sim = float(np.dot(q_vec, i_vec) / denom) if denom > 0 else 0.0
            scored.append({
                "chunk_id": item["id"],
                "text": item["document"],
                "dense_score": round(max(0.0, min(1.0, sim)), 4),
                "metadata": item["metadata"],
                "doc_id": item["metadata"].get("doc_id", ""),
                "filename": item["metadata"].get("filename", ""),
                "chunk_index": item["metadata"].get("chunk_index", 0)
            })
        scored.sort(key=lambda x: x["dense_score"], reverse=True)
        return scored[:top_k]

    def delete_document(self, doc_id: str) -> None:
        if self.collection is not None:
            self.collection.delete(where={"doc_id": doc_id})
        else:
            self._memory_store = [x for x in self._memory_store if x["metadata"].get("doc_id") != doc_id]

    def count(self) -> int:
        if self.collection is not None:
            return self.collection.count()
        return len(self._memory_store)

    def reset(self) -> None:
        if self.collection is not None and self.chroma_client is not None:
            self.chroma_client.delete_collection(self.collection_name)
            self.collection = self.chroma_client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"}
            )
        self._memory_store = []

dense_store = DenseVectorStore()
