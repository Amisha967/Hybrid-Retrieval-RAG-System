import math
import threading
from typing import List, Dict, Any, Tuple
from backend.app.config import settings

class CrossEncoderReranker:
    """
    Cross-Encoder re-ranker for precise query-document relevance scoring.
    """
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(CrossEncoderReranker, cls).__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self):
        if self._initialized:
            return
        
        self.model_name = settings.RERANKER_MODEL_NAME
        self.device = settings.RERANKER_DEVICE
        self._model = None
        self._initialized = True

    def _get_model(self):
        if self._model is None:
            try:
                from sentence_transformers import CrossEncoder
                self._model = CrossEncoder(self.model_name, device=self.device)
            except Exception as e:
                print(f"[Reranker] Loading fallback cross-encoder due to: {e}")
                self._model = self._FallbackCrossEncoder()
        return self._model

    class _FallbackCrossEncoder:
        """Heuristic fallback cross-encoder computing exact match & token density alignment"""
        def predict(self, pairs: List[List[str]]) -> List[float]:
            scores = []
            for query, doc in pairs:
                q_words = set(query.lower().split())
                d_words = doc.lower().split()
                if not q_words or not d_words:
                    scores.append(0.0)
                    continue
                
                # Match count and consecutive phrase check
                matches = sum(1 for w in d_words if w in q_words)
                exact_phrase_bonus = 2.0 if query.lower() in doc.lower() else 0.0
                
                raw_score = (matches / len(q_words)) * 2.0 + exact_phrase_bonus
                scores.append(raw_score)
            return scores

    @staticmethod
    def _sigmoid(x: float) -> float:
        try:
            return 1.0 / (1.0 + math.exp(-x))
        except OverflowError:
            return 0.0 if x < 0 else 1.0

    def rerank(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        if not candidates:
            return []
        
        model = self._get_model()
        pairs = [[query, item.get("text", "")] for item in candidates]
        
        raw_scores = model.predict(pairs)
        
        reranked_candidates = []
        for item, score in zip(candidates, raw_scores):
            score_val = float(score)
            # Sigmoid normalization if logit scale or min-max clip
            calibrated_score = self._sigmoid(score_val) if abs(score_val) > 1.0 else max(0.0, min(1.0, score_val))
            
            entry = dict(item)
            entry["rerank_score"] = round(calibrated_score, 4)
            entry["raw_rerank_score"] = round(score_val, 4)
            reranked_candidates.append(entry)

        # Sort descending by rerank_score
        reranked_candidates.sort(key=lambda x: x["rerank_score"], reverse=True)
        
        # Assign 1-indexed final rank
        for idx, item in enumerate(reranked_candidates):
            item["final_rank"] = idx + 1

        return reranked_candidates[:top_k]

reranker = CrossEncoderReranker()
