import os
import re
import math
import pickle
import threading
from typing import List, Dict, Any, Optional
from backend.app.config import settings
from backend.app.core.chunking import TextChunk

# Standard English stopwords
STOPWORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and",
    "any", "are", "aren't", "as", "at", "be", "because", "been", "before", "being",
    "below", "between", "both", "but", "by", "can", "can't", "cannot", "could",
    "couldn't", "did", "didn't", "do", "does", "doesn't", "doing", "don't", "down",
    "during", "each", "few", "for", "from", "further", "had", "hadn't", "has",
    "hasn't", "have", "haven't", "having", "he", "he'd", "he'll", "he's", "her",
    "here", "here's", "hers", "herself", "him", "himself", "his", "how", "how's",
    "i", "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is", "isn't", "it",
    "it's", "its", "itself", "let's", "me", "more", "most", "mustn't", "my",
    "myself", "no", "nor", "not", "of", "off", "on", "once", "only", "or",
    "other", "ought", "our", "ours", "ourselves", "out", "over", "own", "same",
    "shan't", "she", "she'd", "she'll", "she's", "should", "shouldn't", "so",
    "some", "such", "than", "that", "that's", "the", "their", "theirs", "them",
    "themselves", "then", "there", "there's", "these", "they", "they'd", "they'll",
    "they're", "they've", "this", "those", "through", "to", "too", "under", "until",
    "up", "very", "was", "wasn't", "we", "we'd", "we'll", "we're", "we've", "were",
    "weren't", "what", "what's", "when", "when's", "where", "where's", "which",
    "while", "who", "who's", "whom", "why", "why's", "with", "won't", "would",
    "wouldn't", "you", "you'd", "you'll", "you're", "you've", "your", "yours",
    "yourself", "yourselves"
}

def tokenize_text(text: str) -> List[str]:
    """Tokenizes text, strips punctuation, lowercases, and removes stopwords."""
    words = re.findall(r'\b[a-zA-Z0-9_]+\b', text.lower())
    return [w for w in words if w not in STOPWORDS and len(w) > 1]

class PureBM25Okapi:
    """
    Self-contained, fast implementation of BM25Okapi ranking algorithm.
    """
    def __init__(self, corpus_tokens: List[List[str]], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.corpus_size = len(corpus_tokens)
        self.doc_lengths = [len(doc) for doc in corpus_tokens]
        self.avgdl = sum(self.doc_lengths) / self.corpus_size if self.corpus_size > 0 else 0
        
        # Word frequencies per document and document frequency across corpus
        self.doc_freqs: List[Dict[str, int]] = []
        self.df: Dict[str, int] = {}
        self.idf: Dict[str, float] = {}
        
        for doc in corpus_tokens:
            freq: Dict[str, int] = {}
            for word in doc:
                freq[word] = freq.get(word, 0) + 1
            self.doc_freqs.append(freq)
            
            for word in freq.keys():
                self.df[word] = self.df.get(word, 0) + 1

        for word, freq in self.df.items():
            # Standard Lucene/BM25 IDF formula
            self.idf[word] = math.log((self.corpus_size - freq + 0.5) / (freq + 0.5) + 1)

    def get_scores(self, query_tokens: List[str]) -> List[float]:
        scores = [0.0] * self.corpus_size
        if self.corpus_size == 0 or self.avgdl == 0:
            return scores

        for word in query_tokens:
            if word not in self.idf:
                continue
            idf_val = self.idf[word]
            for i, doc_freq in enumerate(self.doc_freqs):
                if word in doc_freq:
                    tf = doc_freq[word]
                    doc_len = self.doc_lengths[i]
                    numerator = tf * (self.k1 + 1)
                    denominator = tf + self.k1 * (1 - self.b + self.b * (doc_len / self.avgdl))
                    scores[i] += idf_val * (numerator / denominator)
        return scores

class SparseBM25Store:
    """
    Persistent BM25 sparse search engine managing tokenized corpora and chunk lookups.
    """
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(SparseBM25Store, cls).__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self):
        if self._initialized:
            return
        
        self.persist_path = settings.BM25_PERSIST_PATH
        os.makedirs(os.path.dirname(os.path.abspath(self.persist_path)), exist_ok=True)
        
        self.chunks_data: List[Dict[str, Any]] = []
        self.corpus_tokens: List[List[str]] = []
        self.bm25_model: Optional[PureBM25Okapi] = None
        
        self._load_from_disk()
        self._rebuild_index()
        self._initialized = True

    def _rebuild_index(self):
        if self.corpus_tokens:
            self.bm25_model = PureBM25Okapi(self.corpus_tokens)
        else:
            self.bm25_model = None

    def add_chunks(self, chunks: List[TextChunk]) -> None:
        with self._lock:
            for c in chunks:
                # Check for existing chunk id replacement
                existing_idx = next((i for i, ch in enumerate(self.chunks_data) if ch["chunk_id"] == c.chunk_id), None)
                tokens = tokenize_text(c.text)
                
                chunk_entry = {
                    "chunk_id": c.chunk_id,
                    "doc_id": c.doc_id,
                    "filename": c.metadata.get("filename", ""),
                    "chunk_index": c.chunk_index,
                    "text": c.text,
                    "metadata": c.metadata
                }
                
                if existing_idx is not None:
                    self.chunks_data[existing_idx] = chunk_entry
                    self.corpus_tokens[existing_idx] = tokens
                else:
                    self.chunks_data.append(chunk_entry)
                    self.corpus_tokens.append(tokens)
            
            self._rebuild_index()
            self._save_to_disk()

    def search(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        with self._lock:
            if not self.bm25_model or not self.chunks_data:
                return []
            
            query_tokens = tokenize_text(query)
            if not query_tokens:
                return []

            raw_scores = self.bm25_model.get_scores(query_tokens)
            max_score = max(raw_scores) if raw_scores else 0.0
            
            # Pair scores with chunk objects
            scored_chunks = []
            for i, score in enumerate(raw_scores):
                if score > 0.0:
                    normalized_score = (score / max_score) if max_score > 0 else 0.0
                    chunk_entry = self.chunks_data[i]
                    scored_chunks.append({
                        "chunk_id": chunk_entry["chunk_id"],
                        "text": chunk_entry["text"],
                        "sparse_score": round(float(normalized_score), 4),
                        "raw_bm25_score": round(float(score), 4),
                        "metadata": chunk_entry["metadata"],
                        "doc_id": chunk_entry["doc_id"],
                        "filename": chunk_entry["filename"],
                        "chunk_index": chunk_entry["chunk_index"]
                    })

            # Sort descending
            scored_chunks.sort(key=lambda x: x["sparse_score"], reverse=True)
            return scored_chunks[:top_k]

    def delete_document(self, doc_id: str) -> None:
        with self._lock:
            new_chunks = []
            new_tokens = []
            for ch, tok in zip(self.chunks_data, self.corpus_tokens):
                if ch["doc_id"] != doc_id:
                    new_chunks.append(ch)
                    new_tokens.append(tok)
            self.chunks_data = new_chunks
            self.corpus_tokens = new_tokens
            self._rebuild_index()
            self._save_to_disk()

    def count(self) -> int:
        with self._lock:
            return len(self.chunks_data)

    def reset(self) -> None:
        with self._lock:
            self.chunks_data = []
            self.corpus_tokens = []
            self.bm25_model = None
            if os.path.exists(self.persist_path):
                os.remove(self.persist_path)

    def _save_to_disk(self) -> None:
        try:
            with open(self.persist_path, "wb") as f:
                pickle.dump({"chunks_data": self.chunks_data, "corpus_tokens": self.corpus_tokens}, f)
        except Exception as e:
            print(f"[SparseStore] Failed to persist BM25 index: {e}")

    def _load_from_disk(self) -> None:
        if os.path.exists(self.persist_path):
            try:
                with open(self.persist_path, "rb") as f:
                    data = pickle.load(f)
                    self.chunks_data = data.get("chunks_data", [])
                    self.corpus_tokens = data.get("corpus_tokens", [])
            except Exception as e:
                print(f"[SparseStore] Failed to load BM25 index from disk: {e}")

sparse_store = SparseBM25Store()
