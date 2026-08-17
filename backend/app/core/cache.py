import time
import hashlib
import json
import threading
from collections import OrderedDict
from typing import Any, Optional, Dict, Tuple

class LRUKVCache:
    """
    Thread-safe in-memory LRU Key-Value Cache with Time-To-Live (TTL) support
    and telemetry stats (hits, misses, evictions).
    """
    def __init__(self, max_size: int = 1000, ttl_seconds: int = 3600):
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self._cache: OrderedDict[str, Tuple[Any, float]] = OrderedDict()
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0
        self.evictions = 0

    @staticmethod
    def generate_key(prefix: str, payload: Any) -> str:
        """
        Creates a deterministic hash key from a prefix and JSON-serializable payload or string.
        """
        if isinstance(payload, str):
            serialized = payload.strip().lower()
        else:
            try:
                serialized = json.dumps(payload, sort_keys=True)
            except Exception:
                serialized = str(payload)
        
        digest = hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:16]
        return f"{prefix}:{digest}"

    def get(self, key: str) -> Optional[Any]:
        with self._lock:
            if key not in self._cache:
                self.misses += 1
                return None
            
            value, timestamp = self._cache[key]
            
            # Check TTL
            if time.time() - timestamp > self.ttl_seconds:
                del self._cache[key]
                self.misses += 1
                return None
            
            # Move to end (most recently used)
            self._cache.move_to_end(key)
            self.hits += 1
            return value

    def set(self, key: str, value: Any) -> None:
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
            self._cache[key] = (value, time.time())
            
            # Evict if exceeding max size
            if len(self._cache) > self.max_size:
                self._cache.popitem(last=False)
                self.evictions += 1

    def clear(self) -> None:
        with self._lock:
            self._cache.clear()
            self.hits = 0
            self.misses = 0
            self.evictions = 0

    def get_stats(self) -> Dict[str, Any]:
        with self._lock:
            total_reqs = self.hits + self.misses
            hit_rate = (self.hits / total_reqs * 100.0) if total_reqs > 0 else 0.0
            return {
                "hits": self.hits,
                "misses": self.misses,
                "total_entries": len(self._cache),
                "max_size": self.max_size,
                "hit_rate_pct": round(hit_rate, 2),
                "evictions": self.evictions
            }

# Global singleton cache instance
global_cache = LRUKVCache(max_size=1000, ttl_seconds=3600)
