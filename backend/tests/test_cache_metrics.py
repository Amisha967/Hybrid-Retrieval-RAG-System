import time
import pytest
from backend.app.core.cache import LRUKVCache
from backend.app.core.metrics import MetricsTracker, LatencyTimer

def test_lru_kv_cache_operations():
    cache = LRUKVCache(max_size=3, ttl_seconds=10)
    
    # Test set & get
    key1 = LRUKVCache.generate_key("q", "hello world")
    cache.set(key1, {"ans": "123"})
    assert cache.get(key1) == {"ans": "123"}
    assert cache.hits == 1
    
    # Test cache miss
    assert cache.get("non_existent_key") is None
    assert cache.misses == 1
    
    # Test eviction when exceeding max size (3)
    key2 = LRUKVCache.generate_key("q", "q2")
    key3 = LRUKVCache.generate_key("q", "q3")
    key4 = LRUKVCache.generate_key("q", "q4")
    
    cache.set(key2, "v2")
    cache.set(key3, "v3")
    cache.set(key4, "v4")  # key1 should be evicted as LRU
    
    assert cache.get(key1) is None
    assert cache.get(key4) == "v4"
    assert cache.evictions >= 1
    
    stats = cache.get_stats()
    assert stats["total_entries"] == 3
    assert stats["max_size"] == 3

def test_lru_cache_ttl():
    cache = LRUKVCache(max_size=10, ttl_seconds=0.1)
    key = "temp_key"
    cache.set(key, "data")
    assert cache.get(key) == "data"
    
    time.sleep(0.15)
    assert cache.get(key) is None

def test_metrics_tracker_percentiles():
    tracker = MetricsTracker(max_history=100)
    tracker.reset()
    
    # Feed deterministic latencies: 10, 20, 30, 40, 50, 60, 70, 80, 90, 100 ms
    for val in range(10, 110, 10):
        tracker.record_query({
            "embedding_ms": val * 0.1,
            "dense_search_ms": val * 0.2,
            "sparse_search_ms": val * 0.2,
            "fusion_ms": val * 0.1,
            "rerank_ms": val * 0.2,
            "generation_ms": val * 0.2,
            "total_pipeline_ms": float(val)
        })
        
    pct = tracker.get_percentiles()
    assert pct["p50_ms"] == pytest.approx(55.0, abs=5.0)
    assert pct["p95_ms"] == pytest.approx(95.5, abs=5.0)
    assert pct["min_ms"] == 10.0
    assert pct["max_ms"] == 100.0
    
    stage_avg = tracker.get_stage_averages()
    assert stage_avg["total_pipeline_ms"] == 55.0
