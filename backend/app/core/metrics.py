import time
import threading
import numpy as np
from typing import Dict, List, Any, Optional
from collections import deque

class LatencyTimer:
    """
    Context manager for measuring execution time of code blocks in milliseconds.
    """
    def __init__(self):
        self.start_time = 0.0
        self.elapsed_ms = 0.0

    def __enter__(self):
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.elapsed_ms = (time.perf_counter() - self.start_time) * 1000.0

class MetricsTracker:
    """
    Thread-safe latency tracker computing p50, p90, p95, p99 percentiles
    and stage-by-stage breakdowns.
    """
    def __init__(self, max_history: int = 5000):
        self._lock = threading.Lock()
        self.max_history = max_history
        self.total_queries = 0
        self.total_pipeline_latencies: deque = deque(maxlen=max_history)
        
        # Stage-specific latencies
        self.stage_latencies: Dict[str, deque] = {
            "embedding_ms": deque(maxlen=max_history),
            "dense_search_ms": deque(maxlen=max_history),
            "sparse_search_ms": deque(maxlen=max_history),
            "fusion_ms": deque(maxlen=max_history),
            "rerank_ms": deque(maxlen=max_history),
            "generation_ms": deque(maxlen=max_history),
            "total_pipeline_ms": deque(maxlen=max_history)
        }

    def record_query(self, latencies: Dict[str, float]) -> None:
        with self._lock:
            self.total_queries += 1
            total_ms = latencies.get("total_pipeline_ms", 0.0)
            self.total_pipeline_latencies.append(total_ms)
            
            for stage, val in latencies.items():
                if stage in self.stage_latencies:
                    self.stage_latencies[stage].append(val)

    def get_percentiles(self) -> Dict[str, float]:
        with self._lock:
            if not self.total_pipeline_latencies:
                return {
                    "p50_ms": 0.0,
                    "p90_ms": 0.0,
                    "p95_ms": 0.0,
                    "p99_ms": 0.0,
                    "avg_ms": 0.0,
                    "min_ms": 0.0,
                    "max_ms": 0.0
                }
            
            data = np.array(list(self.total_pipeline_latencies))
            return {
                "p50_ms": round(float(np.percentile(data, 50)), 2),
                "p90_ms": round(float(np.percentile(data, 90)), 2),
                "p95_ms": round(float(np.percentile(data, 95)), 2),
                "p99_ms": round(float(np.percentile(data, 99)), 2),
                "avg_ms": round(float(np.mean(data)), 2),
                "min_ms": round(float(np.min(data)), 2),
                "max_ms": round(float(np.max(data)), 2)
            }

    def get_stage_averages(self) -> Dict[str, float]:
        with self._lock:
            stage_avgs = {}
            for stage, vals in self.stage_latencies.items():
                if vals:
                    stage_avgs[stage] = round(float(np.mean(vals)), 2)
                else:
                    stage_avgs[stage] = 0.0
            return stage_avgs

    def get_full_report(self) -> Dict[str, Any]:
        return {
            "total_queries": self.total_queries,
            "latency_percentiles": self.get_percentiles(),
            "stage_breakdown_avg": self.get_stage_averages()
        }

    def reset(self) -> None:
        with self._lock:
            self.total_queries = 0
            self.total_pipeline_latencies.clear()
            for key in self.stage_latencies:
                self.stage_latencies[key].clear()

# Global singleton metrics instance
global_metrics = MetricsTracker()
