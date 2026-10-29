import threading
import time
from collections import defaultdict, deque


_lock = threading.Lock()
_request_counts = defaultdict(int)
_error_counts = defaultdict(int)
_response_times = defaultdict(lambda: deque(maxlen=1000))
_start_time = time.time()


def record_request(method, path, status_code, duration_ms):
    """Record request totals, errors, and recent response times."""
    key = (method, path)

    with _lock:
        _request_counts[key] += 1
        _response_times[key].append(float(duration_ms))

        if status_code >= 400:
            _error_counts[key] += 1


def get_metrics():
    """Return a thread-safe snapshot of in-memory API metrics."""
    with _lock:
        endpoints = []

        for method, path in sorted(_request_counts):
            times = sorted(_response_times[(method, path)])

            endpoints.append({
                "method": method,
                "path": path,
                "request_count": _request_counts[(method, path)],
                "error_count": _error_counts[(method, path)],
                "average_response_ms": round(
                    sum(times) / len(times), 2
                ) if times else 0.0,
                "max_response_ms": round(max(times), 2)
                if times else 0.0,
                "recent_sample_count": len(times),
            })

        return {
            "uptime_seconds": round(time.time() - _start_time, 2),
            "total_requests": sum(_request_counts.values()),
            "total_errors": sum(_error_counts.values()),
            "endpoints": endpoints,
        }
