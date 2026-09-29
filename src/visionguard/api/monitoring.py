"""Per-application Prometheus metrics without global registry collisions."""

from typing import cast

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram, generate_latest


class ApiMetrics:
    def __init__(self) -> None:
        self.registry = CollectorRegistry()
        self.requests = Counter(
            "visionguard_http_requests_total",
            "HTTP requests",
            ["method", "path", "status"],
            registry=self.registry,
        )
        self.latency = Histogram(
            "visionguard_http_request_duration_seconds",
            "HTTP request duration",
            ["method", "path"],
            registry=self.registry,
        )
        self.model_ready = Gauge(
            "visionguard_model_ready",
            "Whether the inference model is ready",
            registry=self.registry,
        )

    def render(self) -> bytes:
        return cast(bytes, generate_latest(self.registry))
