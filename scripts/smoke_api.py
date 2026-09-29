"""Small repeatable API smoke/latency check for release validation."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from pathlib import Path

import httpx


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--api-key")
    parser.add_argument("--requests", type=int, default=3)
    args = parser.parse_args()
    headers = {"X-API-Key": args.api_key} if args.api_key else {}
    latencies: list[float] = []
    with httpx.Client(base_url=args.url, headers=headers, timeout=120) as client:
        health = client.get("/health")
        ready = client.get("/ready")
        health.raise_for_status()
        ready.raise_for_status()
        if not ready.json()["ready"]:
            raise SystemExit(f"Model is not ready: {ready.json()}")
        for _ in range(args.requests):
            started = time.perf_counter()
            with args.image.open("rb") as image:
                response = client.post(
                    "/api/v1/inspections",
                    files={"image": (args.image.name, image, "image/png")},
                )
            response.raise_for_status()
            latencies.append((time.perf_counter() - started) * 1000)
    print(
        json.dumps(
            {
                "requests": len(latencies),
                "mean_ms": round(statistics.mean(latencies), 2),
                "p95_ms": round(
                    sorted(latencies)[max(0, math.ceil(len(latencies) * 0.95) - 1)], 2
                ),
                "minimum_ms": round(min(latencies), 2),
                "maximum_ms": round(max(latencies), 2),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
