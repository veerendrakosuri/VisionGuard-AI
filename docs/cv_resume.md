# VisionGuard AI Résumé and Interview Notes

## One-line project description

Built a full-stack industrial anomaly-detection MVP using PatchCore, PyTorch, Anomalib, FastAPI,
SQLAlchemy, PostgreSQL-compatible persistence, and Next.js.

## Résumé bullets

- Developed an end-to-end sheet-metal inspection system that returns anomaly scores, PASS/FAIL
  decisions, localized heatmaps, persistent inspection history, and dashboard analytics.
- Designed a typed FastAPI service that loads one PatchCore checkpoint at startup, validates image
  uploads, serializes inference results, and exposes readiness, history, heatmap, analytics, and
  Prometheus endpoints.
- Added API-key authentication, request tracing, rate limiting, Alembic migrations, database
  backup/restore, Docker deployment assets, and GitHub Actions validation.
- Evaluated the 1% PatchCore baseline on MVTec AD 2 sheet metal data, recording 0.6824 image AUROC
  and 0.8219 pixel AUROC, then added reproducible threshold calibration and stress evaluation.
- Verified 26 Python tests plus Ruff, strict MyPy, TypeScript, and optimized Next.js builds while
  documenting accuracy and deployment limitations.

## Interview explanation

The project solves the engineering problem around an anomaly model, not only the model itself. The
API validates and stores an image safely, reuses a loaded model under a concurrency lock, records the
result in SQL storage, and serves the heatmap to a Next.js dashboard. Configuration supports CPU or
GPU execution, SQLite for local work, and PostgreSQL for deployment.

The first model uses a 1% PatchCore coreset so the complete pipeline can run on CPU. Its metrics and
stress results show that this model is not accurate enough for a manufacturing gate. That limitation
is recorded rather than hidden. The next experiment should compare a 10% coreset on GPU and calibrate
against independent normal and defective production images.

## Claims to avoid

Do not describe the project as production-ready, highly accurate, real-time, or deployed to the
cloud. Docker and PostgreSQL compatibility were prepared, but neither Docker nor a PostgreSQL server
was available for a live local deployment test.
