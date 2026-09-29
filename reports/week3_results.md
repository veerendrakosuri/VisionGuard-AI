# Week 3 Results — Persistence and Operations Dashboard

## Completed functionality

- SQLAlchemy inspection repository supporting PostgreSQL and SQLite
- Automatic schema creation and configurable 30-day startup retention cleanup
- Durable inspection metadata in addition to filesystem heatmaps
- Paginated and PASS/FAIL-filtered history API
- Aggregate totals, failure rate, anomaly-score, and latency API
- Responsive Next.js dashboard for image upload, KPIs, history, and heatmap access
- PostgreSQL/API development Compose configuration
- Database and dashboard environment examples

## Validation

- Python tests: 20 passed
- Python coverage: 75% overall; persistence module: 100%
- Ruff: passed
- strict MyPy: passed
- Dashboard TypeScript: passed
- Dashboard production build: passed
- Dashboard production server: HTTP 200 with rendered VisionGuard page
- npm high-severity audit: 0 vulnerabilities
- Local database: SQLite exercised successfully
- Real checkpoint upload: HTTP 201, PASS, score 0.0, 1578.97 ms CPU inference
- Persisted history query: HTTP 200 with one stored inspection
- Analytics query: HTTP 200 with matching totals
- Persisted heatmap retrieval: HTTP 200

## Environment limitation

PostgreSQL and Docker are not installed on this machine. PostgreSQL uses the supported SQLAlchemy
2.x and Psycopg 3 path and has Compose configuration, but it has not been executed locally. This
is reported explicitly rather than presenting SQLite validation as PostgreSQL validation.

## Model limitation carried forward

Decisions remain provisional because Week 1 validation contained no anomalous samples and the
baseline missed a tested defect. The dashboard displays this warning and must not be interpreted
as a production quality gate.

## Week 4 handoff

Deploy the API, PostgreSQL, and dashboard; add authentication and authorization, production
secrets, rate limits, migrations, backup/restore, monitoring, and calibrated acceptance criteria.
Run a representative performance and user-acceptance test before the final demonstration.
