# Week 4 Results — Production Readiness and Final MVP

## Completed deliverables

- API-key authentication with constant-time key comparison
- Server-side dashboard proxy for protected API access
- Per-client rate limiting and security headers
- Prometheus request, latency, and model-readiness metrics
- Alembic migration configuration and initial inspection-table revision
- Portable database metadata backup and restore
- API and dashboard production container definitions
- Secret-driven PostgreSQL/API/dashboard Compose stack
- Automated GitHub Actions checks for Python and TypeScript
- Repeatable live API smoke/latency script

## Acceptance status

The four-week engineering MVP is complete: model training and evaluation, reusable inference,
HTTP API, persistent history, analytics dashboard, tests, and production-readiness assets are all
present. This is an application-completeness statement, not a claim that the model is accurate
enough to control a manufacturing line.

## Validation evidence

- Python tests: 22 passed in the final release sweep
- Ruff: passed
- strict MyPy: passed
- Dashboard TypeScript: passed
- Dashboard optimized production build: passed
- npm audit at high severity: 0 vulnerabilities
- Alembic upgrade/downgrade/re-upgrade: passed on SQLite
- PostgreSQL migration SQL generation: passed with the Psycopg/PostgreSQL dialect
- Backup/restore: passed with one real inspection record
- Unauthenticated inspection request: HTTP 401
- Health and Prometheus metrics: HTTP 200
- Security header check: `X-Content-Type-Options: nosniff`
- Authenticated real-model smoke test: 3/3 requests passed
- Authenticated CPU latency: 1115.83 ms mean, 1188.34 ms p95/maximum
- Dashboard page and server-side history proxy: HTTP 200; seven persisted records observed

Release version: `1.0.0`.

## Known limitations

- The model's Week 1 image AUROC is 0.6824 and it missed a tested defective image.
- No anomalous validation set exists, so PASS/FAIL decisions remain provisional.
- PostgreSQL, Docker, and an external cloud deployment were not available locally.
- The in-memory rate limiter is appropriate for the single-process MVP. A horizontally scaled
  deployment should replace it with a shared Redis or API-gateway limiter.
- API-key authentication is suitable for an internal MVP, not multi-user identity management.
- JSON backup covers inspection metadata; the outputs volume requires a separate backup.

## Final demonstration checklist

1. Start the API and confirm `/health`, `/ready`, and `/metrics`.
2. Start the dashboard and open `http://localhost:3000`.
3. Upload a known-good image and show its stored record and heatmap.
4. Upload a defective image and explain the provisional threshold limitation honestly.
5. Filter inspection history and review analytics cards.
6. Show the passing automated checks and GitHub repository history.
7. Explain how PostgreSQL, secrets, migrations, backups, and containers are configured.

## Recommended post-MVP work

Collect representative anomalous validation data, calibrate the operating threshold, benchmark a
10% PatchCore coreset on a GPU, validate PostgreSQL/Compose in staging, add user identity and
roles, use shared rate limiting, configure alerts/backups, and conduct formal user acceptance and
manufacturing safety reviews.
