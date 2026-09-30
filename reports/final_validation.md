# Final Validation — VisionGuard AI

Validation date: 2026-09-29

## Completed functionality

- PatchCore training, evaluation, checkpoint reuse, scoring, and heatmap generation
- FastAPI upload validation, inference, retrieval, readiness, metrics, and structured errors
- SQLAlchemy inspection persistence, pagination, retention, and analytics
- Next.js upload dashboard, history, KPI summaries, and server-side API proxy
- API-key authentication, rate limiting, request IDs, security headers, Alembic, backup/restore,
  Docker assets, Compose, and CI
- Reproducible threshold calibration and fixed-threshold evaluation commands
- Submission demo script, résumé wording, dashboard evidence, and editable presentation

## Threshold calibration

The final configuration uses `0.04933097958564758`. Maximum balanced accuracy selected this value
from 19 normal validation images and 15 regular-view public anomalous images.

| Measure | Result |
| --- | ---: |
| Balanced accuracy | 0.6281 |
| Sensitivity | 0.4667 |
| Specificity | 0.7895 |
| Precision | 0.6364 |
| F1 | 0.5385 |
| Confusion counts | TP 7, FP 4, TN 15, FN 8 |

## Excluded-variant stress evaluation

The selected threshold was frozen, then evaluated on the 20 normal and 75 anomalous exposure/shift
variants excluded from calibration.

| Measure | Result |
| --- | ---: |
| Balanced accuracy | 0.5667 |
| Sensitivity | 0.7333 |
| Specificity | 0.4000 |
| Precision | 0.8209 |
| F1 | 0.7746 |
| Confusion counts | TP 55, FP 12, TN 8, FN 20 |

These variants share source parts with the regular views, so this is a robustness check rather than
an independent test of generalization. The low specificity shows that exposure and position changes
cause many false alarms.

## Final live application evidence

- `GET /health`: HTTP 200, `{"status":"ok"}`
- `GET /ready`: HTTP 200, model ready
- Real defective upload: HTTP 201
- Defective sample score: `0.04933097958564758`
- Configured threshold: `0.04933097958564758`
- Defective sample decision after calibration: FAIL
- Measured CPU inference time for the final browser run: 2103 ms
- Dashboard refreshed to show the new FAIL record without browser console errors

## Quality validation

- Python tests: 26 passed
- Python coverage: 72%
- Ruff: passed
- strict MyPy: passed
- Next.js optimized build: passed after the form fix
- Docker/PostgreSQL runtime: not executed because neither service is installed locally
- External cloud deployment: not executed because no provider or deployment account was supplied

## Model limitations

- Public test image AUROC remains 0.6824 for the 1% CPU-friendly model.
- The calibrated regular-view sensitivity is 46.7%, so more than half of those defects remain missed.
- The excluded-variant specificity is 40.0%, producing a high false-alarm rate.
- The final threshold uses labeled development data and requires independent manufacturing
  validation before operational use.

## Recommended next experiment

Train or evaluate a 10% PatchCore coreset on GPU, reserve independent normal and defective images by
physical part rather than image variant, select a threshold against documented false-reject and
false-accept costs, then validate Docker Compose with PostgreSQL in a staging environment.
