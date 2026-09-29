# VisionGuard AI

Production-oriented Week 1 foundation for industrial visual anomaly detection. It uses PatchCore
to learn normal product appearance, then produces anomaly scores and localization maps for unseen
images. The initial configuration targets one MVTec AD 2 category so the four-week MVP remains
achievable.

## Week 1 scope

- Typed, validated YAML configuration
- MVTec AD 2 data integration through Anomalib
- PatchCore training, evaluation, and single-image inference commands
- Structured console/file logging and isolated artifact directories
- Unit tests, linting, type checking, and reproducible dependency bounds
- Git-safe handling of datasets, checkpoints, generated outputs, and secrets

## Prerequisites

- Python 3.10, 3.11, or 3.12
- Git
- NVIDIA GPU with a compatible PyTorch build is recommended, but CPU works for smoke tests

## Setup (PowerShell)

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

PyTorch installation varies by CPU/CUDA platform. If Anomalib's dependency resolution does not
select the desired build, install PyTorch first using the command generated at
https://pytorch.org/get-started/locally/ and then run the editable install.

## Dataset

Download **MVTec AD 2** from its official source and accept its license. Extract it beneath
`data/mvtecad2/`, then confirm that `data.category` in
`configs/patchcore_mvtecad2.yaml` matches an extracted category. The default is `sheet_metal`.
Dataset content is never committed.

## Commands

```powershell
# Validate the lightweight project foundation
python -m pytest
python -m ruff check .
python -m mypy src

# Build the PatchCore memory bank and checkpoint
visionguard train --config configs/patchcore_mvtecad2.yaml

# Evaluate the newest checkpoint and save artifacts/evaluation_metrics.json
visionguard evaluate --config configs/patchcore_mvtecad2.yaml

# Inspect one image (writes result.json and heatmap_overlay.png)
visionguard predict --config configs/patchcore_mvtecad2.yaml `
  --image path\to\sample.png --output outputs\sample
```

The thin scripts in `scripts/` provide equivalent entry points. Prefer the installed
`visionguard` command for normal use.

## Repository layout

```text
configs/                 Experiment configuration
data/                    Local datasets (ignored)
outputs/                 User-facing predictions (ignored)
scripts/                 Convenience entry points
src/visionguard/         Application package
tests/                   Fast unit tests
artifacts/                Checkpoints, metrics, and logs (ignored)
```

## Reproducibility notes

The YAML file is the source of truth for an experiment. Keep the exact config and Git commit hash
with reported results. PatchCore is primarily a feature-memory method rather than a conventional
gradient-trained network; “training” builds its representative memory bank from normal images.
The checked-in configuration uses a 1% coreset so Week 1 can be completed on CPU. A benchmark run
on a suitable GPU should restore `coreset_sampling_ratio: 0.1` for the standard PatchCore baseline.

## Definition of done for Week 1

1. A single MVTec AD 2 category is available locally.
2. `visionguard train` produces a checkpoint under `artifacts/`.
3. `visionguard evaluate` prints image- and pixel-level metrics supplied by Anomalib.
4. `visionguard predict` writes a JSON result for a known good and defective image.
5. Tests and lint checks pass before committing.

The completed CPU baseline and its measured metrics are recorded in
[`reports/week1_results.md`](reports/week1_results.md).

## Week 2 — Inference API

The FastAPI service loads the saved PatchCore model once during application startup and exposes
it through versioned inspection endpoints. Uploaded files are size-, type-, filename-, and
content-validated; server-generated inspection IDs prevent clients from controlling storage
paths. Runtime uploads, results, and heatmaps remain under the ignored `outputs/` directory.

```text
Client -> upload validation -> shared PatchCore service -> result JSON + heatmap
```

Start the service:

```powershell
visionguard api --config configs/patchcore_mvtecad2.yaml
```

Swagger UI is available at `http://127.0.0.1:8000/docs`. Key endpoints are:

- `GET /health` — process liveness
- `GET /ready` — checkpoint/model readiness
- `POST /api/v1/inspections` — submit a PNG or JPEG
- `GET /api/v1/inspections/{inspection_id}` — retrieve a result
- `GET /api/v1/inspections/{inspection_id}/heatmap` — retrieve the overlay

PowerShell upload example:

```powershell
$response = Invoke-RestMethod -Method Post `
  -Uri "http://127.0.0.1:8000/api/v1/inspections" `
  -Form @{ image = Get-Item "path\to\sample.png" }
$response
```

Equivalent curl example:

```powershell
curl.exe -F "image=@path\to\sample.png" http://127.0.0.1:8000/api/v1/inspections
```

The response contains the inspection ID, score, PASS/FAIL decision, inference time, model
metadata, threshold source, and heatmap URL. With `decision.threshold: null`, the API labels the
source `model_provisional`; this is deliberately not a production acceptance threshold.

Configuration lives under the `api` and `decision` sections of the YAML file. Common deployment
values can be overridden with the variables shown in `.env.example`. A real `.env` file remains
ignored.

### Docker

Build with `docker build -t visionguard-api .`. The checkpoint is excluded from the image; mount
the local artifacts directory and identify the checkpoint at runtime:

```powershell
docker run --rm -p 8000:8000 `
  -v "${PWD}/artifacts:/app/artifacts:ro" `
  -e VISIONGUARD_CHECKPOINT=/app/artifacts/Patchcore/MVTecAD2/sheet_metal/v1/weights/lightning/model.ckpt `
  visionguard-api
```

If `/ready` is false, verify that the configured checkpoint exists and is readable. A 413 response
means the upload exceeds `api.max_upload_bytes`; 415 means its extension or MIME type is not
allowed. See [`reports/week2_results.md`](reports/week2_results.md) for measured validation.

## Week 3 — Inspection history and dashboard

Week 3 adds SQL-backed inspection history, aggregated quality metrics, retention cleanup, and a
Next.js operations dashboard. SQLAlchemy uses SQLite locally and the same repository supports
PostgreSQL through `VISIONGUARD_DATABASE_URL`.

New API queries:

- `GET /api/v1/inspections?limit=20&offset=0&decision=FAIL`
- `GET /api/v1/inspections/analytics/summary`

The history endpoint supports pagination and optional PASS/FAIL filtering. Summary analytics
include totals, pass/fail counts, failure rate, average anomaly score, and average inference time.
Records older than `database.retention_days` are removed when the service starts.

Run the API and dashboard in separate PowerShell windows:

```powershell
visionguard api --config configs/patchcore_mvtecad2.yaml

cd dashboard
npm install
npm run dev
```

Open `http://localhost:3000`. The dashboard supports new uploads, live KPI cards, recent
inspection history, and heatmap links. Set `NEXT_PUBLIC_API_URL` when the API is not on
`http://127.0.0.1:8000`.

For PostgreSQL, set a deployment secret rather than committing credentials:

```powershell
$env:VISIONGUARD_DATABASE_URL = "postgresql+psycopg://visionguard:password@localhost:5432/visionguard"
```

`compose.yaml` provides a development PostgreSQL and API stack when Docker is available. Its
placeholder password must be changed outside local development. PostgreSQL and Docker were not
installed on the Week 3 development machine, so compatibility is implemented but only SQLite was
executed locally. See [`reports/week3_results.md`](reports/week3_results.md).
