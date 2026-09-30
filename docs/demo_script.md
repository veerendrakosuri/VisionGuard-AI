# VisionGuard AI Demo Script

Target duration: 5 to 7 minutes.

## 1. Problem and scope

Explain that manual inspection is repetitive and inconsistent. VisionGuard learns normal sheet
metal appearance with PatchCore and returns an image-level anomaly score plus a localized heatmap.
State that this is an engineering MVP and not a certified manufacturing control system.

## 2. Start the services

API terminal:

```powershell
cd C:\Users\veere\Documents\Codex\2026-09-28\referenced-chatgpt-conversation-this-is-an
.\.venv\Scripts\Activate.ps1
visionguard api --config configs\patchcore_mvtecad2.yaml
```

Dashboard terminal:

```powershell
cd C:\Users\veere\Documents\Codex\2026-09-28\referenced-chatgpt-conversation-this-is-an\dashboard
$env:VISIONGUARD_API_URL = "http://127.0.0.1:8000"
npm run dev
```

Open `http://localhost:3000` and `http://127.0.0.1:8000/docs`.

## 3. Show service readiness

Open `/health` and `/ready`. Explain that the checkpoint loads once during API startup and the
same model instance serves later requests.

## 4. Run two inspections

Upload `data/mvtecad2/sheet_metal/test_public/good/000_regular.png`. Show the score, PASS result,
stored history record, and heatmap.

Upload `data/mvtecad2/sheet_metal/test_public/bad/000_regular.png`. With the configured provisional
threshold, the measured score `0.04933097958564758` returns FAIL. Open its heatmap.

## 5. Explain the architecture

Describe the path from browser upload to the Next.js server proxy, FastAPI validation, shared
PatchCore inference service, heatmap output, SQLAlchemy persistence, and dashboard analytics.
Mention API-key authentication, request IDs, rate limiting, Prometheus metrics, Alembic migrations,
Docker assets, and CI checks.

## 6. Present measured results honestly

- Public test image AUROC: 0.6824
- Public test pixel AUROC: 0.8219
- Final automated checks: 26 Python tests passed
- Earlier three-request API smoke mean: 1115.83 ms on CPU
- Calibrated threshold: 0.04933097958564758
- Calibration sensitivity/specificity: 46.7% / 78.9%
- Exposure/shift stress sensitivity/specificity: 73.3% / 40.0%

Explain that the threshold catches the demonstrated defect but has poor robustness. A 10% coreset,
GPU benchmarking, independent anomalous validation data, and manufacturing acceptance criteria are
the next model-development steps.

## 7. Close

Show the GitHub history and final reports. Summarize the deliverable as a complete, tested full-stack
computer-vision MVP with measured model limitations.
