# Week 2 Results — FastAPI Inference Backend

## Completed functionality

- Lifespan-managed FastAPI application factory and OpenAPI documentation
- Shared, thread-safe PatchCore inference service loaded from the Week 1 checkpoint
- Liveness and model-readiness endpoints
- Validated PNG/JPEG inspection upload with server-controlled temporary filenames
- Persistent result lookup and heatmap delivery by UUID
- Structured API errors and request IDs
- Configurable CORS, upload limits, output location, checkpoint, host, port, and threshold
- PowerShell CLI entry point, environment example, Dockerfile, and `.dockerignore`

## Validation results

- Automated tests: 18 passed
- Ruff: passed
- strict MyPy: passed
- Real checkpoint readiness: true
- Real integration image: `sheet_metal/test_public/good/000_regular.png`
- HTTP upload result: 201 Created
- Decision: PASS (model threshold, explicitly marked provisional)
- Anomaly score: 0.0
- Measured CPU model inference: 847.67 ms
- Result retrieval: 200 OK
- Heatmap retrieval: 200 OK

The integration time is a single local measurement and should not be treated as a throughput
benchmark. Startup model loading is excluded from the reported inference time.

## Known limitations

- Week 1 image AUROC was 0.6824 and a tested defective image was missed.
- The validation split contains only normal samples, so model threshold decisions are provisional.
- CPU inference is serialized with a lock to protect the shared model.
- Results use local filesystem persistence; there is no database or retention policy yet.
- Authentication, authorization, rate limiting, and production observability are not implemented.

## Recommended Week 3 work

Build the inspection database and dashboard integration, add retention/background-job policies,
calibrate the decision threshold with representative defects, and benchmark a 10% coreset on a
GPU before setting any manufacturing acceptance target.
