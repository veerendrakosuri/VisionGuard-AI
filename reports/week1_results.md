# Week 1 Results — Sheet Metal Baseline

## Completion status

The Week 1 pipeline was trained and validated end to end on the MVTec AD 2 `sheet_metal`
category. Dataset images, model weights, logs, and generated heatmaps remain local and are
excluded from Git because they are large or machine-specific.

## Experiment

- Date: 2026-09-28
- Runtime: Python 3.12.14, PyTorch 2.14.0 CPU, Anomalib 2.6.2
- Model: PatchCore with Wide ResNet-50-2 features from `layer2` and `layer3`
- Coreset: 1% CPU-friendly MVP setting
- Training images: 137 normal images
- Validation images: 19 normal images
- Public test images evaluated by Anomalib: 90
- Checkpoint: `artifacts/Patchcore/MVTecAD2/sheet_metal/v1/weights/lightning/model.ckpt`

## Public test metrics

| Metric | Result |
| --- | ---: |
| Image AUROC | 0.6824 |
| Image F1 | 0.5000 |
| Pixel AUROC | 0.8219 |
| Pixel F1 | 0.2863 |

These are first-baseline results, not production acceptance targets. The 1% coreset was chosen
to make a complete CPU run practical. Accuracy work should compare a 10% coreset on a GPU and
calibrate the PASS/FAIL threshold using representative anomalous validation images. The supplied
validation split contains only normal samples, so its learned threshold is provisional.

## Verified outputs

- Reusable model checkpoint and file logging under `artifacts/`
- Evaluation metrics JSON under `artifacts/evaluation_metrics.json`
- Known-good and defective sample JSON plus heatmap overlays under `outputs/week1/`
- Unit tests, Ruff linting, and strict MyPy checks

## Week 2 handoff

Use the inference command as the integration boundary for the API. Before treating the decision
as a quality-control gate, collect or reserve anomalous validation examples, select an operating
threshold from the desired false-reject/false-accept tradeoff, and record that threshold in the
experiment configuration.
