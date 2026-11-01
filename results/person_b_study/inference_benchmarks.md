# Measured image-to-CSV runtime

Each model processed the same 180 unlabelled test images once, sequentially on the RTX 5060 Ti. Image files were read once immediately before each pass to warm the filesystem cache. No GPU warm-up was excluded. No probability archives were saved during these benchmark passes.

The timer includes image read/decode, preprocessing, transfer, model prediction, connected components, RLE encoding and exact round-trip checking, and CSV output/closing. Model loading/setup is excluded; process wall time is stored separately. This is the checked export pipeline, not model-only latency.

| Model | Seed | Validation PQ | Export seconds | ms/image | Predicted instances |
|---|---:|---:|---:|---:|---:|
| Original baseline | 42 | 0.32013 | 114.35 | 635.27 | 2255 |
| No added transforms (selection seed) | 42 | 0.33808 | 106.00 | 588.88 | 2075 |
| 1024 whole images (selection seed) | 42 | 0.33745 | 89.64 | 497.99 | 1899 |
| Representative no-added-transform model | 45 | 0.34481 | 96.54 | 536.31 | 1831 |

These are single-pass descriptive timings on this computer. Different predictions require different component/RLE work, so they do not isolate neural-network speed. Validation PQ uses labelled validation images; no test accuracy is inferred. The near-tied 1024 alternative remains a retained option, and these timing measurements do not change the previously frozen selection.

The representative canonical export additionally saved 180 probability/instance archives and took 174.12 seconds. Its CSV is byte-identical to the no-archive benchmark CSV. It contains 1,831 instances; three images have no detections and therefore no fabricated CSV object rows. All 180 images are accounted for in the sidecar and prediction archives. Kaggle acceptance and score remain unverified by the user's no-upload instruction.

All numerical records and their source sidecars are retained in `inference_benchmarks.json` and `benchmark_records/`.
