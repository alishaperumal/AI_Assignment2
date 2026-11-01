# Person C handoff

Person B's training study is complete. The selected method is **U-Net / ResNet34 with no added augmentation transforms**, and the representative is **seed 45, checkpoint from epoch 17** of its completed 20-epoch run. Person C can use its probabilities to develop instance separation and post-processing without retraining the network, then investigate compression separately.

**Model reloads and final exports are verified.** All six reloaded models, comprising three full-budget selection candidates and three fresh seeds, reproduced their recorded validation metrics. The representative has 122 validation prediction archives and 180 test prediction archives. The transfer package's `handoff.json` records its deep artifact checks, and `manifest.json` records its file hashes. Consult `results/person_b_completion_verification.json` in the original repository for the independent final package check; this study document is frozen before package creation and does not itself certify that later check.

The fixed design is explained in [methodology.md](methodology.md), with the selection rules in [protocol.json](protocol.json). The chosen configuration is documented in [configuration_selection.json](configuration_selection.json), and the exact representative is identified in [selection.json](selection.json).

## What was selected

| Setting | Selected value |
|---|---|
| Architecture | U-Net with a ResNet34 encoder; one grayscale input and one foreground output |
| Training loss | Binary cross entropy + Dice, equal weights |
| Training input | Random 512 × 512 crops at original image scale; 70% foreground-directed crop attempts |
| Training target | Separate annotation sets, rather than a voted consensus |
| Added augmentation | Disabled; random crop sampling still operates |
| Preprocessing | No disc mask, CLAHE, or local normalization |
| Normalization | Mean `0.32688403099775315`, standard deviation `0.22290047382928638`, after loading pixels into `[0,1]` |
| Training | AdamW, learning rate `0.0003`, weight decay `0.0001`, cosine schedule, mixed precision |
| Full budget | 20 epochs × 960 examples; physical batch 4; no gradient accumulation or early stopping |
| Inference | Whole 2048 × 2048 image; mixed precision on CUDA; no resizing, tiling, or test-time augmentation |
| Baseline instance conversion | Probability strictly above `0.5`, then 8-connected components |

The stored augmentation probabilities remain visible in the full configuration, but `data.augment: false` disables those transformations. Do not interpret the method name as disabling random crops or as evidence that every possible augmentation is harmful.

The three fresh runs used the frozen configuration, independently of selection seed 42:

| Fresh seed | Best validation PQ | Best epoch | Completed epochs |
|---|---:|---:|---:|
| 43 | 0.3363588843 | 18 | 20 |
| 44 | 0.3296855785 | 20 | 20 |
| 45 | 0.3448101728 | 17 | 20 |

The mean is **0.3369515452**, with **sample standard deviation 0.0075796948** across these three runs. Use that mean and variation when describing the selected method. Seed 45 supplies the exported representative because it has the highest validation PQ, as specified before the fresh runs; its individual score is not the three-run mean. The underlying checks and hashes are in [fresh_seed_verification.json](fresh_seed_verification.json).

The representative weights hash recorded after training is:

```text
617f931eecf9f8b5956748ef9c5c912bfc32d9c87067b524c42d687f35c81377
```

### Verified final exports

[Reload verification](reevaluation_verification.json) covers six models: the three full-budget selection candidates and the three fresh seeds. Their reloaded validation metrics agree with the recorded values to the stated absolute tolerance of `1e-12`. The representative evaluation covers **122 observations / 200 annotation sets** and reproduces **PQ 0.3448101728**.

The canonical representative CSV contains **1,831 predicted instances** across **180 processed test images**. Three images have no detections: `20180507140210Th`, `20200310113230Ch`, and `20211214174010Mh`. They contribute no artificial CSV rows and are explicitly recorded in the canonical sidecar. Exported masks passed the encoder's exact RLE round-trip checks. CSV SHA256:

```text
4a7fbc136f7a8f9aad0a9bbffa328be3b90609d99aa4a73fa8f1f4167be38120
```

The separately timed image-to-CSV run produced a byte-identical CSV in **96.5350 seconds**, or **536.3057 milliseconds per image** over 180 images. The canonical run that also saved NPZ prediction archives took **174.1198 seconds**. These measure different output work and must remain labelled separately. The benchmark includes image reading/preprocessing, inference, components, RLE checking, and CSV output; model loading is outside its timer. It is one pass on this computer, not a repeated runtime estimate. See [the timing results](inference_benchmarks.json) and [timing protocol](inference_timing_protocol.json).

The independent [data-preservation check](data_preservation_verification.json) confirms all 888 raw files and the frozen split are unchanged. It also verifies the epoch-2 baseline archive and the original first two history entries retained in the completed 20-epoch baseline.

### Retained 1024 alternative

The full-budget selection-seed comparison was very close: `no_augmentation` scored **0.3380807049**, and `resolution_1024` scored **0.3374512925**. The difference, about **0.0006294 PQ**, follows the registered numerical selection rule but does not establish statistical superiority. Only one selection seed was used for that comparison.

The alternative remains at `runs/full/resolution_1024_seed42` in the original repository. It uses augmented whole images reduced to 1024 for training and inference, with probabilities restored to original resolution. It changes field of view, pixel scale, sampling, and compute together. Its completed image-to-CSV benchmark took **89.6377 seconds / 497.9875 ms per image**, compared with **105.9981 seconds / 588.8784 ms per image** for the selection-seed `no_augmentation` model. Both processed 180 images without saving probability archives; these are descriptive single-pass timings. Retain its checkpoint/config and these measurements as useful evidence for Person C; do not silently substitute it for the frozen selected method. The representative package does not automatically contain this alternative's weights. Any later use should be a separately recorded Person C experiment.

## Package and interfaces

The transfer directory is `runs/person_b_final`. The packager defines and validates the following contents. Its verification records travel with the files; the independent final completion record remains outside the immutable package in the original repository.

| Package path | Use |
|---|---|
| `best_weights.pt` | Representative inference checkpoint; no optimizer state |
| `config.original.yaml` | Untouched resolved training configuration |
| `config.yaml` | Same numerical settings, with portable filesystem paths |
| `splits/split_v1.json` | Frozen observation split |
| `evaluation/predictions/` | 122 validation prediction archives |
| `test_predictions/` | 180 test prediction archives |
| `evaluation/summary.json`, `evaluation/config.yaml` | Validation score, settings, and checkpoint provenance |
| `submission.csv`, `submission.summary.json` | Canonical test CSV, coverage, settings, and checkpoint provenance |
| `summary.json`, `history.csv`, `run_metadata.json` | Training evidence and environment/source information |
| `study/` | Stable protocol, selection, results, methodology, and handoff documents |
| `manifest.json`, `handoff.json` | File hashes and package verification facts |

Each `{observation}.npz` contains:

- `probability`: float32 `(2048, 2048)`, values in `[0,1]`, before thresholding.
- `instances`: uint32 `(2048, 2048)`, with zero for background and positive IDs for predicted filaments.
- `observation`: a scalar string matching the filename.

Read archives with `numpy.load(path, allow_pickle=False)`. Keep their original names and use the observation ID to align images, annotations, and predictions. An empty instance map is a valid possible output, not a reason to invent an object.

The source snapshot provides these interfaces:

- `models.load_checkpoint_model(path, device)` returns the restored model, its saved config, and checkpoint dictionary; it does not redownload pretrained weights.
- `inference.predict_probability(model, normalized_images, cfg['inference'])` accepts a floating tensor `[B,1,H,W]` and returns probabilities at the original input size. Normalize once, before this call.
- `inference.probability_to_instances(probability, threshold)` is the current simple conversion to replace in a separate Person C pipeline.
- `predict.instance_rows` and `predict.write_submission` provide the existing COCO RLE exporter. Zero-detection images contribute no fake CSV rows and must remain recorded in coverage metadata.

Reuse the evaluator's handling of original overlapping annotation layers. Scoring only the cached integer ground-truth maps would lose that correction. Keep pooled `PQ_dataset` as the main comparison score; pixel Dice alone does not adequately penalize merges and fragmentation.

## Reproduce in a separate working copy

Preserve the transferred Person B package as an immutable reference. Make a separate working copy for reproduction or Person C experiments, and run the following commands from that copy's root. `python` should refer to an environment matching `requirements.txt` and `run_metadata.json`, with a compatible PyTorch build. CUDA uses the GPU; use `--device cpu` if necessary, while recording that changed execution setting.

Point to the supplied Kaggle dataset and rebuild only the local cache. Do not regenerate the split:

```powershell
$datasetRoot = 'C:\path\to\MAGFiLO_1.0_Kaggle_2026'
python data.py build-cache --data-root "$datasetRoot" --cache-dir cache --workers 2
```

Reproduce validation into a new directory:

```powershell
python evaluate.py --checkpoint best_weights.pt --config config.yaml --data-root "$datasetRoot" --device cuda --output runs/person_c/reproduced_evaluation --save-predictions
```

Reproduce the test CSV and optional probability archives:

```powershell
python predict.py --checkpoint best_weights.pt --config config.yaml --data-root "$datasetRoot" --device cuda --output runs/person_c/reproduced_submission.csv --save-probabilities runs/person_c/reproduced_test_predictions
```

`--config config.yaml` is important: the checkpoint and original provenance retain the training computer's absolute paths. The portable config retains every numerical/model choice while relocating data/cache/split/output paths. These commands leave the canonical exports unchanged. GPU/CPU or software changes may prevent exact numerical reproduction, so retain the environment and compare scores as well as hashes where appropriate.

## Data, evaluation, and experiment boundaries

The split contains 584 training observations, 122 validation observations, and one excluded gap observation. Its SHA256 is:

```text
58245fc1fdc3c39c7e482d703cd4c9e26ebdbcae6e08c6c328149d417284eec3
```

Use only the supplied Kaggle dataset. Keep it read-only and check transferred data against [raw_dataset_manifest.json](raw_dataset_manifest.json). Do not obtain external annotations for test images, turn metadata such as spines/categories into extra supervision, or move validation observations into training. The 180 test images have no supplied labels; their visual panels do not provide measured accuracy.

Keep each Person C change in its own configuration and output directory. Tune thresholds, morphological cleanup, separation, or compression using the same local validation split. Record failed ideas as well as improvements. Compare fragmentation/merging alongside PQ, TP, FP, and FN: lower relation counts can simply mean more missed objects.

Person B supplies a trained foreground model, a simple instance baseline, and repeated-run evidence. Person C's remaining scope includes instance separation, post-processing, and a measured compression study. Mixed-precision inference and omission of optimizer state are already part of this handoff; they are not evidence of a completed compression experiment. Compare compressed and original models on quality, deployable size, and consistently measured end-to-end latency. The agreed timing boundaries are in [inference_timing_protocol.json](inference_timing_protocol.json); prediction-archive export adds storage overhead and must not be confused with the separate image-to-CSV benchmark.

## Transfer and human deliverables

`runs/`, model weights, cached data, and the virtual environment are ignored by Git. Cloning the repository alone will not transfer the model or prediction archives. Transfer the complete verified `person_b_final` directory separately, preserving its relative layout, and check the manifest after transfer. The raw dataset and resumable optimizer checkpoints are intentionally excluded from that package. Retained alternative weights need a separate explicit transfer if Person C will use them.

The large Person B handoff is an internal research artifact, not the assignment's final submission ZIP. Person B's slide content, results discussion, speaker notes and references are prepared in the original repository's `delivery/` directory. The team must combine the A/B/C contributions, add presenter names and final conclusions, and record the video. The assignment requires all members to appear, roles to be explained, the video to be at most 10 minutes and 50 MB, and the final ZIP to contain code, the submission CSV, video, and PowerPoint slides.

No Kaggle or ClickUP upload is performed in this local study. A human must complete those actions when authorized, retain the exact CSV actually uploaded, record the real Kaggle score/acceptance, and assemble/check the required submission archive. Do not describe a locally validated CSV as already accepted by Kaggle, or this Person B handoff as completion of the entire assignment.
