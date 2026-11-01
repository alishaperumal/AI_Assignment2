# Handoff: Person A fixes, Person B pipeline, and Person C interfaces

This file describes the corrected data/evaluation interfaces, the completed Person B training study, and the boundary with Person C. Use `README.md` for measured results and commands, `COS711-A2.pdf` for assignment requirements, and the Word team plan for the division of work. The concise delivery guide is [Person C handoff](results/person_b_study/person_c_handoff.md).

For current continuation priorities and future LLM context, read [PERSON_C_START_HERE.md](PERSON_C_START_HERE.md) first. It supplements the frozen technical handoff with the distinction between the selected recipe and the exported seed's extra misses, issue severity, proposed experiments, and the saved-probability evaluation runner that C still needs to implement. [AGENTS.md](AGENTS.md) is the short assistant entry point. The original study/package remains unchanged.

## Completed work and remaining boundaries

| Area | Delivered | Remaining boundary |
|---|---|---|
| Person A data | Frozen split protection, configurable paths, validated inputs, deterministic crops/augmentations, Windows-safe workers, bounded training-only statistics; EDA outputs regenerated on the supplied data | Rerun EDA if its data or analysis choices change |
| Person A evaluation | Official-reference pooled PQ, original overlapping GT masks during evaluation, clearly separated diagnostic metrics | Preserve this scoring policy for later comparisons |
| Person B training | Original baseline completed with its original schedule; all 17 research runs complete; six full-budget checkpoints independently reloaded with matching scores | Preserve saved configurations and original run records |
| Person B experiments | Eleven six-epoch screens, two fresh 20-epoch confirmations plus the completed baseline, then three fresh 20-epoch seeds of the selected configuration | Do not mix screening scores with full-budget scores or selection seed 42 with fresh seeds 43/44/45 |
| Basic submission | Original-resolution masks, checked COCO RLE exporter, 180 test archives, CSV and provenance/coverage sidecar | Kaggle acceptance/score and ClickUP uploads are deliberately deferred |
| Person B presentation | Methodology/results content, notes and references prepared from the recorded study | Human recording, presenter names and team assembly |
| Person C | Probability/instance interfaces, checkpoints/configs, detailed validation outputs, shared inference function | Advanced separation/post-processing, compression and its quality/speed/size comparison, team conclusions and final assembly |

**Person C update (1 November 2026, later the same day).** The Person C row above has since been worked through: a configurable post-processing stage (`postprocess.py`, `postprocess` config section whose defaults reproduce the baseline), an offline scorer for saved probabilities (`offline_eval.py`, which first reproduced the seed-45 result exactly), declared post-processing searches with split-half and bootstrap checks, flip test-time augmentation (opt-in `inference.tta`), a compression study (`compress.py`, `compress_onnx.py`: FP16 weights, distillation with a label-only control, structured pruning, ONNX/INT8), the final checked CSV, the team deck builder and the ZIP/video helpers. Results and provenance: [results/person_c/person_c_results.md](results/person_c/person_c_results.md). Person B's package, study records and split were not modified. The paragraphs below describe the interfaces as handed to Person C.

Tests cover data reproducibility (including two spawned workers), split protection, metrics, models/losses, inference, submission encoding, configuration, training/checkpoint behavior, study reporting, and handoff packaging. Training completion is recorded in each `summary.json` and, for suites, `records/*.json` / `experiments.csv`. There are 17 completed research runs and 186 completed epochs across the staged budgets.

The selected configuration is **U-Net/ResNet34 with BCE + Dice and no added geometric/appearance transforms**. Random foreground-biased 512-pixel crops remain enabled. The three fresh seeds give **PQ 0.33695 ± 0.00758** (sample SD); representative seed 45 achieved **0.34481** at epoch 17. The original baseline completed 20 epochs at **0.32013**. Selection seed 42 narrowly preferred no added transforms (0.33808) over 1024 whole images (0.33745); that small single-seed difference does not establish superiority. See [the fixed protocol](results/person_b_study/protocol.json), [selection decision](results/person_b_study/configuration_selection.json), and [fresh-run verification](results/person_b_study/fresh_seed_verification.json).

The old two-epoch baseline artifacts are preserved under `runs/E0_baseline_seed42/archive_epoch2/`. `results/implementation_validation.json` describes that historical implementation check. The current baseline checkpoints and history belong to the completed 20-epoch run. The [preservation check](results/person_b_study/data_preservation_verification.json) confirms all 888 raw files, the frozen split, archived files and first two history entries are unchanged. The final Python suite passed 101 tests. [Completion verification](results/person_b_completion_verification.json) records the delivery/package/presentation checks; [the study report](results/person_b_study/study_report.md) and [plain interpretation](results/person_b_study/interpretation.md) explain the measured results.

## Data facts and corrections

- The provided training data has **707 observations (distinct images), 1,154 annotation sets, and 8,199 filament annotations**. Of the observations, 411 have one set, 145 have two, and 151 have three. Set counts alone do not establish an annotator's identity.
- Images are 2048 × 2048 grayscale. The recorded mean foreground coverage is approximately **0.36%**, so background greatly outnumbers filament pixels. This makes a foreground-aware loss and suitable sampling useful. It does **not** prove that BCE always learns an all-background prediction.
- The frozen split has **584 training, 122 validation, and 1 gap-excluded observation**. All annotation sets for one observation stay in the same partition. Validation uses 14-day time blocks with a one-day exclusion gap and an instrument mix matched approximately to the test filenames. The split file is unchanged.
- Both train and test span 2011–2022. That fact does not tell us whether local validation will be easier or harder than Kaggle's hidden data.
- `results/eda_summary.json` records annotator agreement. Treat it as a **reference for how differently people draw filaments**, not a mathematical ceiling on model performance. The regenerated EDA uses **100 sampled multi-annotated observations and 404 ordered pairs of cached instance maps**. Those maps resolve overlapping pixels to the later annotation, unlike lossless benchmark evaluation. Both directions of each pair are included, so pairs are not independent observations.
- The regenerated disc-mask check samples **60 annotation sets**, all nonempty, not the whole dataset. Its zero outside-disc foreground fraction supports that sample only. The notebook and summary record both sample counts. Disc masking and local normalization were measured in the six-epoch screen; neither was promoted to full-budget confirmation, so their screening results do not establish full-budget superiority or inferiority.
- In five annotation sets, original filament masks overlap. The integer training cache assigns overlapping pixels to the later annotation. Validation reloads original binary layers for affected sets, preserving the source masks for scoring.
- Consensus uses a foreground vote with ties included: with **two annotation sets it is a union**, and with three it requires two votes. Connected components of the resulting mask can merge nearby/touching filaments. Also, `per_set` samples an observation once per annotation set, whereas `consensus` has one item per observation. The comparison changes both target construction and observation weighting; it is not a pure test of voting alone.

Do not regenerate `splits/split_v1.json`. A differing `save_split` overwrite raises an error, and saving an identical split leaves existing bytes untouched. A future alternative split must have a new versioned filename. Cache building does not alter the split.

## Data API and reproducibility

The training program passes the full `cfg['data']` mapping to:

```python
import data as D
train_loader, val_loader = D.make_loaders(cfg['data'])
```

Supported data keys are `data_root`, `cache_dir`, `split`, `crop`, `train_resize`, `pos_frac`, `batch_size`, `num_workers`, `val_num_workers`, `target`, `preprocess`, `aug`, `augment`, `epoch_length`, `seed`, `pin_memory`, and `drop_last`. Unknown settings are rejected. Required images, masks, and split membership are checked before training starts. `configs/data_default.yaml` is a legacy data-only example; use `configs/baseline.yaml` and its extensions for the full training CLI.

Training batches are dictionaries:

| Key | Shape and type | Meaning |
|---|---|---|
| `image` | `(B, 1, H, W)`, float32 | Preprocessed and normalized grayscale input |
| `mask` | `(B, 1, H, W)`, float32 | Binary foreground training target |
| `instances` | `(B, H, W)`, int32 | Cropped instance IDs; 0 is background; positive IDs need not be consecutive |
| `name` | list of strings | Observation IDs |

For the baseline, `H = W = 512`. `pos_frac: 0.7` makes 70% of crop attempts center on a foreground pixel when one exists; augmentation can subsequently move or remove foreground. `crop: 0` uses the whole training image. Optional `train_resize: 1024` resizes the training image before cropping, using area downsampling for images and nearest-neighbor sampling for integer masks. Set matching `inference.resize` for that experiment. Ground-truth validation stays at original resolution.

Validation always has batch size one because observations have different numbers of annotation sets:

| Key | Shape and type | Meaning |
|---|---|---|
| `image` | `(1, 1, 2048, 2048)`, float32 | Whole normalized image |
| `mask` | `(1, 1, 2048, 2048)`, float32 | Consensus foreground, available for an optional validation loss |
| `gt_sets` | `(1, K, 2048, 2048)`, int32 | Cached maps for the K available annotation sets |
| `set_ids` | K one-element tuples after default collation | Original annotation-set IDs, aligned with `gt_sets` |
| `name` | one-element list of strings | Observation ID |

Use `evaluate_model` for benchmark scoring: it uses `set_ids` and the original training JSON when an integer cache map would lose overlapping ground truth. Directly looping only over cached `gt_sets` would miss that correction.

Each crop and augmentation is determined by the run seed, epoch, and sample index. Calling a sample twice in the same epoch gives the same result; a later epoch gives new crops/augmentations. This remains independent of worker scheduling. The trainer calls `train_loader.dataset.set_epoch(epoch)` and resets its shuffle generator for each epoch. Custom training loops must do the same **before creating the iterator**. Workers are not persistent, so spawned workers receive the updated epoch. `num_workers` and `val_num_workers` are configured separately and can both be zero for debugging.

If `epoch_length` differs from the number of items, the dataset uses balanced shuffled cycles per epoch. Short runs therefore do not repeatedly take only the beginning of the dataset, and oversampling does not always duplicate the same prefix. `drop_last` can still discard a partial batch; experiment budgets should be divisible by the effective batch size. The suite checks that condition.

Preprocessing (disc masking, CLAHE, local normalization) happens before augmentation; normalization happens after augmentation. Do not normalize again in the model. `compute_norm_stats` samples no more than `max_images` deterministically from the IDs supplied by its caller; supply only `split['train']`. The CLI and experiment runner do this. The runner stores separate statistics for each preprocessing recipe, keyed by its data/split/catalog/settings, so a transformed-image trial cannot silently reuse plain-image statistics.

## Models, checkpoints, and metrics

`models.build_model(cfg['model'])` returns a one-input-channel, one-output-channel U-Net or U-Net++ with raw logits. `losses.build_loss(cfg['loss'])` consumes logits and binary masks. Shared inference applies sigmoid exactly once. ImageNet encoder initialization is used by the chosen baseline; checkpoint reloads do not download it again.

`models.load_checkpoint_model(path, device)` returns `(model_in_eval_mode, saved_config, checkpoint_dictionary)`. Use trusted project checkpoints. `best_weights.pt` is the compact handoff; `last.pt` and `best.pt` also retain training states for complete-epoch resume. Keep the full config, split hash, seed, and normalization values with every model. Changing the total planned epochs changes the learning-rate schedule and is not a compatible continuation; `--max-epochs` limits one invocation while preserving that horizon.

The [organizer's Self Evaluation Notebook v6](https://www.kaggle.com/code/azimahmadzadeh/self-evaluation-notebook) was inspected through its public source API. Provenance is stored in `results/evaluation_reference.json` (source SHA256 `ae7c9445b4eee2b77b216df5c9772fdf92fda15611e98abd7ef5f02a4e380801`). No external annotations were downloaded or notebook training/evaluation code executed to obtain that reference.

| Field or output | Interpretation |
|---|---|
| `PQ_dataset`, `SQ_dataset`, `RQ_dataset` | Primary pooled score and its shape/detection components across all annotation sets; matches require IoU **strictly greater than 0.5** |
| `observation_PQ_mean` and related observation summaries | Equal weight per observation after averaging its annotation-set results; additional diagnostics |
| `PQ_mean`, `PQ_std` | Variation over annotation-set scores, not variability across training seeds |
| `filaments.csv`, `filament_scores.npz` | Best-overlap Dice/IoU per ground-truth filament, including zero for a missed filament |
| `overlap_pairs.csv`, `overlap_scores.npz` | Every positive-overlap prediction/reference pair, matching the reference's overlap-distribution convention |
| `any_overlap_one_to_many`, `any_overlap_many_to_one` | Fragmentation/merging counts using any positive overlap, as in the reference diagnostics |
| `custom_one_to_many` | Reference filament split across at least two predictions: each prediction lies at least 50% inside it and covers at least 10% of it |
| `custom_many_to_one` | Prediction merging at least two reference filaments: it covers at least 50% of each reference, and each contributes at least 10% of the prediction |
| `annotation_sets.csv`, `observations.csv` | Detailed scores and counts for checking difficult cases and aggregation |

An empty prediction produces false negatives for all reference filaments. The custom strict relationships and the official any-overlap relationships answer different questions; keep their labels when plotting. Neither should silently replace the pooled PQ selection metric. Local validation is a different holdout from Kaggle, so numerical equality or a fixed direction of difference is not expected.

Inference timings in validation separate prediction, connected components, and scoring. They are not the same as full test-pipeline timing. The test exporter measures image reading, preprocessing, transfer/model prediction, components, RLE round-trip verification, and CSV writing; optional probability export is included if enabled, and the cold start is not excluded. Compare timings with the same settings/device and state what was timed. Full optimizer-checkpoint size is also different from compact inference-weight size.

`report.py --runs RUN_DIR [RUN_DIR ...] --output OUTPUT_DIR [--examples]` turns saved run/evaluation artifacts into `model_comparison.csv`, `person_b_results.md`, and learning curves. With evaluation score arrays it also plots best-overlap-per-GT distributions, official positive-overlap-pair distributions, and overlap-link counts separately. `--examples` uses saved validation probabilities and shows the three highest/lowest observation-mean-PQ examples under each run's report subdirectory. These are diagnostic illustrations; the comparison table uses pooled `PQ_dataset`. The report rejects tiny-overfit diagnostic runs and never invents missing scores.

## Person C handoff

Use `evaluate.py --save-predictions` for validation and `predict.py --save-probabilities DIR` for test data. Each `{observation}.npz` contains:

| Key | Format |
|---|---|
| `probability` | float32 array, `(2048, 2048)`, values in `[0, 1]`, before thresholding |
| `instances` | uint32 array, `(2048, 2048)`, 0 background, positive filament IDs |
| `observation` | Scalar string identifying the source image |

Keep the associated checkpoint, resolved config, normalization values, split, evaluation summary, and checkpoint hash with those files. Evaluation writes `config.yaml` and a `summary.json` with checkpoint provenance. Submission export writes `submission.summary.json` containing its resolved config, checkpoint path/hash, image coverage, empty-detection list, and timings. The archives themselves do not duplicate all of that metadata.

For online post-processing, `inference.predict_probability(model, normalized_image, cfg['inference'])` returns probabilities at the input image's original shape. Optional tiles overlap and blend their probabilities **before** one full-image threshold/component operation, avoiding independent instance IDs at tile borders. Person C can replace the basic `probability_to_instances` stage with separation/cleanup and compare on the unchanged validation split. Preserve the probabilities so several post-processing choices can be checked without retraining.

The basic exporter already writes the required two-column instance CSV and verifies each RLE mask round trip. Reuse `predict.instance_rows` / `write_submission` for replacement instance maps. Do not invent an object for a zero-detection image. Its lack of rows is recorded in the sidecar; actual upload acceptance remains to be confirmed.

The Person B training study is complete. Person C should inspect the saved missed/merged/fragmented cases and compare processing changes in separate output directories, retaining the original B artifacts. Record the changed processing settings and evaluate them on the unchanged validation split. All current comparisons used physical batch size 4 without gradient accumulation; equal sample/update-attempt budgets do not imply equal pixels or compute. Person C's compression study must separately report quality, runtime, and deployable size; compact weights that omit optimizer state are not model compression.

The 1024-resolution configuration changes both pixel scale and field of view: it trains on a reduced whole image instead of a 512-pixel full-resolution crop. Equal sample/update budgets therefore do not imply equal processed pixels or compute. Describe it as a comparison of these two practical input strategies, not an isolated test of downsampling alone. Source hashes are saved with runs, but direct training resume does not reject changed source code; finish code changes before comparable research runs and preserve the run metadata.

Use only the provided training segmentation masks for supervision. Do not use annotation `spine`, `bbox`, or `category_id` fields as extra labels; derive any skeleton target from masks. Capture times/instrument codes are for analysis and splitting, not model inputs. Do not obtain the public MAGFiLO release's hidden-test annotations. Select changes using local validation and keep unsuccessful runs in the experiment log.
