# Person B staged modelling study

## Study design

Screening identifies promising methods, full-budget runs confirm candidates, and fresh seeds estimate training variability. Seed 42 is used for selection and excluded from the fresh-seed mean.

Selected method: **no_augmentation**. Highest best-checkpoint PQ_dataset at full budget. Exact ties prefer baseline, then fewer parameters, then variant name.

Completed screening / confirmation / fresh runs: 11/11, 3/3, 3/3.

## Screening results

| Variant | Seed | PQ | Epochs completed / planned | Status |
|---|---:|---:|---:|---|
| baseline | 42 | 0.25536 | 6 / 6 | completed |
| lr_low | 42 | 0.24648 | 6 / 6 | completed |
| lr_high | 42 | 0.24240 | 6 / 6 | completed |
| resolution_1024 | 42 | 0.31039 | 6 / 6 | completed |
| disc_mask | 42 | 0.27831 | 6 / 6 | completed |
| clahe | 42 | 0.27677 | 6 / 6 | completed |
| local_normalization | 42 | 0.25974 | 6 / 6 | completed |
| no_augmentation | 42 | 0.28560 | 6 / 6 | completed |
| consensus | 42 | 0.27229 | 6 / 6 | completed |
| tversky | 42 | 0.24159 | 6 / 6 | completed |
| unetplusplus | 42 | 0.27909 | 6 / 6 | completed |

## Full-budget confirmation

| Variant | Seed | PQ | Epochs completed / planned | Status |
|---|---:|---:|---:|---|
| baseline | 42 | 0.32013 | 20 / 20 | completed |
| no_augmentation | 42 | 0.33808 | 20 / 20 | completed |
| resolution_1024 | 42 | 0.33745 | 20 / 20 | completed |

## Fresh-seed results

| Metric | Mean | Sample SD across runs | Runs |
|---|---:|---:|---:|
| PQ_dataset | 0.33695 | 0.00758 | 3 |
| SQ_dataset | 0.67535 | 0.00291 | 3 |
| RQ_dataset | 0.49890 | 0.00909 | 3 |
| tp | 818.66667 | 30.92464 | 3 |
| fp | 1175.33333 | 148.93735 | 3 |
| fn | 471.33333 | 30.92464 | 3 |
| sum_iou | 552.83806 | 18.90839 | 3 |
| filament_dice_mean | 0.62825 | 0.02223 | 3 |
| filament_iou_mean | 0.51003 | 0.01778 | 3 |
| pixel_dice_mean | 0.64046 | 0.00292 | 3 |
| pixel_iou_mean | 0.48794 | 0.00433 | 3 |
| observation_PQ_mean | 0.33092 | 0.00633 | 3 |
| any_overlap_one_to_many | 172.33333 | 8.96289 | 3 |
| any_overlap_many_to_one | 22.00000 | 4.00000 | 3 |
| custom_one_to_many | 86.33333 | 3.21455 | 3 |
| custom_many_to_one | 19.00000 | 2.64575 | 3 |
| baseline_prediction_and_components_ms_per_image | 119.74162 | 0.58022 | 3 |

## Representative model and timing

Model: no_augmentation, seed 45. Highest validation PQ among the three fresh-seed best checkpoints; exact ties prefer lower seed.

| Quantity | Recorded value |
|---|---:|
| Parameters | 24430097.000 |
| Compact weights MiB | 93.360 |
| Training-loop seconds | 743.582 |
| Epoch-loop seconds including validation | 1431.578 |
| Latest training subprocess wall seconds | 1454.112 |
| Planned optimizer-update attempts | 4800.000 |
| Recorded optimizer-update attempts | 4800.000 |
| Applied optimizer updates | 4797.000 |
| AMP-skipped optimizer updates | 3.000 |
| Test export milliseconds per image | 536.306 |
| Test images | 180.000 |

Test timing scope: read + preprocess + transfer + model + components + RLE round-trip + CSV; optional probability export included; no warm-up excluded

Reported test milliseconds per image: 1000 * end_to_end_seconds / n_images; complete CSV-operation average

Training subprocess timing scope: latest recorded train.py subprocess invocation, including setup, epoch loops and checkpoint writes; resumed runs may have earlier invocations

Kaggle score: **unavailable — no submission requested**. Test examples have no accuracy labels.

## Interpretation and limitations

- Screening and full-budget confirmation are separate comparisons; compare only matching budgets.
- Seed 42 selected the method. Fresh seeds estimate training variability on the same validation split; they do not remove validation-selection bias or provide an independent test score.
- Across-run standard deviations use ddof=1. Per-observation/annotation-set variation is a different statistic and is never substituted for run variability.
- pixel_dice_mean and pixel_iou_mean average foreground-mask scores across annotation sets. They measure semantic pixel coverage, distinct from pooled instance PQ and best-per-GT filament overlap. Good foreground coverage can still merge or split individual filaments.
- The 1024 trial changes resolution, field of view, pixels per update and foreground sampling: whole reduced images are compared with full-resolution 512-pixel crops.
- Consensus changes target construction and observation weighting; with two annotators its tie-inclusive vote is a union. It is not a pure voting-only experiment.
- Tversky weights missed foreground more heavily; it does not directly enforce thin structures or topology. Any continuity benefit must be supported by observed errors.
- Official overlap distributions include every positive-overlap pair. Best-per-GT distributions include missed objects as zero. Relation counts repeat predictions across annotation sets.
- Read fragmentation/merging counts alongside PQ, RQ, matched objects (TP), missed objects (FN) and extra predictions (FP). Fewer relation errors can result from missing more filaments, rather than improved segmentation. All these counts are pooled over annotation-set comparisons.
- Training seconds cover the training loop; epoch totals include validation but exclude setup and checkpoint serialization. Job sidecars, when supplied, record invocation wall time.
- Experiment-record subprocess wall time includes setup, epoch loops and checkpoint writes. It covers the latest recorded train.py invocation, not necessarily every invocation of a resumed run. Baseline job-sidecar wall time likewise remains scoped to its resumed invocation.
- Training budgets match samples and planned optimizer-update attempts. Mixed-precision gradient scaling may skip an update on overflow; completed updates and AMP skips are reported separately. A few recorded skips are not by themselves a failed run.
- Default inference uses mixed precision with float32 stored weights. Compact checkpoints omit optimizer state; that alone is not model compression.
- Test images have no supplied labels. Their panels illustrate predictions without accuracy rankings. Kaggle scores are unavailable because no upload is requested.
- GPU training runs are sequential.
- Same frozen split, 960 samples and 240 planned optimizer update attempts per epoch; input resolution changes compute and field of view, not just pixel scale.
- Separate training-only normalization per preprocessing variant, bounded to 200 training observations.
- Consensus changes target construction and observation weighting.
- Tversky prioritizes missed foreground; it does not explicitly constrain topology.
- Best validation PQ chooses checkpoints and configurations. Validation scores are not an unbiased hidden-test estimate.
- Three fresh seeds estimate training-run variability; per-image standard deviations are separate.
- Single-seed full-budget comparisons cannot establish statistical significance of gains over the baseline.
- Threshold0.5, original-resolution scoring, and simple connected components stay fixed; advanced separation/compression remain Person C work.
- No Kaggle or ClickUP uploads; no external test labels or prohibited metadata.
- Six-epoch screening may favor quick-learning configurations; only the two promoted alternatives and baseline are compared at full budget. No global optimum is claimed.
- The no_augmentation variant disables added geometric and photometric transforms; random foreground-biased crop sampling remains enabled.
- The representative checkpoint may be the best fresh seed by validation PQ; its individual score/plots do not replace the mean over all fresh seeds or establish significance over a single baseline run.

## Figures

Links stay within this study folder when it is copied into a handoff package.

- [Screening comparisons](<figures/screening.png>): Short equal-budget runs using the selection seed.
- [Full-budget comparisons](<figures/confirmation.png>): Promoted candidates and baseline using the selection seed.
- [Learning curves](<figures/learning_curves.png>): Training loss and validation PQ; loss scales can differ between methods.
- [Official overlap-pair distributions](<figures/official_overlap_distributions.png>): Every positive-overlap pair; missed objects do not contribute zeros.
- [Best-per-GT overlap distributions](<figures/best_gt_distributions.png>): Each reference filament contributes its best match, including zero for a miss.
- [Fragmentation and merging](<figures/overlap_relationships.png>): Any-positive-overlap counts pooled across annotation sets; read alongside PQ, RQ, TP, FP and FN.
- [Paired validation examples](<figures/validation_paired.png>): Same labelled images for baseline and selected model; highest/lowest scores, changes and a typical case are deliberately selected.
- [Representative test predictions](<figures/test_examples.png>): Unlabelled visual examples; no best/worst accuracy ranking.

### Validation details

Each panel retains the whole image and marks a crop derived from reference-mask pixels. PQ labels refer to the whole observation, not the displayed crop.

- [20170612135150Bh](<figures/validation_zoom_20170612135150Bh.png>): Highest selected-model observation mean PQ.
- [20110221173814Mh](<figures/validation_zoom_20110221173814Mh.png>): Lowest selected-model observation mean PQ.
- [20220104043050Uh](<figures/validation_zoom_20220104043050Uh.png>): Largest observed drop (or smallest gain).
- [20151011144554Bh](<figures/validation_zoom_20151011144554Bh.png>): Typical selected-model observation score.
