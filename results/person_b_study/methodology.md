# Person B modelling methodology

We train a neural network to identify solar filaments in grayscale images of the Sun. The network predicts a foreground probability for every pixel. A simple conversion then turns connected foreground regions into individual predicted filaments. The study asks which of a small, fixed set of modelling choices gives the strongest local validation result, and how much the chosen method varies across repeated training runs.

The experiment order, budgets, selection rules, and limitations are recorded in the [preregistered protocol](protocol.json). This document explains that design; measured outcomes belong in the study's result tables. In the original repository, `COS711-A2.pdf` and `COS711 Assignment 2 Team Plan and Implementation Guide.docx` define the wider task and division of work.

## Data and validation

The supplied labelled data contains 707 distinct observations and 1,154 annotation sets. Some observations were labelled by more than one person. The unlabelled test set contains 180 observations. These counts and the exploratory findings are recorded in `results/eda_summary.json` in the original repository.

We keep the existing frozen split, `splits/split_v1.json` in the original repository: 584 observations for training, 122 for validation, and one excluded observation in the gap between the groups. Every annotation set belonging to an observation stays on the same side. The validation observations contain 200 annotation sets, all of which are used for scoring. The transfer package retains this same file at `splits/split_v1.json` relative to its package root.

The split groups observations into 14-day time blocks and excludes training observations within one day of a validation observation. Its instrument proportions were checked against test-image filenames. This reduces the risk of evaluating on near-duplicates of training images. It does not prove complete statistical independence or demonstrate generalization to an unseen instrument: instruments occur on both sides. The split is never changed in response to model results.

Training uses the supplied segmentation masks. Annotation spines, bounding boxes, categories, capture times, and instrument codes are not additional model inputs or training targets. Filenames support exploration and splitting; no external test labels are used.

## Baseline model and training

The baseline is a U-Net with a ResNet34 encoder initialized from ImageNet weights. An encoder extracts image features; U-Net's decoder combines those features with spatial detail to produce a pixel-level prediction. The model receives one grayscale channel and returns one foreground logit per pixel. A sigmoid converts logits to probabilities during inference.

Training combines binary cross entropy, which penalizes wrong pixel predictions, with soft Dice loss, which rewards overlap with foreground. Both terms have weight 1. The overlap term is calculated per image and averaged across the batch, including crops with no foreground. This is an explicit implementation of the chosen loss, rather than a claim that every library's Dice implementation is identical.

The baseline uses AdamW, an initial learning rate of 0.0003, weight decay of 0.0001, a cosine learning-rate schedule, and gradient clipping at 1. CUDA mixed precision reduces training memory use. The physical batch size is four and gradient accumulation is one for every configured comparison.

The baseline trains on random 512 × 512 crops at the original image scale. When foreground exists, 70% of crop attempts select a foreground pixel to include. Later augmentation can move or remove that foreground. Geometric and appearance transformations include flips, rotations, affine transformations, brightness/contrast changes, noise, and blur. Their exact probabilities are saved in each resolved configuration.

An epoch means **960 sampled training examples** in this study. It does not necessarily mean one pass through all distinct observations. Balanced shuffled cycles determine which items are sampled, and crop/augmentation randomness is determined by the run seed, epoch, and sample index. This avoids repeatedly sampling only the beginning of the dataset and makes sampling independent of worker scheduling.

## Fixed comparisons

All comparisons retain the frozen split and the baseline settings except for the stated change and any directly required preprocessing statistics.

| Configuration | Change being examined |
|---|---|
| `baseline` | U-Net, ResNet34, BCE + Dice, 512-pixel crops |
| `lr_low` | Initial learning rate 0.0001 |
| `lr_high` | Initial learning rate 0.001 |
| `resolution_1024` | Whole images reduced to 1024 × 1024 for training and inference |
| `disc_mask` | Mask the region outside the detected solar disc |
| `clahe` | Apply local contrast enhancement |
| `local_normalization` | Apply local intensity normalization |
| `no_augmentation` | Disable the added geometric and appearance transformations |
| `consensus` | Build one voted foreground target per observation |
| `tversky` | Replace the Dice term with Tversky, retaining BCE |
| `unetplusplus` | Replace U-Net with U-Net++, retaining the ResNet34 encoder |

Normalization mean and standard deviation are measured from at most 200 training observations, using the same deterministic observation sample for each preprocessing recipe. Validation and test images are excluded. Each recipe has its own cached statistics, identified by the data location, split, catalog, and preprocessing settings. These statistics are then fixed across seeds.

Three comparisons need particularly careful interpretation:

- **Input strategy:** the 1024 variant sees a reduced whole image, whereas the baseline sees a full-resolution crop. It changes pixel scale, field of view, foreground sampling, and compute together. A 1024 example contains four times as many input pixels as a 512 crop. Equal example counts therefore do not imply equal pixel counts or computing cost. The comparison tests two practical input strategies; it cannot isolate downsampling alone. All current configurations use whole-image inference. Tiled inference is available in the implementation but is not a measured variant in this study.
- **Annotation targets:** the baseline has one training item per annotation set, so an observation with several annotations receives more weight. Consensus has one item per observation. Its foreground vote includes ties: two annotators give a union, while three require two votes. Thus consensus changes both target construction and observation weighting. Connected components of a consensus foreground mask can also join touching regions. Validation still scores each original annotation set separately.
- **Augmentation and loss:** disabling augmentation leaves random, foreground-biased crop sampling enabled. Tversky uses false-positive weight 0.3 and false-negative weight 0.7, putting more emphasis on missed foreground. It does not explicitly enforce thin structures, skeletons, or continuity; any such improvement must be demonstrated by the results.

## Budgets and selection

The study has three stages:

1. **Screening:** train all 11 configurations from their usual initialization for six epochs with seed 42. Each receives 5,760 sampled examples and 1,440 planned optimizer-update attempts. Rank by the best validation `PQ_dataset`. Promote the top two non-baseline configurations; exact ties prefer fewer parameters, then configuration name.
2. **Full-budget confirmation:** train those two configurations afresh for 20 epochs with seed 42. Compare them with the baseline at the same 20-epoch budget. A six-epoch checkpoint is not extended, because changing the cosine-schedule horizon would create a different training history. The original baseline may be reused only if it completes all 20 epochs; otherwise a fresh fixed-budget baseline supplies the comparison. Select the highest best-checkpoint `PQ_dataset`, with exact ties preferring the baseline, then fewer parameters, then configuration name. No combinations or additional tuning are introduced.
3. **Fresh-seed estimation:** freeze the selected configuration and train it afresh for 20 epochs with seeds 43, 44, and 45. Report the mean and sample standard deviation across all three runs. Seed 42 is excluded from that mean. Select the representative checkpoint for exported predictions by the highest validation PQ among the three fresh runs; exact ties prefer the lower seed.

Suite runs disable early stopping. A 20-epoch run receives 19,200 examples and 4,800 planned update attempts. Mixed-precision gradient scaling can skip an update after detecting an overflow. The experiment budget fixes samples and attempted updates; applied updates and skipped updates are reported separately. A small number of recorded skips does not by itself make a run invalid.

Validation selects the best checkpoint within each run as well as the configuration between runs. Scores therefore describe validation-selected performance, not an unbiased estimate on unseen test labels. Six-epoch screening can favor quick-learning methods, and only the two promoted alternatives and baseline receive full-budget confirmation. No globally optimal configuration is claimed. The best fresh seed's individual score must not replace the three-run mean. A comparison against a single baseline seed cannot establish statistical significance.

## Predictions and scores

Inference produces probabilities at the original 2048 × 2048 resolution. The 1024 strategy restores probabilities to that size before scoring. Pixels with probability **strictly above 0.5** become foreground; each 8-connected component becomes one instance. Threshold, connectivity, and scoring resolution remain fixed. Touching filaments can merge and broken filaments can fragment under this deliberately simple conversion.

The primary score is **pooled Panoptic Quality (`PQ_dataset`)**. Predicted and reference filaments match when their intersection-over-union is strictly above 0.5. PQ combines the overlap quality of matched objects, called SQ, with the quality of detecting the right objects, called RQ. It penalizes missed objects and extra predictions as well as poor boundaries.

The organizer's reference pools matched IoUs, false positives, and false negatives across all annotation sets. We follow that policy; 200 annotation-set comparisons are not the same as 122 equally weighted observations. The official source and its hash are recorded in `results/evaluation_reference.json` in the original repository. Original binary annotation layers are used for rare overlapping annotations so that the integer-mask cache does not discard scoring ground truth.

Additional outputs answer different questions:

- Pixel Dice/IoU measure foreground overlap and can remain high despite incorrect separation into individual filaments.
- Observation-mean PQ gives each observation equal weight after averaging its annotation-set scores; it is a diagnostic, not the selection metric.
- Best-overlap-per-reference-filament Dice/IoU distributions include missed filaments as zero. Official overlap-pair distributions include every pair with positive overlap instead. Their labels must remain distinct.
- Fragmentation and merging counts are reported with PQ, TP, FP, and FN. Fewer relation errors can result from detecting fewer filaments, so a lower count alone does not establish an improvement. Counts repeat predictions across annotation-set comparisons, following the evaluation policy.

Per-image or per-annotation variation is different from variation across training seeds. Only the three fresh runs supply the reported across-run standard deviation.

## Time, size, and handoff

Training-loop time includes loading batches and learning updates. Epoch-loop time also includes validation, but excludes setup and checkpoint serialization. Job sidecars, where supplied, measure invocation wall time; a resumed invocation is not automatically the entire training run. These quantities are labelled separately.

Validation prediction timing includes transfer to the device, model inference, and return to the CPU. Connected-component time and scoring time are recorded separately. This prediction timing excludes image reading and preprocessing, so it is not an end-to-end deployment number.

The test exporter measures reading, preprocessing, transfer/inference, components, RLE encoding and round-trip verification, and CSV output. Saving probability archives is included when enabled; model loading and setup occur before this measurement. The first image is not excluded as a warm-up. Comparisons must use the same device and export settings, and state these boundaries. No compression speedup should be inferred from differently timed pipelines.

Model size is reported as parameter count and compact inference-checkpoint size. Full training checkpoints also contain optimizer and resume state and are larger. Stored weights remain float32 even when inference uses mixed precision. Removing optimizer state is not model compression.

The representative model's validation probabilities, test probabilities, instance maps, configuration, frozen split, and checked submission CSV form the Person C handoff. Checkpoint hashes bind those exports to the weights that produced them. Test images have no supplied labels: their panels can support cautious visual discussion, but cannot be ranked by measured accuracy. Labelled validation examples provide the measured success/failure comparisons.

Advanced instance separation, post-processing, and compression remain Person C work. The local study does not perform Kaggle or ClickUP uploads; a verified CSV does not establish leaderboard acceptance or a Kaggle score. The Person B handoff package is also distinct from the assignment's final presentation-and-code submission archive.
