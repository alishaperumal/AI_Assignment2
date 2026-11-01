# What the completed Person B study shows

The selected setup is the original ResNet34 U-Net with the added augmentation transforms disabled. Across three fresh training seeds, its validation Panoptic Quality (PQ) was **0.33695 ± 0.00758**. PQ rewards both finding individual filaments and tracing their shapes. These results come from 122 held-out images, scored against all 200 available annotation sets.

The improvement comes mainly from reducing unmatched extra predictions. It does not mean that every filament is traced better or that fewer filaments are missed. The exported representative model, seed 45, makes substantially fewer extra predictions than the baseline but also misses more reference objects. That tradeoff matters when explaining the result.

The underlying numbers are in the [study summary](summary.json), [stage tables and figure gallery](study_report.md), and [fresh-seed results](fresh_seed_summary.csv).

## How the model was chosen

Eleven configurations received the same six-epoch screening budget. The two best alternatives were then trained from scratch for 20 epochs and compared with the completed 20-epoch baseline, all using selection seed 42. The split, foreground threshold of 0.5, and connected-component conversion remained fixed.

| Full-budget method, seed 42 | Best validation PQ | Best epoch |
|---|---:|---:|
| Original baseline | 0.320127 | 20 |
| No added augmentation transforms | 0.338081 | 20 |
| Whole images reduced to 1024 pixels | 0.337451 | 17 |

The winning margin over the 1024 alternative was only **0.000629 PQ**. The recorded rule selected the numerical winner; this is a near-tie, not convincing evidence that one method is generally superior. Only one selection seed was used for this comparison. See the [full-budget comparison](figures/confirmation.png) and [recorded decision](configuration_selection.json).

Here, `no_augmentation` disables the added flips, rotations, brightness/contrast changes, noise and blur. Training still uses changing, foreground-aware random crops. The finding applies to this particular transform recipe, dataset and training budget; it does not establish that augmentation is generally harmful.

The 1024 experiment also changes the field of view and pixels processed per update: it uses whole reduced images instead of full-resolution 512-pixel crops. Its result cannot be attributed to downsampling alone.

## What did and did not help in screening

The six-epoch baseline scored 0.255361. Lower learning rate (0.246478), higher learning rate (0.242397), and BCE plus Tversky (0.241593) did not improve it in this short comparison. Tversky places more emphasis on missed foreground; this experiment does not show a continuity benefit from that change.

U-Net++ (0.279091), disc masking (0.278308), CLAHE (0.276773), consensus targets (0.272291), and local normalization (0.259735) exceeded the short baseline but were not promoted under the fixed top-two rule. They were not tested at the full budget, so they should not be called failed methods. Consensus changes both the target and the weighting of observations with multiple annotations.

The 1024 strategy led screening at 0.310386, ahead of no added transforms at 0.285602. Their ordering narrowly reversed at the full budget. This illustrates why short screening can favor methods that learn quickly. The study identifies a useful setup within a bounded search; it does not establish a global optimum. See [all screening results](screening.csv) and [learning curves](figures/learning_curves.png).

## What the three fresh seeds tell us

After the configuration was frozen, it was trained with seeds 43, 44 and 45. Seed 42 was excluded from this estimate.

| Fresh seed | Best validation PQ | Best epoch |
|---|---:|---:|
| 43 | 0.336359 | 18 |
| 44 | 0.329686 | 20 |
| 45 | 0.344810 | 17 |

The average is **0.33695**, with a **sample standard deviation of 0.00758** across the three runs. The standard deviation describes changes caused by training randomness. It is different from variation between easy and difficult images. The fresh runs still use the same validation split for checkpoint selection, so they do not provide an independent test-set estimate.

Seed 45 was exported because it had the highest validation score among the fresh runs, following the recorded rule. Its 0.34481 score and example images must not replace the three-run average. Neither the baseline nor the 1024 alternative received three fresh repeats, so this study does not establish statistical significance of the differences between methods.

## Fewer extra predictions, but more misses

A true positive (TP) is a prediction/reference pair whose overlap exceeds the matching threshold. A false positive (FP) is an unmatched prediction; a false negative (FN) is a reference object without a sufficiently good match. An FN can still have some predicted pixels on it. Counts below are pooled across annotation sets, so the same image is evaluated separately against each available set of human annotations.

| Validation measure | Baseline, seed 42 | Representative, seed 45 |
|---|---:|---:|
| PQ | 0.320127 | 0.344810 |
| Matched reference objects, TP | 841 | 783 |
| Unmatched extra predictions, FP | 1,416 | 1,007 |
| References without a good match, FN | 449 | 507 |
| Any-overlap fragmented references | 193 | 178 |
| Any-overlap merging predictions | 23 | 22 |
| Best-per-reference filament Dice, including misses | 0.645861 | 0.602816 |
| Mean foreground pixel Dice across annotation sets | 0.638629 | 0.643721 |

The representative model has **409 fewer unmatched predictions but 58 fewer matched reference objects**. Its average best-per-reference Dice is lower. The lower fragmentation count therefore cannot be described as an unqualified structural improvement: predicting less can remove both unwanted fragments and useful parts of real filaments. Merging barely changes.

The controlled seed-42 comparison tells a milder version of the same story: no added transforms reduced FP from 1,416 to 1,219, while TP changed from 841 to 839 and SQ stayed almost unchanged. That supports fewer extra detections as the main source of its PQ gain. The representative comparison also includes seed variation and selection of the best fresh seed.

The [official overlap histograms](figures/official_overlap_distributions.png) include only pairs that overlap at all. The [best-per-reference histograms](figures/best_gt_distributions.png) include a zero when a reference filament has no predicted overlap. Both views are needed: the first alone does not show completely missed objects. The [relation-count plot](figures/overlap_relationships.png) should always be read with TP, FP, FN and PQ.

## What the inspected examples show

The examples were deliberately selected by whole-observation scores and changes, not chosen as a random sample. Each detail panel retains a full-image locator. Crops come from reference-mask pixels, and their displayed PQ remains the score for the whole observation.

- **Successful removal of extras — 20170612135150Bh.** Both models trace the one annotated filament reasonably well. The baseline adds three unmatched predictions elsewhere; the selected model removes those extras. Whole-observation PQ rises from 0.306 to 0.771. The [full-image comparison](figures/validation_20170612135150Bh.png) shows the removed extras; the [detail panel](figures/validation_zoom_20170612135150Bh.png) shows that tracing of the main filament is similar.
- **Lost extent and fragmentation — 20110221173814Mh.** The [detail panel](figures/validation_zoom_20110221173814Mh.png) shows the selected model retaining only a much smaller part of the elongated human outline, plus a tiny detached island. Its two predictions both fail the matching threshold, giving PQ 0; the baseline has one successful match and PQ 0.414. This is a concrete warning against assuming a cleaner-looking prediction is a better segmentation.
- **Small, faint filament missed entirely — 20220104043050Uh.** The baseline matches the small annotated structure, whereas the selected model predicts no object. PQ falls from 0.632 to 0. The [detail panel](figures/validation_zoom_20220104043050Uh.png) makes this difficult small structure visible.
- **A typical score is not an improvement on every image — 20151011144554Bh.** The selected model's observation score is near its median, but falls from the baseline's 0.403 to 0.354. It has five successful matches instead of six and two fragmented references instead of one. The [whole-image comparison](figures/validation_20151011144554Bh.png) shows the multiple structures; the [detail](figures/validation_zoom_20151011144554Bh.png) shows that both models can draw a wider region than the human outline.

These examples support a practical error taxonomy: unmatched extra objects, incomplete filament extent, detached fragments, completely missed faint/small objects, and remaining merges. They do not establish the prevalence of each visual cause beyond the recorded counts.

## Test predictions and their limits

The representative export processed all **180 test images**, producing **1,831 instance rows** and a probability archive for every image. Three images produced no instances: `20180507140210Th`, `20200310113230Ch`, and `20211214174010Mh`. The CSV omits rows for those images rather than inventing objects.

The [test panel](figures/test_examples.png) includes one zero-detection image and two nonempty examples with nine and ten predicted instances. The nonempty examples visibly follow several dark elongated structures. Without test labels, their missed or extra objects cannot be measured. Likewise, the zero-detection example could be an appropriate empty prediction or a failure to detect subtle filaments. These are visual illustrations, not best/worst accuracy claims.

Kaggle acceptance and leaderboard scores remain **unavailable because no upload was requested**. Local RLE round-trip validation is evidence of encoding consistency, not of competition-server acceptance or test accuracy.

## Size, training cost and inference time

The representative model contains **24,430,097 parameters** and its compact checkpoint is **93.36 MiB**. Its training loop took **743.58 seconds**; epoch loops including validation took **1,431.58 seconds**; the complete training subprocess took **1,454.11 seconds**. These are different timing scopes. It completed 4,800 optimizer-update attempts, of which three were skipped by mixed-precision overflow handling, leaving 4,797 applied updates. The rare recorded skips did not make the run incomplete.

On the same RTX 5060 Ti, the representative's complete image-to-CSV benchmark took **96.535 seconds for 180 images**, or **536.306 ms/image**. This includes reading/preprocessing, inference, connected components, RLE checks and CSV writing/closing, with probability-archive saving disabled. Model setup is outside that timer; there was no explicit GPU warm-up. The benchmark CSV is byte-identical to the canonical export. See the [timing protocol](inference_timing_protocol.json) and [measured timing records](inference_benchmarks.json).

The corresponding single-pass averages were 635.271 ms/image for the baseline, 588.878 for no added transforms at seed 42, and 497.987 for the 1024 alternative. The 1024 method is therefore a useful measured speed/quality alternative in this study's near-tie. These are one-pass measurements, not repeated runtime estimates. Different numbers of predicted objects change the work required for component processing and encoding. The selected model has the same architecture and stored precision as the baseline; these timings do not demonstrate compression.

## What remains useful for Person C

The main next target is preserving faint and small filament regions while controlling extra detections, with instance separation and simple post-processing checked on the frozen validation data. Any improvement should report PQ together with TP, FP, FN and fragmentation/merging counts. Compression still needs its own measured quality, size and runtime comparison. The saved probabilities, masks, configs and checkpoints make those studies possible without changing the completed Person B results.

All six full-budget/fresh checkpoints were reloaded and reproduced the recorded validation metrics within the stated `1e-12` tolerance; see [reload verification](reevaluation_verification.json). The staged run records and figures support a reproducible local modelling study. They do not supply an independent hidden-test score, prove that the selected settings are universally best, or complete Person C's enhancement and compression work.
