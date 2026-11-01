# COS711 Assignment 2: solar filament segmentation

The project teaches a neural network to find **solar filaments**: thin, dark structures in grayscale images of the Sun. The final answer must identify each filament separately, rather than only mark all dark pixels. It uses the training images and segmentation annotations provided for the [Filament Segmentation 2026 competition](https://www.kaggle.com/competitions/filament-segmentation-2026).

The assignment specification is `COS711-A2.pdf`; the division of work is in `COS711 Assignment 2 Team Plan and Implementation Guide.docx`. This repository contains corrected data preparation and exploratory analysis, completed model training and controlled experiments, evaluation, Person C's instance post-processing and compression study, the final checked submission CSV, the team slide builder, and ZIP/video helpers. Recording the video and the authorized Kaggle/ClickUP uploads remain human actions.

**Continuing the work or using a new coding assistant? Start with [PERSON_C_START_HERE.md](PERSON_C_START_HERE.md)**, whose first section summarises the completed Person C work and the remaining human actions. [AGENTS.md](AGENTS.md) supplies concise repository guidance for assistants. These current guides supplement the frozen B handoff and do not alter its verified package.

## Person C: post-processing, compression and final submission

**Score-improvement round (after Kaggle 0.32):** `improve.py` trains preprocessing variants (CLAHE + disc mask, local normalisation + disc mask) and scores/exports probability-averaging ensembles. Best validation result: teacher + distilled ResNet18 ensemble, PQ 0.4143 vs 0.4031 submitted (not yet confirmed on Kaggle). See [results/improvements/README.md](results/improvements/README.md).

Full results, tables and figures: [results/person_c/person_c_results.md](results/person_c/person_c_results.md).

* **Post-processing** (`postprocess.py`, configured by the `postprocess` section of any config; defaults reproduce the baseline). The selected rule bridges fragments up to 16 px apart into one instance (no pixels added) and removes regions under 320 px, raising pooled validation PQ of the seed-45 model from **0.3448 to 0.4034** on the same probabilities. It was chosen from declared searches scored by `offline_eval.py`, which first reproduced Person B's result exactly; split-half and bootstrap checks are recorded.
* **Compression** (`compress.py`, `compress_onnx.py`): FP16 weights, structured channel pruning with fine-tuning, distillation with a label-only control, and ONNX Runtime FP32/INT8 on CPU, each compared on validation PQ, file size and end-to-end time.
* **Final model and CSV**: 30% channel-pruned + FP16 U-Net/ResNet34 (11.93 M parameters, 22.9 MB), validation PQ 0.4031, 475 ms per image end-to-end on an RTX 3070. `runs/person_c/final/submission.csv` holds 1,421 filaments for the 180 test images (3 without detections) and was independently decoded and checked. It has not been uploaded.

```powershell
.\.venv\Scripts\python.exe evaluate.py --checkpoint runs/person_b_final/best_weights.pt --config configs/person_c/final_postprocess.yaml --output runs/person_c/check_final_postprocess
.\.venv\Scripts\python.exe compress.py benchmark --variant runs/person_c/compression/pruned30_fp16.pt --name final --config configs/person_c/final_postprocess.yaml --output runs/person_c/rerun_final --precision fp16
```

The first command applies the selected post-processing to the original seed-45 checkpoint; the second reruns the final model's validation and complete test export (CSV plus timing) into a new directory.

## Completed modelling study

Person A's corrected notebook was executed on the supplied data. The original split is unchanged: 584 training observations, 122 validation observations, and one excluded gap observation. Validation scores compare predictions with all 200 available validation annotation sets.

The original baseline was resumed with its original 20-epoch schedule and early stopping; it completed all 20 epochs at **0.32013 pooled validation PQ**. Eleven configurations then completed six-epoch screens. The two strongest alternatives were trained afresh for 20 epochs with selection seed 42:

| Full-budget configuration | Validation PQ |
|---|---:|
| Original baseline | 0.32013 |
| Whole images reduced to 1024 | 0.33745 |
| No added geometric/appearance transforms | 0.33808 |

The last configuration won by only **0.00063 PQ**, a near-tie rather than evidence of statistical superiority. It retains random foreground-biased 512-pixel crops. After freezing that choice, three fresh 20-epoch runs gave:

| Fresh seed | Best validation PQ | Best epoch |
|---|---:|---:|
| 43 | 0.33636 | 18 |
| 44 | 0.32969 | 20 |
| 45 | 0.34481 | 17 |

The fresh-run mean is **0.33695 ± 0.00758 PQ**, using sample standard deviation across the three training runs. Selection seed 42 is excluded. Seed 45 is the representative model under the recorded rule. Its individual score does not replace the mean, and the single-seed baseline does not establish statistical significance of an improvement. PQ measures agreement on separate objects and their shapes; it is not the percentage of pixels classified correctly.

There are **17 completed research training runs and 186 completed epochs** across the staged budgets. The [protocol](results/person_b_study/protocol.json), [configuration decision](results/person_b_study/configuration_selection.json), [fresh-run checks](results/person_b_study/fresh_seed_verification.json), and [plain methodology](results/person_b_study/methodology.md) record the evidence and limitations. All six full-budget checkpoints were reloaded and reproduced their recorded validation metrics. The final Python test suite passed **101 tests**; its 53 warnings concern the installed pycocotools/NumPy deprecation, with mask checks passing.

The representative model has **24,430,097 parameters** and **93.36 MiB** of compact weights. Its complete checked image-to-CSV pipeline took **96.54 seconds for 180 images (536.31 ms/image)** on this GPU, excluding model loading and probability-archive saving. This is one measured pass, not a repeated runtime estimate. The canonical export also saved prediction archives and took 174.12 seconds. The two CSV files are byte-identical.

The representative made fewer extra validation detections than the baseline (1,007 versus 1,416 pooled false positives), but missed more reference filaments (507 versus 449 false negatives). A higher PQ therefore does not imply improvement on every metric. See the [simple results interpretation](results/person_b_study/interpretation.md) and the [full tables and figures](results/person_b_study/study_report.md).

### Final local deliverables

| Artifact | Location |
|---|---|
| Methodology, measured results, learning curves, overlap distributions and paired examples | [Study report](results/person_b_study/study_report.md), [methodology](results/person_b_study/methodology.md), [interpretation](results/person_b_study/interpretation.md) |
| Model/configuration, 122 validation and 180 test prediction archives, CSV and file manifest | [Person B transfer package](runs/person_b_final/) |
| Locally verified submission: 1,831 instances, three zero-detection images recorded in its sidecar | [submission.csv](runs/person_b_final/submission.csv) |
| Person C interfaces, reproduction instructions and remaining work | [Person C handoff](results/person_b_study/person_c_handoff.md) |
| Ten-slide Person B presentation, narration and sources | [PowerPoint](delivery/person_b_results.pptx), [presentation guide](delivery/README.md), [speaker notes](delivery/speaker_notes.md), [references](delivery/references.md) |
| Final checks, artifact hashes and human actions | [Completion verification](results/person_b_completion_verification.json) |

The CSV is locally verified, but competition acceptance and a Kaggle score are deliberately unverified. Person C's separation/compression study, combining the team's presentation, adding names, recording each member, and later authorized uploads remain separate. The complete video must meet the assignment's 10-minute and 50-MB limits. The large internal model/prediction package is not the final assignment ZIP.

The old two-epoch integration artifacts are preserved in `runs/E0_baseline_seed42/archive_epoch2/`. `results/implementation_validation.json` is the historical implementation check, including a tiny memorization diagnostic; it is not the final research result. Large weights and prediction archives are under the Git-ignored `runs/` directory and must be transferred separately. No Kaggle or ClickUP upload has been performed.

## How the pieces fit together

```text
Training images + segmentation annotations
    -> data.py: cached masks, frozen split, preprocessing and crops
    -> models.py + losses.py + train.py: learn foreground probabilities
    -> inference.py: whole-image probabilities at the original resolution
    -> threshold + connected components: one integer ID per predicted filament
    -> evaluate.py + metrics.py: compare with each available annotation set
    -> predict.py: test predictions and a local Kaggle-format CSV
```

The baseline is a **U-Net with a ResNet34 encoder**, initialized from ImageNet weights. It receives one grayscale channel and predicts one foreground score per pixel. Training combines binary cross entropy (BCE, a pixel classification loss) with Dice loss (a foreground overlap loss), using AdamW and a cosine learning-rate schedule. Training uses 512 × 512 crops; validation and exported predictions remain 2048 × 2048. A probability above 0.5 becomes foreground, and each 8-connected group of foreground pixels becomes an instance. Touching filaments can therefore merge, and broken filaments can split: this simple conversion is the baseline for Person C to improve.

| File or directory | Purpose |
|---|---|
| `data.py`, `eda.ipynb` | Data validation, masks, split, crops, preprocessing, and exploratory analysis |
| `splits/split_v1.json` | Frozen observation-level train/validation split |
| `config.py`, `configs/` | Validated YAML settings and experiment variants |
| `models.py`, `losses.py`, `train.py` | Models, losses, resumable training and checkpoint selection |
| `inference.py`, `evaluate.py`, `metrics.py` | Shared prediction and instance-level evaluation |
| `predict.py` | Test predictions, checked RLE encoding, and submission CSV |
| `postprocess.py`, `offline_eval.py` | Person C instance conversion steps and offline scoring of saved probabilities |
| `compress.py`, `compress_onnx.py` | Person C compression variants, distillation/pruning training and benchmarks |
| `person_c_report.py`, `verify_inputs.py`, `make_submission_zip.py` | Person C results/figures, input hash checks, final ZIP assembly |
| `experiments.py` | Sequential experiment runs and result/seed summaries |
| `report.py`, `study_report.py` | Per-run and staged-study tables, learning curves, overlap plots, and prediction examples |
| `local_job.py`, `package_person_b.py` | Sequential job logging/sleep protection and verified model/prediction packaging |
| `delivery/` | Person B slide content, speaker notes, references, and presentation build files |
| `tests/` | Synthetic regression checks, including Windows worker startup and resume |
| `results/`, `figures/` | Recorded EDA and evaluation-reference information |
| `HANDOFF.md` | Data/model interfaces, methodological cautions, and Person C handoff |

## Setup on this Windows computer

Run the commands from the repository root in PowerShell. Person B's `.venv` used Python 3.14.7 and the already installed GPU packages: **PyTorch 2.14.0+cu130 / torchvision 0.29.0**, with an NVIDIA RTX 5060 Ti. Person C's machine uses Python 3.12.9 with **PyTorch 2.13.0+cu132** on an **RTX 3070 (8 GB)**; the same pinned requirements installed there, the original 101 tests passed, and the seed-45 checkpoint reproduced its validation result (PQ 0.34492 against the recorded 0.34481, an expected small hardware difference in mixed precision). Each `.venv` was created with `--system-site-packages` to reuse the large GPU packages; remaining dependencies were installed into `.venv`. This avoids duplicating the GPU stack on the available disk space. Package versions are pinned in `requirements.txt`; Person C's compression tools additionally need `onnx onnxruntime torch-pruning` and the deck builder needs `python-pptx`.

If `.venv` already exists, use it directly. To recreate this arrangement on a computer whose base Python already has the compatible GPU packages:

```powershell
python -m venv --system-site-packages .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

For another computer, install an appropriate PyTorch/torchvision build into its environment first, then install the remaining pinned dependencies. These pins record the tested environment; a different Python/CUDA combination needs its own compatibility check. No system driver changes are required by this project's setup instructions.

Check the interpreter and GPU before starting a long run:

```powershell
.\.venv\Scripts\python.exe -c "import torch; print(torch.__version__); print('CUDA available:', torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

### Adding the data and Person B's model package (each teammate, locally)

The dataset and trained weights are too large for Git, so every teammate adds them to their own copy. `data/`, `cache/` and `runs/` are Git-ignored, so nothing here is ever pushed.

1. **Dataset.** Use the competition zip (`filament-segmentation-2026.zip`, from the Kaggle Data page after accepting the rules; about 671 MB). It contains one folder, `MAGFiLO_1.0_Kaggle_2026`. Extract it and rename that folder to `data` inside the repository, so that `data\train\train_images`, `data\train\MAGFiLO_1.0_Annotations_kaggle2026_train.json` and `data\test\test_images` exist. This is the default `data_root` of every config, so no environment variable is needed.
2. **Person B's package** (`person_b_final.zip`, about 1.5 GB: selected weights, configs, saved validation/test probabilities). Extract it into `runs\` so that `runs\person_b_final\best_weights.pt` exists.
3. **Check both** against the hashes recorded by Person B, then build the annotation cache (about one minute):

```powershell
tar -xf "$HOME\Downloads\filament-segmentation-2026.zip" -C "$env:TEMP"
Move-Item "$env:TEMP\MAGFiLO_1.0_Kaggle_2026" .\data
tar -xf "$HOME\Downloads\person_b_final.zip" -C .\runs
.\.venv\Scripts\python.exe verify_inputs.py
.\.venv\Scripts\python.exe data.py build-cache --data-root data --cache-dir cache --workers 4
```

`verify_inputs.py` must report 888/888 dataset files and 386/386 package files unchanged. (`Expand-Archive` also works but is much slower than the built-in `tar`.) If you prefer to keep the data elsewhere, point `FILAMENT_DATA` (and optionally `FILAMENT_CACHE`) at it, or pass `--data-root`; note that the portable configs in `runs/person_b_final/config.yaml` and `configs/person_c/` name `data`/`cache` explicitly, and explicit YAML paths take precedence over environment variables. Person B's original machine used `$env:FILAMENT_DATA = 'C:\Users\user-pc\Desktop\COS711_Assig2\filament-segmentation-2026\MAGFiLO_1.0_Kaggle_2026'`.

The data directory must contain `train/train_images`, `train/MAGFiLO_1.0_Annotations_kaggle2026_train.json`, and `test/test_images`. `FILAMENT_CACHE` can similarly change the cache directory. Explicit YAML `data.data_root` / `data.cache_dir` settings take precedence over environment defaults. Relative YAML paths resolve from the repository root; `extends` paths resolve from the YAML file containing them.

Build the local cache and calculate normalization values from training observations only (the statistics step is only needed for new training studies):

```powershell
.\.venv\Scripts\python.exe data.py build-cache --data-root data --workers 2
.\.venv\Scripts\python.exe data.py stats --data-root data --workers 2 --max-images 200
.\.venv\Scripts\python.exe -m pytest -q
```

`--workers` controls cache building; the statistics command runs its bounded sample sequentially. Statistics are saved to `cache/norm_stats.json`. The plain-image training sample on this dataset gives mean `0.32688403099775315` and standard deviation `0.22290047382928638`. Put the measured values in `data.preprocess.mean` / `std`; the experiment runner does this automatically, separately for each preprocessing variant. `pos_weight` is a diagnostic estimate, not automatically enabled in the baseline loss.

The split already exists: **584 training observations, 122 validation observations, and 1 excluded gap observation**. Load it unchanged. `data.py split` and `data.py all` reuse an existing split; `save_split` refuses a differing overwrite. `--overwrite` applies only to mask rebuilding. If a future study needs another split, use a different versioned filename and treat its results as a separate comparison.

The corrected `eda.ipynb`, `results/eda_summary.json`, and EDA figures have been regenerated from the supplied dataset and cache. Agreement analysis processes one observation at a time to bound memory. It explicitly labels its cached-mask reference and records 100 sampled observations / 404 ordered pairs; the disc-mask check records its 60 sampled annotation sets. These are exploratory findings, separate from trained-model evaluation.

The dataset, cache, `.venv`, `runs/`, weights, and machine-specific `configs/local*.yaml` are ignored by Git; the shared `configs/local_normalization.yaml` experiment is explicitly included. Keep space for full optimizer checkpoints and prediction archives. Cloning the code does not transfer the trained model or prediction package. The first ImageNet-initialized model creation may download encoder weights.

## Train, resume, evaluate, and export

`configs/local_baseline.yaml` is the machine-specific configuration prepared for this dataset. On a fresh checkout, create a local YAML extending `baseline.yaml`, set dataset/cache paths if not using the environment variables, and fill in the measured mean/std. For example:

```yaml
extends: baseline.yaml
data:
  preprocess:
    mean: 0.32688403099775315
    std: 0.22290047382928638
```

For a new baseline study, use a new output directory and start with the configured GPU. The existing baseline in this checkout is already complete:

```powershell
.\.venv\Scripts\python.exe train.py --config configs/local_baseline.yaml
```

Use a unique `run.output_dir` for each new experiment. Training refuses to reuse a nonempty run directory without an explicit compatible resume. The baseline config plans 20 epochs, with early stopping after 8 validations without improvement. Its 960 sampled crops per epoch keep comparison budgets explicit. Full runs and final scores must be checked in their output directories; do not infer completion from the presence of a config or checkpoint alone.

Each run saves:

- `config.yaml` and `run_metadata.json`: resolved settings, environment, and split hash.
- `history.csv` / `history.json`: loss, learning rate, validation scores, and recorded timings per epoch.
- `last.pt`: latest complete epoch, including optimizer, scheduler, mixed-precision state, and random states.
- `best.pt`: the full checkpoint with the best validation `PQ_dataset`.
- `best_weights.pt`: compact model/config checkpoint for evaluation and prediction.
- `summary.json`: best score/epoch, completed epochs, stop reason, parameter count, and checkpoint sizes.

To pause a run after a limited number of epochs, use `--max-epochs N` without changing `train.epochs`. Resume from its last complete epoch:

```powershell
.\.venv\Scripts\python.exe train.py --config configs/local_baseline.yaml --resume runs/E0_baseline_seed42/last.pt
```

Resume checks training settings and the frozen split. It keeps the original cosine-schedule horizon. An interruption halfway through an epoch reruns that epoch; `best_weights.pt` cannot resume optimizer training. Reproducibility is tested for the same setup, not promised bit-for-bit across different hardware/software versions.

A separate local diagnostic config can use `--overfit 2` to check whether the network learns two fixed training images/crops. Use a separate output directory. This disables augmentation and fixes the samples; its tiny-subset validation scores are **not research results**.

Reproduce the selected checkpoint's validation predictions in a separate output directory:

```powershell
.\.venv\Scripts\python.exe evaluate.py --checkpoint runs/final/no_augmentation_seed45/best_weights.pt --output runs/reproduced_evaluation --save-predictions
```

Reproduce its test CSV, optionally keeping probabilities and instance maps:

```powershell
.\.venv\Scripts\python.exe predict.py --checkpoint runs/final/no_augmentation_seed45/best_weights.pt --output runs/reproduced_test/submission.csv --save-probabilities runs/reproduced_test/predictions
```

Both commands use the checkpoint's saved settings by default. `--data-root` relocates the dataset; `--device cpu` or `--device cuda` selects the device. `--config` supplies a complete resolved override, so use it deliberately and keep the saved output config when comparing results. A model-definition mismatch is rejected.

The complete staged-study report can be regenerated from the existing run records:

```powershell
.\.venv\Scripts\python.exe study_report.py --require-complete
```

This updates the report in the original repository. Preserve the verified transfer package as a frozen snapshot. For a separate per-run report:

```powershell
.\.venv\Scripts\python.exe report.py --runs runs/E0_baseline_seed42 --output runs/reports --examples
```

Pass several run directories after `--runs` to compare them. Outputs include `model_comparison.csv`, `person_b_results.md`, and `learning_curves.png`. Evaluation arrays add `filament_distributions.png`, `official_overlap_distributions.png`, and `overlap_relationships.png`. `--examples` needs the saved validation predictions from `evaluate.py --save-predictions`; it writes best/worst validation panels under a report subdirectory for each run. The ranking uses observation-mean PQ for illustration; the table uses the primary pooled `PQ_dataset`. Tiny-overfit diagnostic runs are rejected.

The CSV has `filament_id,segmentation_rle`, one row per predicted filament. Encoding uses COCO compressed RLE and verifies an exact decode round trip for every 2048 × 2048 mask. An image with no detections contributes no artificial object; it is listed in the `.summary.json` sidecar. The command writes local files only. Actual Kaggle upload acceptance and the leaderboard score still need to be checked on the competition account.

## Controlled experiments and scoring

Available variants cover learning rate, whole images reduced to 1024, disc masking, CLAHE, local normalization, augmentation removal, consensus targets, BCE + Tversky, and U-Net++. Keep the same split, seed, training budget, and inference policy unless that setting is the deliberate comparison.

Run a small comparison or prepare its exact configs before training:

```powershell
.\.venv\Scripts\python.exe experiments.py --base configs/local_baseline.yaml --configs baseline tversky --output runs/example_screen --epochs 6 --samples-per-epoch 960 --seeds 42 --prepare-only
.\.venv\Scripts\python.exe experiments.py --base configs/local_baseline.yaml --configs baseline tversky --output runs/example_screen --epochs 6 --samples-per-epoch 960 --seeds 42
```

`--base` supplies only the machine's data/cache/split paths. Named experiment configs supply model and training choices. The runner computes training-only normalization for each preprocessing choice, starts one fresh training process at a time, and writes `configs/`, `logs/`, `records/`, `experiments.csv`, and `seed_summary.csv` beneath its output directory. Suite runs disable early stopping so every comparison receives the requested fixed epoch budget; direct `train.py` runs retain configurable early stopping. Use `--resume` for interrupted compatible runs; completed compatible runs are skipped. A changed budget/config needs a new output directory. `--aggregate-only` rebuilds result tables from recorded runs. Seed aggregates also separate runs with different recorded source-code fingerprints.

This study selected with seed 42, then used **fresh seeds 43, 44, and 45** at the same full budget. Preserve that separation when regenerating its reports. Seed summaries use the sample standard deviation across completed independent runs. A single run has no estimated across-run standard deviation. Per-image variation is a different statistic. Failed, partial, and diagnostic runs are excluded from seed aggregates but retained in experiment records; missing Kaggle scores remain blank.

The primary score is **`PQ_dataset`**: a filament must match a reference filament at an intersection-over-union (IoU) strictly greater than 0.5, then matched IoUs and detection counts are pooled across all annotation sets. This follows the organizer's [Self Evaluation Notebook, version 6](https://www.kaggle.com/code/azimahmadzadeh/self-evaluation-notebook), recorded with its source hash in `results/evaluation_reference.json`. PQ combines segmentation quality (SQ) and recognition quality (RQ): finding the right objects and tracing their shapes both matter. The public reference does not establish every private leaderboard edge case.

Validation also reports equal-observation means, per-filament overlap distributions, and fragmentation/merging diagnostics. The organizer's relation plots use **any positive overlap**; stricter custom relation counts are labeled separately. Original annotation masks are decoded for rare overlapping-label sets during evaluation so that cache overlap handling does not erase evaluation ground truth. See `HANDOFF.md` for field names and interpretation.

Local validation and Kaggle use different images. Neither score is guaranteed to be higher. Tune on the frozen local validation split, record negative results, and use Kaggle as the required external check rather than repeatedly selecting changes from leaderboard feedback. Only supplied training segmentation labels are used: no external test annotations, annotation spines/bounding boxes/categories as supervision, or time/instrument metadata as model inputs.
