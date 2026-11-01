# Score-improvement round (branch `feat/improvements`)

Motivation: the submitted CSV (30%-pruned FP16 model, cutoff 0.5, fragment bridging 8, minimum area 320) scored **0.32 on Kaggle**, against **0.4031** local validation PQ. All numbers below are pooled `PQ_dataset` on the unchanged frozen validation split (122 observations, 200 annotation sets, strict IoU > 0.5), the same scorer as the organiser's self-evaluation notebook (version 6). Intervals are paired 95% observation-bootstrap intervals (1,000 resamples) of the difference from the submitted configuration.

## What was tried

| Idea | Validation PQ | Change vs submitted 0.4031 |
|---|---:|---:|
| New model: CLAHE contrast enhancement + zero outside solar disc (seed 46, 20 epochs) | 0.3996 | -0.0035 |
| New model: local brightness normalisation + zero outside disc (seed 47, 20 epochs) | 0.3808 | -0.0223 [-0.0383, -0.0068] |
| **Ensemble: teacher + distilled ResNet18** (average probabilities), t=0.5, min area 320 | **0.4143** | **+0.0112 [-0.0011, +0.0245]** |
| Ensemble: teacher + distilled ResNet18 + CLAHE + local-norm models, min area 240 | 0.4080 | +0.0048 [-0.0073, +0.0171] |
| Ensemble: all six models, t=0.45, min area 320 | 0.4079 | +0.0047 [-0.0079, +0.0177] |
| Cutoff 0.35-0.45 instead of 0.5 (any ensemble) | at most 0.4109 | not better than 0.5 |

Both preprocessing models were trained with the selected recipe (random foreground-biased 512 crops, no added transforms, BCE + Dice, AdamW 3e-4, cosine, 20 epochs) and training-only normalisation statistics; checkpoints were selected with the deployed post-processing. Neither preprocessing method beat the plain-image model on its own, consistent with Person B's six-epoch screen. Every preprocessing comparison here is a single training run.

Split-half check for the two-model ensemble: chosen on validation half B, it gained **+0.0150** on held-out half A; chosen on half A, a near-identical setting (t=0.45) was selected and lost 0.0027 on half B. The improvement is therefore suggestive, not established.

## Output CSVs (both checked: every RLE decoded, unique consecutive IDs, no overlapping masks, 177 of 180 images with rows)

| File (under `runs/improvements/submissions/`) | Models | Parameters | Weights | End-to-end ms/image (RTX 3070) | Filaments |
|---|---|---:|---:|---:|---:|
| `A_teacher_kd18/submission.csv` (recommended) | U-Net/ResNet34 + U-Net/ResNet18 (distilled) | 38.75 M | 148.1 MiB | 556 | 1,286 |
| `B_four_model_preproc/submission.csv` (hedge) | A + CLAHE model + local-norm model | 87.61 M | 334.8 MiB | 860 | 1,418 |

SHA-256: A `513aa0cec69b6db86879e1f59e4bf958e89abf80a596a58b79ee4f459edc4b65`; B `f4ce783fef1f1ee8f58a71a23a71f570035caef70349e2d62e748fffadbb5347`.

The export path was checked against the scored validation probabilities (maximum difference 1e-5 on three validation images per model). Training time: about 12 minutes of epoch loops per new model on the RTX 3070 (the CLAHE run's recorded 1,545 s includes one epoch slowed by another GPU application).

Kaggle scores for A and B are not yet known; record them here after uploading. The large local-to-Kaggle gap means a validation gain of about 0.01 may not transfer.

## Reproduce

```powershell
.\.venv\Scripts\python.exe improve.py train --config configs\improvements\clahe_disc_seed46.yaml
.\.venv\Scripts\python.exe improve.py train --config configs\improvements\localnorm_disc_seed47.yaml
.\.venv\Scripts\python.exe improve.py convert --predictions runs\person_c\compression\bench_teacher_amp16\validation\predictions --output runs\improvements\probs\teacher
.\.venv\Scripts\python.exe improve.py cache --model runs\improvements\clahe_disc_seed46\best_weights.pt --output runs\improvements\probs\clahe46 --subsets val
.\.venv\Scripts\python.exe improve.py score --spec configs\improvements\ensemble_with_localnorm.yaml --output runs\improvements\score.json
.\.venv\Scripts\python.exe improve.py export --models runs\person_b_final\best_weights.pt runs\person_c\compression\kd_resnet18\best_weights.pt --output runs\improvements\submissions\A_teacher_kd18\submission.csv
```

(`convert` is repeated for `bench_kd_resnet18`, `bench_label_resnet18`, `bench_pruned30_fp16`, `bench_pruned50` as `kd18`, `label18`, `pruned30`, `pruned50`; `cache` for the local-norm model as `localnorm47`.) Raw score tables: `score_existing.json`, `score_with_clahe.json`, `score_with_localnorm.json` in this folder.
