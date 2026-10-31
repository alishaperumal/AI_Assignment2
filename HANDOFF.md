# Handoff: Person A → Person B

Person A's part (data and evaluation) is done and pushed. This file covers what exists,
how to use it, and what to watch out for. Your part is the baseline, losses, training
loop, tuning and the experiment log (Steps 4 and 5 of the team plan).

## 1. What Person A delivered

| File | What it gives you |
|---|---|
| `data.py` | Cached instance masks, the frozen split, preprocessing, augmentation, and PyTorch datasets and loaders |
| `metrics.py` | PQ, SQ, RQ, per-filament Dice and IoU, one-to-many and many-to-one counts |
| `eda.ipynb` | EDA, figures and numbers for the Dataset slide |
| `splits/split_v1.json` | The **frozen** train/validation split. Never regenerate it. |
| `tests/` | 23 tests (`pytest -q`) |
| `results/eda_summary.json` | Machine-readable EDA facts |

## 2. One-time setup on your machine

```bash
git pull
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
ln -s /path/to/MAGFiLO_1.0_Kaggle_2026 data     # your own copy of the Kaggle data
python data.py build-cache                       # about 2 min, creates cache/ (git-ignored)
pytest -q                                        # expect: 23 passed
```

`cache/` and `data/` are never committed, so everyone builds their own cache. It is
deterministic, so you will get identical masks. On Kaggle or Colab, set the data location
with `export FILAMENT_DATA=/path/to/data` and the cache location with `export FILAMENT_CACHE=/path`.

Then run `python data.py stats`. It prints the image mean and std and the BCE `pos_weight`.
Put the mean and std into your config (see section 4).

## 3. Facts about the data

- 1154 annotation sets over 707 observations (one observation = one image). Observations
  have 1, 2 or 3 annotation sets (411 / 145 / 151). Not every image has every annotator.
- Images are 2048×2048 8-bit grayscale JPEGs. Filaments are thin dark ribbons on the solar disc.
- Filaments cover about **0.36%** of pixels, roughly 280 background pixels per filament pixel.
  Plain BCE will predict all background. This is why the loss needs Dice, Tversky or similar.
- About 7 filaments per annotation set on average (max 26). No set is empty.
- 6 instruments (Bh, Ch, Lh, Mh, Th, Uh), evenly spread in train and test.
- **Split:** 584 train observations, 122 validation, 1 dropped by a 1-day gap. Validation is
  whole 14-day time blocks, chosen so its instrument mix matches the test file names.
- Train and test dates both span 2011 to 2022. A held-out time block is probably *harder*
  than the leaderboard, so expect local PQ a little below Kaggle PQ.
- Annotator agreement (the human ceiling) is in `results/eda_summary.json` under
  `annotator_agreement`. Quote that number next to your model's PQ.

## 4. Getting data into your training loop

```python
import data as D

train_dl, val_dl = D.make_loaders({
    "crop": 512,
    "pos_frac": 0.7,                  # 70% of crops are centred on a filament
    "batch_size": 8,
    "num_workers": 4,
    "target": "per_set",              # "consensus" is ablation E2
    "preprocess": {"mean": 0.5, "std": 0.25},   # replace with values from `python data.py stats`
    "aug": {},                        # {} = defaults; set a key to 0 to switch it off
    "augment": True,
    "epoch_length": None,             # None = one pass over all annotation sets; larger = more crops
    "seed": 42,
})
```

**Training batch** (from `train_dl`), a dict:

| Key | Shape / dtype | Meaning |
|---|---|---|
| `image` | (B, 1, 512, 512) float32 | Normalised, **one channel**. Use `in_channels=1`. |
| `mask` | (B, 1, 512, 512) float32 | Binary filament mask, for BCE and Dice |
| `instances` | (B, 512, 512) int32 | Instance ids local to the crop, for distance-transform or watershed targets |
| `name` | list[str] | Observation id |

**Validation batch** (from `val_dl`, batch size is always 1):

| Key | Shape / dtype | Meaning |
|---|---|---|
| `image` | (1, 1, 2048, 2048) float32 | Whole image |
| `mask` | (1, 1, 2048, 2048) float32 | Majority-vote foreground, for a validation *loss* only |
| `gt_sets` | (1, K, 2048, 2048) int32 | One instance map per annotator. K is 1, 2 or 3. **Score against these.** |
| `name` | list[str] | Observation id |

Things to know:
- **Preprocessing switches** are in the `preprocess` dict: `mask_disc`, `clahe`, `local_norm`.
  All are off by default so the baseline is plain. Person A's finding is that `mask_disc` is
  safe (no filament pixels lie outside the disc) and `local_norm` greatly amplifies noise.
  Treat both as ablations and log the result either way.
- **Normalisation is applied after augmentation**, so the brightness augmentation works on [0, 1]
  values. Don't normalise again in your model.
- `pin_memory=True` prints a warning on CPU or Apple silicon. It is harmless.
- On macOS, if `num_workers=4` hangs or errors, set `num_workers=0`.
- Validation reads full 2048×2048 images, so run it under `torch.no_grad()` and consider
  half precision or tiled inference if memory is tight.
- `FilamentDataset` draws random crops, so an epoch is not deterministic even with a fixed seed.
  Call `D.seed_everything(seed)` at the start of every run.

## 5. Scoring a validation prediction

Turn the probability map into instances with connected components, then score against
**each annotator** and add to the accumulator:

```python
import torch, numpy as np
import metrics as M

acc = M.MetricAccumulator()
model.eval()
with torch.no_grad():
    for batch in val_dl:
        prob = torch.sigmoid(model(batch["image"].to(device)))[0, 0].cpu().numpy()
        pred = M.label_components(prob > 0.5)            # 0 = background, 1..N = filaments
        for k, gt in enumerate(batch["gt_sets"][0].numpy()):
            acc.add_pair(pred, gt, name=f'{batch["name"][0]}_{k}')

print(acc.summary())          # PQ_dataset, PQ_mean, PQ_std, SQ, RQ, Dice, IoU, one_to_many, many_to_one
arrays = acc.filament_arrays()    # per-filament dice / iou, for the histograms
```

What `summary()` returns:
- **`PQ_dataset`** pools TP/FP/FN across all images (the Kirillov definition).
- **`PQ_mean`** and **`PQ_std`** are per image. We don't know which one Kaggle uses, so report both.
- **`one_to_many`** counts fragmented ground-truth filaments, and **`many_to_one`** counts merged predictions.
  Person C needs these before and after post-processing.

Other metric functions:
- `M.evaluate_observation(pred, gt_sets, reduce="mean" | "max")` returns PQ/SQ/RQ against all annotators of one image.
- `M.label_components(binary, connectivity=2)` gives 8-connected instances, which is the baseline conversion.

**Assumptions to confirm on the Kaggle Evaluation page** (say so on the slide):
1. The one-to-many and many-to-one definitions are ours: a prediction counts if at least 50% of it lies
   inside a ground-truth filament and it covers at least 10% of that filament.
2. We don't know how Kaggle combines several annotators, which is why both pooled and mean PQ are reported.
3. An image with no filament and no prediction scores PQ = 1.0.

If your local PQ and the Kaggle PQ differ a lot, check these first, then the RLE encoding, before blaming the model.

## 6. Rules you must not break

- **Never regenerate or edit `splits/split_v1.json`.** Every number in the project depends on it.
- **Never tune on the Kaggle leaderboard.** Tune on the validation split only.
- **Train on the segmentation masks only.** Do not use the `spine`, `bbox` or `category_id`
  fields. If you want a skeleton or centreline target, compute it with `skimage.morphology.skeletonize`.
- **Never feed the capture time or instrument code into the model.** They are only for splitting.
- **Never download the public MAGFiLO version that includes test annotations.**

## 7. Your tasks, in order

1. **Baseline (E0):** `smp.Unet("resnet34", encoder_weights="imagenet", in_channels=1, classes=1)`,
   BCE + Dice, AdamW, cosine schedule, mixed precision, 512 crops. Threshold at 0.5, then `M.label_components`.
2. **Score it** with section 5 and record local PQ. Then hand your weights or predictions to Person C
   for `submission.csv`, so the Kaggle score gets recorded next to the local one.
3. **Baseline checkpoint:** the plan wants a scored Kaggle submission early. Do this before any improvement.
4. **Ablations, one change at a time:** loss (Tversky, focal, boundary, clDice), architecture, resolution
   (crops vs downsampled), and the data switches (`mask_disc`, `clahe`, `local_norm`, `target: "consensus"`).
   Log every run, including negative results.
5. **Seeds:** the final configuration needs at least 3 different seeds, reported as mean ± std.
6. **Make `train.py` resumable.** Free GPU sessions end without warning, so save a checkpoint every epoch.
7. **Save the config next to every result.** Copy `configs/data_default.yaml` per experiment
   and change one line.

## 8. Experiment log

One row per experiment, filled in as it finishes: ID, change from the previous run, seeds, val PQ, SQ,
RQ, one-to-many, many-to-one, ms per image, Kaggle PQ. The plan's IDs run E0 (baseline) to E7 (compression).

## 9. Known limits of Person A's work

- `data.py` and `metrics.py` were tested on synthetic data, and the cache and split were run on the real
  data. **The PyTorch loaders have not been run on real data with a GPU**, so if `make_loaders` errors,
  tell Person A with the traceback and don't work around it silently.
- The solar-disc detector was checked visually on one real image and numerically on 40, not on all 1154.
- Overlaps between filaments inside one annotation set are rare (5 sets). The later annotation wins.
- `consensus_instances` merges touching filaments into one component. This is a deliberate simplification
  for the consensus ablation, and it makes consensus targets worse for instance separation.

## 10. Who to ask

- Data, split or metric questions: Person A.
- Post-processing, instance separation and compression: Person C. They need your predictions saved as
  probability maps or instance maps.