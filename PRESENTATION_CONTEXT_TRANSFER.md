# COS711 Assignment 2: full-context transfer file

**Purpose.** This file gives a teammate or another AI assistant everything needed to understand the project, present it, answer questions about it, and finish the submission. It focuses on what the presentation must cover, according to the official template, and on the final presentation that was built from it.

**Snapshot:** 1 November 2026, about 21:10. **The deadline is 1 November 2026 at 23:30 (ClickUP).**

| Item | Location |
|---|---|
| Final presentation (23 slides) | `C:\Users\denis\Downloads\COS711_A2_Final_Presentation_updated.pptx` |
| Team's previous version (18 slides, unchanged) | `C:\Users\denis\Downloads\COS711_A2_Final_Presentation.pptx` |
| Official template (9 slides) | `C:\Users\denis\Downloads\COS711_A2_Presentation_Template.pptx` (copy in [delivery/templates/](delivery/templates/)) |
| Assignment specification | [COS711-A2.pdf](COS711-A2.pdf) |
| Repository / branch | this folder; branch `feat/improvements` (nothing committed from this round yet) |
| Submitted-model CSV | [runs/person_c/final/submission.csv](runs/person_c/final/submission.csv) (Git-ignored) |

---

## 0. Status in one minute

- **Technical work is finished.** The final model is a U-Net with a ResNet34 encoder, compressed by 30% channel pruning and stored in FP16. With our cleanup step it reaches **local validation PQ 0.4031**. It is 22.9 MiB and takes 475 ms per image end to end.
- **Kaggle:** the team reported a public score of about **0.32**. This is lower than local validation (about 0.40); see [section 10](#10-the-validation-to-kaggle-gap).
- **The presentation is complete** apart from the Kaggle placeholders, which the team fills in itself (see [section 9](#9-remaining-actions-before-2330)).
- **Still to do:** fill the Kaggle placeholders, record the video, build the ZIP and upload it to ClickUP.

---

## 1. The assignment

**Task.** Build a deep-learning model that produces a **separate pixel mask for every individual solar filament** in GONG H-alpha images of the Sun. This is called *instance segmentation*. The work is for the Kaggle "Solar Filament Segmentation Challenge 2026", which uses the MAGFiLO dataset.

Submissions are scored with **Panoptic Quality (PQ)**. PQ penalises missed filaments, extra predictions, broken-up filaments and merged filaments. Efficiency matters too, and compression (pruning, distillation and similar) is recommended.

**Rules that matter for the presentation**

- Teams of 2–3, using Python, with any framework.
- Use only the Kaggle data. Using external labels for the test images is misconduct; we used none.
- Every choice must be justified by data or experiments, and hyperparameters must be tuned.
- PQ must be implemented locally, with a careful validation split. Annotation sets of one image must never be split across train and validation, and images close in time or from the same instrument are not independent.
- Report the model size, training time and end-to-end inference time, and experiment with compression.

**Deliverables (one ZIP on ClickUP)**

1. All code.
2. The exact `submission.csv` uploaded to Kaggle.
3. A video of **at most 10 minutes and 50 MB**. Each presenter's webcam (head and shoulders) must be in the **upper-right corner**, every member must appear, and each member's responsibility must be clear.
4. The PowerPoint slides used in the video.

**Marking (100 marks)**

| Category | Marks |
|---|---:|
| Format (presentation requirements) | 5 |
| Data Preparation | 10 |
| Experimental Setup and Methodology | 20 |
| Segmentation Results | 30 |
| Segmentation Enhancement (instance separation, post-processing, compression, …) | 30 |
| Conclusions | 5 |
| Penalty for missing code or submission file | −50% of total |
| Bonus: top 100 on the Kaggle leaderboard (include a screenshot) | up to +5 (capped at 100) |

---

## 2. Team and roles

| Role | Name | Student no. | Responsibility | Presents slides |
|---|---|---|---|---|
| Person A | **Alisha Perumal** | u22512285 | Prepared and analysed the data; built the image, annotation and validation tools | 6–9 |
| Person B | **Vansh Sood** | u23534402 | Trained the segmentation network; compared methods; repeated the selected recipe with fresh seeds | 10–14 (13 shared) |
| Person C | **Denis Woolley** | u23528860 | Improved individual-filament predictions (post-processing); compression; checked the final CSV; final improvement round | 13 (shared), 15–20 |

Everyone appears on slides 1–5. Slide 14 (Kinds of Errors) is labelled for Vansh to balance speaking time. It uses Person C's final model, so Denis can present it instead if the team prefers.

---

## 3. Key numbers cheat sheet

| What | Value |
|---|---|
| Labelled observations / annotation sets / marked filaments | 707 / 1,154 / 8,199 |
| Test images | 180 (unlabelled) |
| Images with 1 / 2 / 3 annotation sets | 411 / 145 / 151 (so 296 have 2–3) |
| Filaments per annotation set | mean 7.1 (sd 4.4), median 7, max 26; 0 empty sets |
| Filament pixel share | 0.36% (median 0.30%); about 278 background pixels per filament pixel |
| Instruments / years | 6 GONG sites (Bh, Ch, Lh, Mh, Th, Uh), 2011–2022 |
| Annotator vs annotator PQ | **0.34** (100 images, 404 ordered pairs); a reference point, not a ceiling |
| Overlapping-outline annotation sets | 5 (kept as drawn) |
| Split | 584 train / 122 validation / 1 excluded; validation has 200 annotation sets |
| Image size | 2048 × 2048, grayscale |
| Original 20-epoch baseline | PQ 0.3201 |
| Selected recipe ("no added transforms"), 3 fresh seeds | **0.3370 ± 0.0076** (seeds 43/44/45: 0.3364 / 0.3297 / 0.3448) |
| Representative seed-45 model | 0.3448 |
| Seed 45 + our cleanup | **0.4034** (+0.0585; 95% CI +0.0486 to +0.0679) |
| Final compressed model + cleanup | **0.4031** (0.3288 without cleanup) |
| Final model size | 11.93 M parameters, 22.9 MiB (teacher: 24.43 M, 93.4 MiB; about 75% smaller) |
| Final end-to-end speed | 475 ms/image over 180 test images on an RTX 3070 (read → CSV, model loading excluded) |
| Training time | Teacher: 12.4 min of epoch loops (RTX 5060 Ti) + pruning fine-tune 4.0 min (RTX 3070) |
| Final CSV | 1,421 filaments; 177 of 180 images have rows (3 zero-detection); SHA-256 `1b08bcb0…5e9fb2f` |
| Kaggle public score | about **0.32** (team-reported; exact value, rank, date and screenshot still to be added) |
| Best ensemble tried (not adopted) | 0.4143 (teacher + distilled ResNet18); 148.1 MiB, 556 ms/image |

---

## 4. Template requirements and where the final deck covers them

Slide numbers refer to the **updated 23-slide deck**. ✅ means covered; ⏳ means a Kaggle placeholder the team fills in.

### Template slide 1: Title
| Template asks | Status | Slides |
|---|---|---|
| "COS711 Assignment 2: Segmenting solar filaments with deep learning" (the template's "Assignment 3" is a typo) | ✅ | 1–3 |
| Team member names and student numbers | ✅ | 1–3 |

### Template slide 2: Team and Responsibilities
| Template asks | Status | Slides |
|---|---|---|
| Name, student number and responsibility of each member | ✅ | 4–5 |
| Every member featured in the video | Must be shown in the recording | presenter label on every content slide |

### Template slide 3: Dataset Preparation
| Template asks | Status | Slides |
|---|---|---|
| What we discovered and how we prepared the data | ✅ | 6–9 |
| Preprocessing and scaling (resolution, normalisation) | ✅ | 8 (2048² images, 512² patches, standardisation), 12 (preprocessing screen), 18 (preprocessing retrains) |
| Augmentation | ✅ | 8 (list of transforms; turned off: 0.3201 → 0.3381), 12 |
| Noisy samples and class imbalance | ✅ | 6–7 (0.36% imbalance), 9 (noisy, faint and disagreeing labels; annotator PQ 0.34) |
| Training/validation split | ✅ | 8 |

### Template slide 4: Experimental Setup and Methodology
| Template asks | Status | Slides |
|---|---|---|
| Libraries, tools and hardware | ✅ | 10 |
| Problem formulation | ✅ | 10 (pixel score > 0.5 → connected regions = filaments) |
| Models, losses and training algorithm, **and why** | ✅ | 10 (U-Net skip links keep thin detail; pretraining because there are only 584 images; Dice because of the imbalance) |
| How hyperparameters were tuned | ✅ | 11 (process), 12 (results of the 11 recipes) |
| How results were evaluated | ✅ | 11 (PQ formula, IoU > 0.5, pooled over 200 sets), 17 (bootstrap and split-half) |

### Template slide 5: Results
| Template asks | Status | Slides |
|---|---|---|
| How well the model did | ✅ | 13 |
| Kaggle leaderboard screenshot | ⏳ | 13 (placeholder) |
| PQ locally and on Kaggle, with means and SDs over runs | ✅ local (0.3370 ± 0.0076); ⏳ Kaggle | 13 |
| Comparison between approaches | ✅ | 12, 13, 17, 18, 19 |
| Kinds of errors (missed, fine-scale, noise, fragmentation, merging) | ✅ | 14 (also 17) |

### Template slide 6: Example Predictions
| Template asks | Status | Slides |
|---|---|---|
| Test inputs next to predicted masks | ✅ | 16 |
| Do the predictions seem viable? | ✅ | 16 |
| Where do they fail, and why? | ✅ | 14, 15, 16 |
| Best vs worst cases | ✅ | 15 |

### Template slide 7: Segmentation Enhancement
| Template asks | Status | Slides |
|---|---|---|
| What we did beyond a standard pipeline, and how much each idea helped | ✅ | 17, 18, 19 |
| Instance separation / keeping filaments intact / fine-scale structure | ✅ | 17 (keeping filaments intact by bridging fragments), 18 (watershed separation: worse; fine-scale: honestly no dedicated method) |
| Quantify each idea | ✅ | 17, 18, 19 |
| Model size, training and inference time, compression | ✅ | 19 |

### Template slide 8: Conclusions
| Template asks | Status | Slides |
|---|---|---|
| Main take-aways, biggest roadblocks, what we would do differently | ✅ | 20 |

### Template slide 9: Bibliography
| Template asks | Status | Slides |
|---|---|---|
| Papers and websites used | ✅ | 21–23 (18 references) |

---

## 5. The final deck, slide by slide

Design: dark theme, Century Gothic, teal accents (#3BC5C7 / #007B7D), orange (#E8916F) for warnings, a University of Pretoria header, Morph transitions, and a presenter label at bottom-left. Slides 1–3, 4–5 and 6–7 are **Morph animation pairs**: the same content appears in a different layout, so narrate once per pair.

Each slide's **narration** below is copied from its PowerPoint speaker notes. All the narration together totals about **1,130 words, roughly 7.5–8 minutes**.

### Slides 1–3: Title (all members)
- "COS711 Assignment 2: Segmenting solar filaments with deep learning". Slide 2 also has the team tagline "VAD is RAD".
- Shows Denis Woolley [u23528860], Vansh Sood [u23534402] and Alisha Perumal [u22512285].
- No speaker notes. Suggested line: *"We built a deep-learning system that marks the exact pixels of each individual solar filament in telescope images of the Sun."*

### Slides 4–5: Team and Responsibilities (all members, about 25 s)
- **On the slide:**
  - Alisha prepared and analysed the data and built the image, annotation and validation tools.
  - Vansh trained the segmentation network, compared methods and repeated the selected recipe.
  - Denis improved individual-filament predictions, compared compression options and checked the final CSV.
  - Goal: give every filament its own pixel mask, not just detect that filaments are present.
- **Narration:** Each member introduces themselves and their responsibility. A mask is a pixel-level marking of an object. Instance segmentation means producing a separate mask for each individual object.

### Slides 6–7: Dataset Preparation (Alisha, about 40 s)
- **On the slide:**
  - Stat cards: 707 labelled observations · 1,154 annotation sets · 8,199 marked filaments · 180 unlabelled test images (six GONG instruments, 2011–2022).
  - Filaments occupy only **0.36% of pixels** (about 278 background pixels per filament pixel).
  - Histograms of the filament fraction and mean filament area per annotation set.
  - Challenges: the imbalance; faint thin structures; image noise; annotator disagreement.
- **Narration:** We analysed the supplied Kaggle images and labels. An annotation is a human marking. Multiple annotation sets can describe the same observation. Filaments cover less than one percent of pixels; this is class imbalance, meaning one category is much more common than the other. A model predicting only background could get most pixels right while finding no filaments. We therefore evaluate individual objects and shapes. We did not obtain external labels for the test images.
- **Evidence:** [results/eda_summary.json](results/eda_summary.json), [figures/eda_imbalance.png](figures/eda_imbalance.png)

### Slide 8: Dataset Preparation: Our Choices (Alisha, about 35 s)
- **On the slide:**
  - Split bar: 584 training / 122 validation / 1 excluded. All labels for one image stay together. Includes a split-timeline figure.
  - 14-day time blocks and a 1-day gap reduce near-duplicate leakage; the validation instrument mix follows the test set.
  - Full 2048 × 2048 images: train on random 512 × 512 patches (70% centred on filaments); predict whole images.
  - Scaling: pixels standardised with the training-set mean and spread.
  - Augmentation (flips, rotations, scaling, brightness, noise, blur) scored lower, so it was switched off: 0.3201 → 0.3381.
- **Narration:** All annotation sets of one image stay on the same side of the split, and we split by 14-day blocks with a one-day gap so near-identical images do not leak; validation matches the test instrument mix. Images are 2048 by 2048 pixels: we train on random 512-pixel patches, mostly centred on filaments, and predict whole images. Pixels are standardised with training-set statistics. Extra augmentation, meaning flips, rotations, scaling, brightness, noise and blur, scored lower, so we switched it off.
- **Evidence:** [splits/split_v1.json](splits/split_v1.json), [configs/person_c/seed45_selected.yaml](configs/person_c/seed45_selected.yaml), [results/person_b_study/confirmation.csv](results/person_b_study/confirmation.csv)

### Slide 9: Dataset Preparation: Noisy Labels (Alisha, about 30 s), new
- **On the slide:**
  - Three image panels:
    - Noisy image: 20160204133734Ch, the most image noise.
    - Faint filament: 20200116073250Uh, the lowest filament contrast.
    - Annotators disagree: 20210626133950Bh, with two annotators' outlines in teal and orange.
  - Stat cards:
    - **0.34**: PQ when one annotator is scored against another (100 images).
    - **296 of 707** images have 2–3 different annotation sets.
    - **5** sets have overlapping outlines, kept as drawn; 0 sets are empty.
  - How we handled it: every annotation set is its own training target, and scoring uses all 200 validation sets. Voted labels were tested (0.272 vs 0.255 in the 6-epoch screen) but not promoted.
- **Narration:** Labels are noisy. These are the noisiest image, the faintest filament and the biggest annotator disagreement from our data analysis. One annotator scored against another gets a PQ of only 0.34, a reference point rather than a ceiling. 296 of 707 images have two or three annotation sets. We kept every annotation set as its own training target and score against all 200 validation sets; voted labels were tested but not promoted.
- **Evidence:** `results/eda_summary.json` (annotator_agreement, sets_per_observation, overlap_sets, noisy_samples); [results/person_b_study/screening.csv](results/person_b_study/screening.csv)

### Slide 10: Experimental Setup and Methodology (Vansh, about 35 s)
- **On the slide:**
  - Flow: grayscale image in → U-Net with a ResNet34 encoder (ImageNet-pretrained because there are only 584 training images; skip links keep thin detail) → filament score per pixel (> 0.5 → connected regions = filaments).
  - Training loss: binary cross entropy + Dice; Dice copes with filaments covering only 0.36% of pixels.
  - Settings: AdamW · learning rate 0.0003 · cosine schedule · batch 4 · mixed precision · 20-epoch full runs.
  - Hardware: Vansh RTX 5060 Ti; Denis RTX 3070 (8 GB). Mixed precision saves memory; ONNX Runtime and torch-pruning were used for compression.
  - Software: Python, PyTorch, segmentation_models_pytorch, OpenCV, Albumentations, pycocotools.
- **Narration:** We treat it as pixel classification: every pixel gets a filament score, we keep pixels above 0.5, and each connected region becomes one filament. U-Net's skip connections keep the fine detail thin filaments need, and the ResNet34 encoder starts from ImageNet because we only have 584 training images. Dice loss copes with filaments being only 0.36 percent of pixels. AdamW with a cosine schedule trains it, and mixed precision saves memory.
- **Evidence:** [results/person_b_study/methodology.md](results/person_b_study/methodology.md), [models.py](models.py), [losses.py](losses.py)

### Slide 11: Methodology: Tuning and Evaluation (Vansh, about 45 s)
- **On the slide:**
  - Funnel: 11 recipes screened for 6 epochs → the strongest two retrained for 20 epochs alongside the baseline → three fresh random seeds after selection. In total, 17 runs and 186 epochs.
  - Compared: learning rates, image strategy, preprocessing, augmentation, voted labels, loss, U-Net++.
  - Panoptic Quality: PQ = sum of matched IoUs / (matched + ½ extra + ½ missed). A match requires IoU > 0.5, and counts are pooled across all 200 validation annotation sets.
- **Narration:** A random seed controls random choices during training. We separated recipe selection from three fresh runs. IoU is shared mask area divided by the area covered by either mask. PQ equals the sum of matched IoUs divided by matched objects plus half the extra and missed objects. Our evaluator follows the organiser's reference and keeps overlapping annotation layers. The fixed sample budgets improve comparability, but the 1024 whole-image recipe also changes field of view and computing cost. Voted labels change observation weighting. A short screen may favour fast learners; it does not prove every rejected recipe would lose after longer training.
- **Evidence:** [results/person_b_study/protocol.json](results/person_b_study/protocol.json), [results/evaluation_reference.json](results/evaluation_reference.json)

### Slide 12: Methodology: Tuning Results (Vansh, about 30 s), new
- **On the slide:** bar chart of the 6-epoch screen (seed 42, same training budget for each recipe), validation PQ:

  | Recipe | PQ |
  |---|---:|
  | **Whole image shrunk to 1024 px** (top two) | **0.3104** |
  | **No added transformations** (top two) | **0.2856** |
  | U-Net++ architecture | 0.2791 |
  | Mask outside the Sun | 0.2783 |
  | Contrast boost (CLAHE) | 0.2768 |
  | Voted labels | 0.2723 |
  | Local brightness normalisation | 0.2597 |
  | Baseline (learning rate 0.0003) | 0.2554 |
  | Learning rate 0.0001 | 0.2465 |
  | Learning rate 0.001 | 0.2424 |
  | Tversky loss | 0.2416 |

  - Cards: the top two went to 20-epoch runs and nearly tied (0.3375 vs 0.3381); a learning rate of 0.0003 beat 0.0001 and 0.001; preprocessing helped short runs slightly (+0.004 to +0.023) but was not in the top two.
  - Caveat: one seed and short runs; gaps under about 0.01 are within run-to-run noise.
- **Narration:** These are the six-epoch screening runs, all with seed 42 and the same training budget. The top two, 1024-pixel whole images and no extra transformations, got full 20-epoch runs and nearly tied. A learning rate of 0.0003 was best. Preprocessing helped short runs slightly but was not top two, and the Tversky loss was lowest. These are single short runs, so small gaps should not be over-read.
- **Evidence:** [results/person_b_study/screening.csv](results/person_b_study/screening.csv), [results/person_b_study/figures/screening.png](results/person_b_study/figures/screening.png)

### Slide 13: Results (Vansh and Denis, about 55 s)
- **On the slide:** validation PQ bars:

  | Approach | Validation PQ |
  |---|---:|
  | Original baseline (20 epochs) | 0.3201 |
  | 1024 whole images (selection seed 42) | 0.3375 |
  | No added transforms (selection seed 42) | 0.3381 |
  | Selected recipe: 3 fresh runs, mean ± SD | 0.3370 ± 0.0076 |
  | Representative seed 45 | 0.3448 |
  | Seed 45 + output cleanup | 0.4034 |
  | Final compressed model + cleanup | 0.4031 |

  - Caption: the two full-budget alternatives nearly tied (+0.00063 PQ); the final compressed result is one run, not a three-run mean.
  - ⏳ **Kaggle card:** "Kaggle: add result / Public PQ: [ADD] / Rank / date: [ADD]" and "[Insert leaderboard screenshot here]".
- **Narration:** The original full-budget baseline scored 0.3201. The two alternatives nearly tied; we do not claim statistical superiority from that difference. Across fresh seeds 43, 44 and 45, the selected recipe averaged 0.3370 with sample standard deviation 0.0076. Standard deviation describes variation across runs. Seed 45 is the representative checkpoint, not the mean. Denis's cleanup raises its score to 0.4034, and the final smaller model scores 0.4031. *(⏳ Notes still say "Fill the Kaggle score…": replace this with the real score and one sentence on the 0.40 vs 0.32 gap.)* Local validation and Kaggle evaluate different images.
- **Evidence:** [results/person_b_study/fresh_seed_summary.csv](results/person_b_study/fresh_seed_summary.csv), [results/person_c/summary.json](results/person_c/summary.json), [results/person_c/final_submission.json](results/person_c/final_submission.json)

### Slide 14: Results: Kinds of Errors (Vansh, about 35 s), new
- **On the slide:** final model with cleanup, 122 validation images, counts pooled over 200 annotation sets:

  | The 1,290 marked filaments | Count | The 1,406 predicted filaments | Count |
  |---|---:|---|---:|
  | Found (IoU > 0.5) | 817 | Correct | 817 |
  | Overlapped, but outline too different | 273 | Touch a filament, wrong shape or split | 295 |
  | Completely missed | 200 | False alarm in empty background | 294 |

  - **Small = missed:** completely missed filaments have a median area of 466 px vs 1,356 px for found ones; 53% are under 500 px (found: 9%).
  - **Split and merged:** 42 filaments broken into pieces; 35 predictions covering two or more filaments.
  - **Shape × detection:** PQ 0.40 = outline quality (SQ) 0.67 × detection quality (RQ) 0.61.
- **Narration:** These are the final model's validation errors after cleanup. Of 1,290 marked filaments, 817 were found, 273 were overlapped with the wrong outline, and 200 were missed completely. The completely missed ones are small, median 466 pixels against 1,356 for found ones, so fine-scale structure is our main weakness. The wrong predictions split evenly between wrong shapes and false alarms in empty background. PQ is outline quality 0.67 times detection quality 0.61.
- **Evidence:** `runs/person_c/compression/bench_pruned30_fp16/validation/` (summary.json, filaments.csv, overlap_pairs.csv, annotation_sets.csv)

### Slide 15: Results: Best and Worst Cases (Denis, about 35 s), changed
- **On the slide:** for each case, a 3-panel row: reference (annotation set 1) | before cleanup | final model.
  - **Best: 20160928143554Bh, PQ 0.66.** Clear, elongated filaments; all 4 found in both annotation sets, with no extra predictions. Cleanup changed nothing (0.66 before and after).
  - **Worst: 20210222222850Bh, PQ 0.00.** Tiny, faint filaments (225–396 px) beside bright active regions. The model's scores there are about 0, so none were found. Its one prediction elsewhere in the image was wrong, and cleanup removed it. The row is zoomed on one missed filament.
  - Selection rule: best = highest image PQ with at least 3 filaments per set (single-filament images reach 0.77 but are too easy); worst = lowest PQ, breaking ties by the most filaments. Colours are object IDs; red marks pixels removed by cleanup.
- **Narration:** Best and worst validation images for the final model. Best: four clear, elongated filaments, all found in both annotation sets with no extras, PQ 0.66. We skipped single-filament images because they are too easy. Worst: tiny, faint filaments beside bright active regions. The model's scores there are near zero, so it found nothing, and three of them are below our 320-pixel cleanup limit anyway.
- **Evidence:** `runs/person_c/compression/bench_pruned30_fp16/validation/observations.csv`, `runs/improvements/deck/final_model_before_cleanup/observations.csv`, [delivery/final_deck_figures.py](delivery/final_deck_figures.py)

### Slide 16: Results: Final Test Predictions (Denis, about 40 s)
- **On the slide:**
  - Test images 20120622063154Uh (8 filaments) and 20131027185354Bh (12 filaments): original input beside the masks decoded from the final CSV.
  - Two fixed scenes; these are not best/worst accuracy rankings.
  - Dark structures look plausible; faint edges and nearby filaments remain difficult.
  - ⏳ "Unlabelled: Kaggle score pending." Update this with the Kaggle score.
- **Narration:** Left: original grayscale input. Right: the final pruned FP16 model's submitted masks overlaid in colour. They were chosen to illustrate different scenes, not ranked by test accuracy. Masks are decoded directly from the hash-checked final CSV, so these are the submitted predictions. Some elongated dark structures are captured, but visual plausibility cannot establish correctness. Faint edges, background texture and adjacent structures remain risks; no test labels are used.
- **Evidence:** `runs/person_c/final/submission.csv`, [results/person_c/final_submission.json](results/person_c/final_submission.json)

### Slide 17: Segmentation Enhancement (Denis, about 45 s)
- **On the slide:**
  - Rule: (1) give fragments up to about 16 px apart one shared ID (bridge radius 8), **without adding pixels**; (2) then remove groups under 320 px.
  - PQ gain **+0.0585** (95% bootstrap interval +0.0486 to +0.0679); held-out half gains +0.0592 and +0.0557.
  - Bars, on the same saved scores: baseline 0.3448 · remove small regions 0.3948 · group fragments 0.3782 · **both rules 0.4034**.
  - Before → after: extra predictions 1,007 → 510; split filaments 84 → 50. Cost: 17 fewer matches, and merged filaments 20 → 25.
  - Checks: the offline scorer first reproduced Vansh's result exactly; 55 + 50 settings were tested; a more complex rule was only +0.0005 better; flip averaging was slower and not adopted.
- **Narration:** Post-processing changes the output after the network predicts. Using saved predictions, we first reproduced Vansh's score exactly. About half of the extra regions were under 320 pixels, against 4 percent of correct ones. So we give fragments within about 16 pixels the same ID, without adding pixels, then remove groups under 320 pixels. That roughly halves extra detections and raises PQ by 0.059; the bootstrap interval and both held-out halves agree. The cost is 17 fewer matches and a few more merges.
- **Evidence:** [results/person_c/postprocess_stage1.csv](results/person_c/postprocess_stage1.csv), [results/person_c/postprocess_stage2.csv](results/person_c/postprocess_stage2.csv), [results/person_c/person_c_results.md](results/person_c/person_c_results.md)

### Slide 18: Segmentation Enhancement: What Else We Tried (Denis, about 35 s), new
- **On the slide:**

  | Post-processing idea | Change in PQ |
  |---|---:|
  | Keep weak pixels next to strong ones (hysteresis) | +0.023 |
  | Close small gaps (morphological closing) | +0.014 |
  | Change only the 0.5 pixel cutoff | ±0.000 |
  | Fill holes inside regions | ±0.000 |
  | Stricter 4-neighbour connections | −0.002 |
  | Split touching filaments (watershed) | −0.002 to −0.010 |
  | Average flipped copies (TTA), 3.8× slower | −0.004 |

  | Retrained or combined model | Validation PQ |
  |---|---:|
  | Plain images (selected recipe, seed 45) | 0.4034 |
  | Contrast boost (CLAHE) + outside-Sun mask (seed 46) | 0.3996 |
  | Local brightness normalisation + mask (seed 47) | 0.3808 |
  | Average of 2 models (+ distilled ResNet18) | 0.4143 |
  | Average of 4 models (including both above) | 0.4080 |

  - Hysteresis and closing helped on their own but added nothing meaningful on top of the selected pair.
  - **Not adopted:** the 2-model gain (+0.011; 95% interval −0.001 to +0.025) is uncertain and needs a 6.5× larger file and 17% more time per image. The preprocessing models are single runs with different seeds.
  - Fine-scale structure: we had no dedicated method; the Tversky loss, which weights missed pixels more, was lowest in screening (0.242).
- **Narration:** Other ideas did not make the final pipeline. Hysteresis and closing helped alone but added nothing on top of our two rules. Cutoff changes and hole filling did nothing; watershed splitting and averaging flipped predictions made PQ worse. Retraining with contrast enhancement or brightness normalisation did not beat plain images. Averaging two models reached 0.414, but the gain is uncertain and the file is 6.5 times larger, so we kept the compressed model.
- **Evidence:** [results/improvements/README.md](results/improvements/README.md), [results/improvements/score_with_localnorm.json](results/improvements/score_with_localnorm.json)

### Slide 19: Segmentation Enhancement: Compression (Denis, about 45 s)
- **On the slide:** GPU variants (PQ with cleanup / MiB / ms per image end to end, RTX 3070):

  | Variant | PQ | MiB | ms/image |
  |---|---:|---:|---:|
  | Teacher, mixed precision | 0.4034 | 93.4 | 456 |
  | Teacher, FP16 storage | 0.4033 | 46.7 | 445 |
  | Distilled ResNet18 | 0.4012 | 54.7 | 489 |
  | Label-only control (ResNet18) | 0.3976 | 54.7 | 461 |
  | 50% channel pruning | 0.3652 | 23.5 | 494 |
  | **30% pruning + FP16 (FINAL)** | **0.4031** | **22.9** | **475** |

  - Definitions: pruning removes internal feature channels; FP16 stores learned numbers with fewer bits; distillation teaches a smaller model using the teacher's outputs.
  - Rule: choose the smallest GPU file within 0.002 PQ of the teacher; speed breaks ties.
  - Final model: 11.93 M parameters; 22.9 MiB (about 75% smaller). Training: 12.4 min for the teacher on Vansh's GPU + 4.0 min pruning fine-tune on Denis's GPU.
  - 475 ms/image over 180 images, read-to-CSV with model loading excluded. A smaller model did not mean faster.
  - CPU INT8: PQ 0.4068; 23.6 MiB; 1,158 ms/image (on 20 images). A separate CPU deployment.
- **Narration:** Pruning removes internal feature channels, then we fine-tune what is left. FP16 stores weights with 16-bit numbers. Distillation trains a smaller student from the labels plus the teacher's predictions, and a label-only control shows its effect. 50 percent pruning lost quality, so we added a milder 30 percent version. Our rule picks the smallest GPU model within 0.002 PQ of the teacher: 30 percent pruning plus FP16, 22.9 megabytes instead of 93.4, at 475 milliseconds per image end to end on an RTX 3070, including reading, prediction, cleanup and writing the CSV. The CPU-only INT8 version is a separate result.
- **Evidence:** [results/person_c/compression.csv](results/person_c/compression.csv), [results/person_c/final_selection_rule.json](results/person_c/final_selection_rule.json), [results/person_c/final_submission_check.json](results/person_c/final_submission_check.json)

### Slide 20: Conclusions (Denis and all, about 30 s), changed
- **On the slide:**
  - **Main gain:** better individual-object cleanup, not a more complicated network. Final local PQ **0.4031**.
  - **Compression** reduced the file by about 75% while keeping nearly the same PQ; it was not the fastest GPU option.
  - **Roadblocks:** tiny, faint filaments; labels that disagree (annotators: PQ 0.34); and local validation (about 0.40) scoring higher than Kaggle.
  - **Do differently:** submit to Kaggle early to check our validation set; repeat final-model runs; try a loss for thin, connected shapes. Contrast enhancement was tested: no gain.
- **Narration:** Our main lesson: the biggest gain came from understanding object-level errors and cleaning up predictions, not from a bigger network. Compression cut the file by about 75 percent with almost no loss. Roadblocks were tiny faint filaments, annotators who disagree, and a validation score higher than our Kaggle score. Next time we would submit to Kaggle earlier, repeat final-model runs and try losses designed for thin, connected shapes.

### Slides 21–23: Bibliography (reference only, no narration)
1. Ahmadzadeh et al. (2024). *A dataset of manually annotated filaments from H-alpha observations* (MAGFiLO), Scientific Data 11:1031. https://doi.org/10.1038/s41597-024-03876-y
2. Kirillov et al. (2019). *Panoptic Segmentation.* https://arxiv.org/abs/1801.00868
3. Ahmadzadeh (2026). *Self Evaluation Notebook*, version 6 (organiser's scorer). https://www.kaggle.com/code/azimahmadzadeh/self-evaluation-notebook
4. Ronneberger, Fischer & Brox (2015). *U-Net.* https://arxiv.org/abs/1505.04597
5. He et al. (2016). *Deep Residual Learning for Image Recognition.* https://arxiv.org/abs/1512.03385
6. Loshchilov & Hutter (2019). *Decoupled Weight Decay Regularization* (AdamW). https://arxiv.org/abs/1711.05101
7. Milletari, Navab & Ahmadi (2016). *V-Net* (Dice loss). https://arxiv.org/abs/1606.04797
8. Salehi, Erdogmus & Gholipour (2017). *Tversky loss.* https://arxiv.org/abs/1706.05721
9. Zhou et al. (2018). *UNet++.* https://arxiv.org/abs/1807.10165
10. segmentation_models_pytorch 0.6.0. https://github.com/qubvel-org/segmentation_models.pytorch
11. PyTorch 2.14 documentation, CosineAnnealingLR.
12. Compression: Hinton et al. (2015), *Distilling the Knowledge in a Neural Network*; Fang et al. (2023), *DepGraph*; ONNX Runtime quantisation documentation.
13. Kaggle (2026). Solar Filament Segmentation Challenge 2026. https://www.kaggle.com/competitions/filament-segmentation-2026
14. Zuiderveld (1994). *Contrast Limited Adaptive Histogram Equalization.* Graphics Gems IV.
15. Buslaev et al. (2020). *Albumentations.* https://doi.org/10.3390/info11020125
16. Paszke et al. (2019). *PyTorch.* https://arxiv.org/abs/1912.01703
17. Lin et al. (2014). *Microsoft COCO* (RLE masks, pycocotools). https://arxiv.org/abs/1405.0312
18. Bradski (2000). *The OpenCV Library.* https://opencv.org

---

## 6. Background detail for questions

### 6.1 Data and split (Person A)
- Each **observation** is one solar image. One observation can have 1–3 **annotation sets**, each drawn by a different annotator. Training uses every annotation set as its own target, so images with several annotators carry more weight. Scoring compares predictions against every annotation set separately and pools the counts, following the organiser's notebook.
- The split groups observations into 14-day time blocks, with a 1-day embargo between the train and validation sides. One observation fell inside the gap and was dropped.
- The validation instrument mix was chosen to resemble the test filenames: validation Lh/Bh/Ch 18.9% each, Mh/Uh 14.8%, Th 13.9%; test Lh 18.9%, Ch/Bh 18.3%, Th/Uh 15.0%, Mh 14.4%.
- Instruments appear on both sides, so this is **not** an unseen-instrument test.
- Normalisation statistics come from at most 200 training images only (mean 0.327, SD 0.223 for plain images).

### 6.2 Model and training study (Person B)
- **Model:** U-Net with a ResNet34 encoder, ImageNet initialisation, 1 input channel, 1 output logit per pixel; 24,430,097 parameters; 93.36 MiB.
- **Loss:** BCE + soft Dice (weights 1 : 1).
- **Optimiser:** AdamW, learning rate 3e-4, weight decay 1e-4, cosine schedule, gradient clipping at 1.0, mixed precision, batch 4.
- **Epochs:** one "epoch" means **960 sampled patches**, not one pass over all images.
- **Selected recipe:** "no added transforms". It still uses random foreground-biased 512 px crops; only the flips, rotations, affine transforms, brightness/contrast, noise and blur were turned off.
- **Fresh seeds** 43 / 44 / 45: 0.33636 / 0.32969 / 0.34481, giving a mean of 0.33695 with sample SD 0.00758. Seed 42 was used only for selection, so it is excluded from the mean.
- **Person B's original pipeline:** 536 ms/image on the RTX 5060 Ti; that CSV had 1,831 instances.

### 6.3 Post-processing study (Person C)
- Person B saved the network's per-pixel scores for all 122 validation images. Person C re-scored them offline with the same scorer, after first reproducing B's PQ to 1e-12.
- Before cleanup, the extra predictions outnumbered misses 2 : 1 (1,007 vs 507). Of the unmatched predicted regions, 48% were under 320 px, compared with 4% of matched regions and 5% of human-drawn filaments.
- **Stage 1** tested 55 settings, changing one thing at a time. **Stage 2** tested 50 combinations, declared after stage 1.
- The best rule overall (cutoff 0.4 + seed threshold 0.9 + min area 320 + bridge 8) reached 0.4039, only +0.0005 above the simpler pair, so the simpler rule was kept.
- The same simpler rule was selected when tuning on half A of the validation time blocks. With this rule, PQ is flat for cutoffs from 0.35 to 0.55.

### 6.4 Compression (Person C)
- **Distillation:** a ResNet18 student trained for 20 epochs on 0.5 × label loss + 0.5 × BCE against the teacher's probabilities. A label-only control used the identical code path.
  - With cleanup, the distilled student scored 0.4012 vs 0.3976 for the control.
  - **Without cleanup the order reverses** (0.3279 vs 0.3368). This is single-run evidence, so do not overclaim.
- **Pruning:** L1-magnitude structured channel pruning (torch-pruning) applied uniformly per layer, keeping the output head, then 10 fine-tuning epochs.
  - 50% pruning (6.11 M parameters) predicted nothing for 9 epochs and recovered to 0.3652.
  - The milder 30% follow-up (11.93 M parameters) reached 0.4028, or 0.4031 once stored in FP16.
- **Selection rule:** declared in [results/person_c/final_selection_rule.json](results/person_c/final_selection_rule.json) **before** the benchmarks. The 30% follow-ups were added later and judged by the unchanged rule.
- **CPU results** (first 20 test images only; compare these only with each other):

  | CPU variant | ms/image | Size |
  |---|---:|---:|
  | PyTorch FP32 | 2,463 | 93.4 MiB |
  | ONNX FP32 | 1,691 | 93.2 MiB |
  | ONNX INT8 (PQ 0.4068, calibrated on 8 training images) | 1,158 | 23.6 MiB |

- Network inference itself takes only about 80–130 ms per validation image. Most of the 475 ms is reading the image, post-processing and RLE encoding, which is why a smaller model is not proportionally faster.

### 6.5 Improvement round after Kaggle 0.32 (Person C, branch `feat/improvements`)
- Two new 20-epoch models were trained with preprocessing applied identically at training and prediction time:
  - CLAHE + outside-disc mask (seed 46): **0.3996**.
  - Local normalisation + outside-disc mask (seed 47): **0.3808**.
- Neither beat plain images (0.4034); each is a single run with a different seed.
- **Ensembles** (averaging the probability maps of several models), scored with paired bootstrap and split-half checks:
  - Teacher + distilled ResNet18: **0.4143** (+0.0112 vs 0.4031; CI −0.0011 to +0.0245). Chosen on validation half B, it gained +0.0150 on half A; chosen on half A, a near-identical setting lost 0.0027 on half B.
  - Four models: 0.4080.
- Two ensemble CSVs exist but are **not** the presented final model (Kaggle scores unknown):

  | CSV | Filaments | Size | ms/image | Parameters |
  |---|---:|---:|---:|---:|
  | `runs/improvements/submissions/A_teacher_kd18/submission.csv` | 1,286 | 148.1 MiB | 556 | 38.75 M |
  | `runs/improvements/submissions/B_four_model_preproc/submission.csv` | 1,418 | 334.8 MiB | 860 | 87.61 M |

- Full write-up: [results/improvements/README.md](results/improvements/README.md).

### 6.6 Final submission checks
- Every one of the 1,421 RLE masks was decoded independently with pycocotools.
- IDs are unique and consecutive per image (`<observation>_1`, `_2`, …); all masks are non-empty and 2048 × 2048; no instances overlap; the smallest mask is 321 px.
- The three zero-detection images are 20180507140210Th, 20200310113230Ch and 20211214174010Mh.
- Receipt: [results/person_c/final_submission_check.json](results/person_c/final_submission_check.json).

---

## 7. Likely questions and short answers

- **Why is PQ only about 0.40?** Even two human annotators score only 0.34 against each other, so the labels are noisy. Filaments are thin and some are tiny (the completely missed ones have a median of 466 px), and strict IoU > 0.5 matching punishes outline differences.
- **Why not pixel accuracy?** About 99.6% of pixels are background, so a model predicting nothing would look excellent. PQ measures individual objects instead.
- **Why switch augmentation off?** It scored lower in the 6-epoch screen and at 20 epochs (0.3381 vs 0.3201). The random filament-centred crops already provide variety.
- **Why not the 0.4143 ensemble?** Its gain was uncertain (the interval includes 0), and it needs a 6.5× larger file and 17% more time per image. The assignment rewards compression.
- **Is the cleanup overfitted to validation?** It was checked with a paired bootstrap (+0.049 to +0.068) and with split-half tuning (held-out gains +0.059 and +0.056), but the absolute validation level may still be optimistic.
- **Did preprocessing help?** Slightly in 6-epoch screens (+0.004 to +0.023), but not in full 20-epoch runs (0.3996 and 0.3808 vs 0.4034).
- **What tools did what?** PyTorch and segmentation_models_pytorch handled the model. OpenCV read and resized images and handled preprocessing and morphology. Albumentations handled augmentation. pycocotools encoded and decoded the Kaggle RLE masks. torch-pruning and ONNX Runtime handled compression.

---

## 8. Glossary

| Term | Meaning |
|---|---|
| Observation | One solar image (identified by timestamp + instrument, e.g. `20160928143554Bh`) |
| Annotation set | One annotator's complete set of filament outlines for one observation |
| Mask / instance segmentation | Pixel marking of an object / a separate mask for every individual object |
| RLE | Run-length encoding: compact text form of a mask used in the Kaggle CSV |
| IoU | Shared area ÷ combined area of a predicted and a true mask |
| TP / FP / FN | Matched filament / extra prediction / missed filament |
| PQ = SQ × RQ | Panoptic Quality = segmentation quality (mean IoU of matches) × recognition quality (detection F1) |
| One-to-many / many-to-one | One filament split into several predictions / one prediction covering several filaments |
| U-Net, encoder, ResNet34 | Segmentation network with skip connections; its feature-extracting part; a 34-layer residual network |
| ImageNet pretraining | Starting from weights learned on a large photo dataset |
| BCE / Dice / Tversky | Per-pixel error / overlap-based loss / overlap loss that weights misses more |
| AdamW, cosine schedule | Optimiser with decoupled weight decay; learning rate that decreases smoothly |
| Epoch (here) | 960 sampled 512 px training patches |
| Seed | Number that fixes the random choices of a run |
| Mixed precision / FP16 / INT8 | Doing calculations / storing weights in 16-bit or 8-bit numbers |
| Pruning / distillation | Removing internal channels / training a small model from a large model's outputs |
| Ensemble | Averaging the predictions of several models |
| TTA | Test-time augmentation: averaging predictions over flipped copies of an image |
| CLAHE | Contrast-limited adaptive histogram equalisation (local contrast boost) |
| Hysteresis / closing / watershed | Keep weak pixels only next to strong ones / fill small gaps / split regions at confident cores |
| Bootstrap CI | Uncertainty range from re-sampling validation images many times |
| Leakage / embargo | Evaluation images influencing training unfairly / time gap that prevents it |

---

## 9. Remaining actions before 23:30

1. **Kaggle placeholders** in the updated deck:
   - Slide 13: the "Kaggle: add result" card (public PQ, rank and date) and the "[Insert leaderboard screenshot here]" box. Also update the slide-13 speaker notes, which still say "Fill the Kaggle score…".
   - Slide 16: change "Unlabelled: Kaggle score pending." to the score. The slide-16 notes still mention the score as pending.
   - Suggested gap sentence: *"Local validation 0.40 vs Kaggle about 0.32: different images and annotators, and our cleanup was tuned on the validation set."*
2. **Confirm which CSV was uploaded to Kaggle.** It is presumably `runs/person_c/final/submission.csv` (SHA-256 `1b08bcb078b2a44ad9e64545490c875b5c0fb01db87a1b34c2570825a5e9fb2f`). The ZIP must contain **that exact file**.
3. **Record the video.**
   - Each presenter's webcam goes in the upper-right corner. It will cover the "COS711 Assignment 2 / Segmenting…" header text on slides 4 onwards; this is acceptable.
   - Keep it to 10 minutes or less. The notes take about 8 minutes, leaving a margin.
   - Merge and compress the parts to 50 MB or less:
     ```powershell
     .\.venv\Scripts\python.exe delivery\prepare_video.py part_A.mp4 part_B.mp4 part_C.mp4 --output presentation.mp4
     ```
4. **Build the ZIP.** It excludes data, checkpoints and caches, and checks the CSV header and video limits.
   ```powershell
   .\.venv\Scripts\python.exe make_submission_zip.py --submission runs\person_c\final\submission.csv --slides "C:\Users\denis\Downloads\COS711_A2_Final_Presentation_updated.pptx" --video presentation.mp4 --screenshot leaderboard.png --output COS711_A2_submission.zip
   ```
   `--screenshot` is optional; include it if the team is in the top 100.
5. **Upload the ZIP to ClickUP.** Only the last upload is marked.
6. Optionally, commit the `feat/improvements` work without data or weights.

**Minor deck quirks (harmless):**
- Slide 2 now contains two identical "Alisha Perumal [u22512285]" text boxes; one was a hidden Morph helper whose wrong number was corrected.
- Slides 1–4 have no speaker notes.
- Slide 14's presenter label can be changed to Denis.

---

## 10. The validation-to-Kaggle gap

Local validation scored about 0.40 and Kaggle about 0.32. The cause is **not verified**. Plausible contributors, to be presented as hypotheses only:

- The cleanup thresholds and the final-model choice were made using the same 122 validation images. The split-half checks support a real gain, but the absolute level can still be optimistic.
- The validation set is small. Kaggle's public leaderboard may score only part of the 180 test images, and the test images may have different annotators or labelling density.
- The three test images with no detections contribute only false negatives.
- The lesson stated on slide 20 is to **submit to Kaggle earlier** to check how well validation predicts the test score.

---

## 11. Where things are (repository root unless stated)

| Topic | Files |
|---|---|
| Spec and team plan | [COS711-A2.pdf](COS711-A2.pdf), `COS711 Assignment 2 Team Plan and Implementation Guide.docx`, [HANDOFF.md](HANDOFF.md), [PERSON_C_START_HERE.md](PERSON_C_START_HERE.md), [README.md](README.md) |
| Data analysis (A) | [eda.ipynb](eda.ipynb), [results/eda_summary.json](results/eda_summary.json), `figures/eda_*.png`, [data.py](data.py), [splits/split_v1.json](splits/split_v1.json) |
| Training study (B) | [train.py](train.py), [models.py](models.py), [losses.py](losses.py), [experiments.py](experiments.py), [evaluate.py](evaluate.py), [metrics.py](metrics.py), [results/person_b_study/](results/person_b_study/) (study_report.md, methodology.md, screening.csv, confirmation.csv, fresh_seed_summary.csv) |
| Post-processing and compression (C) | [postprocess.py](postprocess.py), [offline_eval.py](offline_eval.py), [compress.py](compress.py), [compress_onnx.py](compress_onnx.py), [predict.py](predict.py), [results/person_c/](results/person_c/) (person_c_results.md, summary.json, compression.csv, final_*.json), `figures/person_c/` |
| Improvement round | [improve.py](improve.py), [configs/improvements/](configs/improvements/), [results/improvements/README.md](results/improvements/README.md), `runs/improvements/` (Git-ignored: models, cached probabilities, ensemble CSVs) |
| Presentation builders | [delivery/update_final_presentation.py](delivery/update_final_presentation.py) and [delivery/final_deck_figures.py](delivery/final_deck_figures.py) (build the 23-slide final deck from the team's 18-slide deck), [delivery/build_template_deck.py](delivery/build_template_deck.py) (earlier university-template deck), [delivery/README.md](delivery/README.md) |
| Video and ZIP | [delivery/prepare_video.py](delivery/prepare_video.py), [make_submission_zip.py](make_submission_zip.py) |
| Final model and CSV (Git-ignored) | `runs/person_c/compression/pruned30_fp16.pt`, `runs/person_c/final/submission.csv` |

**Local data setup** (data cannot go on GitHub):
- Extract the Kaggle ZIP into `data/` so that it contains:
  - `data/train/train_images/`
  - `data/train/MAGFiLO_1.0_Annotations_kaggle2026_train.json`
  - `data/test/test_images/`
- Extract Person B's model package into `runs/person_b_final/`.
- Run `.\.venv\Scripts\python.exe verify_inputs.py` to check all 888 dataset files and 386 package files against the recorded hashes.

**Rebuild the final deck** (writes a new file; the input is never changed):
```powershell
.\.venv\Scripts\python.exe delivery\final_deck_figures.py
.\.venv\Scripts\python.exe delivery\update_final_presentation.py --input "C:\Users\denis\Downloads\COS711_A2_Final_Presentation.pptx" --output "C:\Users\denis\Downloads\COS711_A2_Final_Presentation_v2.pptx"
```

**Environment:** Windows PowerShell; `.\.venv\Scripts\python.exe` (Python 3.12.9, PyTorch 2.13.0+cu132, RTX 3070 8 GB on Denis's machine). Run GPU jobs one at a time, and close games and other GPU-heavy applications first.
