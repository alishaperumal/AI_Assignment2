# Person B presentation and recording plan

This is the modelling section to insert into the team's presentation. It does not replace Person A's dataset section or Person C's enhancement/compression section. The assignment PDF is authoritative. The Word guide supplies the team responsibilities and a suggested structure. Its deadline-based schedule is not used to reduce the work.

## Requirements from the supplied documents

| Requirement | Source | Where it belongs |
|---|---|---|
| Explain formulation, models, losses, training algorithms, tuning and evaluation | Assignment §4.2; Word guide Person B role and Steps 4–5 | Person B core slides 1–3 |
| Compare approaches with local PQ and additional metrics, repeated-run mean/std where feasible | Assignment §4.3; Word guide Steps 5 and 7 | Results slides and the experiment log |
| Show error types, both good and poor predictions, plus test-image predictions | Assignment §4.3 | Shared A/B Results section; labelled validation images support quantitative ranking, test images support qualitative discussion only |
| Report final model size, training time and end-to-end inference time | Assignment §3.1; Word guide Step 6 | Person B efficiency slide and Person C's compression comparison |
| Explain improvements and their measured effects | Assignment §4.4 | Person B modelling comparisons, followed by Person C's instance separation/post-processing/compression |
| Bibliography | Assignment §4.6 | Two unspoken reference slides that can merge with the team's bibliography |
| All members appear head-and-shoulders in the upper-right corner and identify their roles | Assignment §4 | Team title/roles slide and each member's recording |
| Video at most 10 minutes and 50 MB | Assignment §4 | Final team recording and export check |
| ZIP contains code, exact uploaded submission.csv, presentation video and PowerPoint slides | Assignment §§1,4 | Final team delivery owner, normally Person C |

The actual PDF lists the presentation topics as numbered items within §4, not numbered subsections. The table's §4.2–§4.6 shorthand refers to those list items.

## Scope and slide sequence

The completed `person_b_results.pptx` has seven spoken slides, an optional test-image backup, and two bibliography slides. Allocate approximately 3–4 minutes to Person B, including their share of Results. Coordinate this with Person A and C to keep the combined video below 10 minutes. The deck has no extra team cover, since it is an insert into the shared presentation.

| Slide | Purpose | Approximate speaking time | Evidence |
|---|---|---:|---|
| Baseline model | Explain grayscale input, U-Net/ResNet34, foreground probability, and component-based instances | 30 s | models.py, inference.py, actual baseline config |
| Training and validation | Explain the loss, sampling, optimizer and metric choice in plain language | 40 s | data.py, losses.py, train.py, config and frozen split |
| Controlled comparisons | Explain the screen, full-budget comparison and separate fresh-seed assessment | 35 s | Completed study plan and experiment records |
| Validation results | Present measured comparisons and the fresh-seed mean/std without confusing it with image variability | 40 s | study_report.py summary and immutable run artifacts |
| Highest validation score | Inspect the actual strongest selected-model labelled case | 25 s | Example with the explicit highest-score selection role |
| Lowest validation score | Inspect the actual weakest selected-model labelled case | 25 s | Example with the explicit lowest-score selection role |
| Efficiency and handoff | State size/time scope and the boundary with Person C's enhancements | 35 s | Selected checkpoint, measured test timing sidecar, handoff manifest |
| Test image illustration | Optional shared-results or backup example without an accuracy ranking | 0 s | Actual unlabelled test output, with its sampling reason |
| Model and loss references | Bibliography | 0 s | Primary research papers and library source |
| Evaluation and training references | Bibliography | 0 s | Official scoring reference, PQ paper, optimizer/scheduler documentation |

All result slides use completed measured artifacts. Smoke runs, synthetic checks and tiny-overfit results are excluded from the research comparison. Negative comparisons remain in the full experiment log.

## Completed study evidence

The GPU study is complete: a 20-epoch baseline, eleven six-epoch screens with seed 42, two 20-epoch candidate confirmations with seed 42, and three 20-epoch fresh runs (43, 44, 45). Seed 42 selects the configuration; fresh seeds estimate variability. The main result is PQ 0.33695 ± 0.00758 (sample standard deviation, three runs). The representative seed-45 checkpoint supplies the pictures and scores 0.34481; it is not the mean run.

The study tables, figures and machine-readable summary are in `results/person_b_study/`. The final model/config/predictions/CSV/manifest are in `runs/person_b_final/`; original final-seed runs remain in `runs/final/{variant}_seed{43,44,45}`. The presentation reads report evidence without running models or recomputing evaluation.

No Kaggle upload is authorized in this work. The deck must say that the leaderboard score is unavailable until a team member actually uploads the exact CSV and records the returned score. A locally valid CSV is evidence of encoding checks, not leaderboard acceptance.

The frozen selection decision chose `no_augmentation` narrowly over `resolution_1024`. It disables added transformations while preserving random foreground-biased 512-pixel crops, U-Net/ResNet34 and BCE + Dice. The exact measured scores and margin come from `configuration_selection.json`; they represent a near-tie at one selection seed, not established statistical superiority. Fresh-seed statistics and the representative checkpoint come from the completed study summary. Training budgets match samples and planned update attempts, since AMP overflow handling can skip a few applied updates. Training-loop time and the latest subprocess wall time have separate scopes. Pixel Dice/IoU measure semantic foreground coverage, which can remain good while individual filaments merge or split.

The representative checkpoint makes fewer extra predictions than baseline (1007 versus 1416 pooled false positives), but misses more reference instances (507 versus 449 false negatives). Higher PQ therefore comes with a tradeoff. The highest-score example retains the same main filament while removing extra predictions elsewhere in the observation. The lowest-score example covers much less of the reference filament and has no qualifying instance match. Close-ups have whole-image locators; their PQ scores belong to the whole observation. The optional test illustration is selected by prediction count, without a ground-truth ranking.

The canonical export covers 180 test images, with 1831 predicted instances and three zero-detection images. The separate checked no-archive image-to-CSV benchmark took 96.535 seconds, averaging 536.3 ms/image. It includes the first image and excludes model loading and probability-archive saving. It is distinct from model-only speed and the longer archival export. Person C receives full-resolution probabilities and instance maps for post-processing and compression experiments.

## Design and source handling

Use a clean 16:9 academic layout with editable text/tables, Arial, strong contrast, and generous whitespace. Reserve the upper-right region (approximately x=1010–1250, y=20–170 on a 1280×720 canvas) for the camera overlay. Keep titles left of that region and all substantial content below it. Keep method details in speaker notes rather than shrinking the slide text.

The ClickUP template mentioned in the assignment and Word guide is absent from the repository. The neutral local deck is complete. A team member should retrieve the template and transfer the content if the team chooses to use it. The PDF says the template may be adapted; its absence does not prevent preparing the requested slide content.

Use supplied images and actual experiment artifacts for evidence. Do not create synthetic model examples. Cite papers and source artifacts in the relevant slide notes. Method sources explain the ideas; they do not establish that a method improved this solar-filament task.

Bibliography details were checked against the primary papers, the publisher, and official library documentation. The MAGFiLO journal article's published title is **A dataset of manually annotated filaments from H-alpha observations**, Scientific Data 11, 1031 (2024). The assignment's bibliography uses the dataset name as shorthand. The project uses ordinary cosine annealing without warm restarts, so cite the implemented PyTorch scheduler rather than implying the full SGDR algorithm. Reading the paper does not authorize downloading the separate annotation release, and no such labels were obtained.

## Remaining human actions

1. Replace the role label with the presenter's name and confirm team responsibilities.
2. Retrieve the optional ClickUP template and combine A/B/C slides, removing duplicate bibliography entries.
3. Rehearse the supplied narration and confirm its measured gains, negative results and observed errors with the team.
4. Record each member's own head-and-shoulders video in the upper-right corner; rehearse the combined timing.
5. Upload the selected exact CSV to Kaggle if the team authorizes it, record acceptance/score, and keep that same file for the final ZIP.
6. Include Person C's actual enhancement/compression evidence, final conclusions, and required test examples before calling the team presentation complete.
7. Check the final video duration/size and ZIP contents, then submit through the authorized student account.

