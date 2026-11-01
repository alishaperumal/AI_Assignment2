# Local Person B presentation

## Updated final presentation (template gaps filled)

`update_final_presentation.py` reads the team's designed `COS711_A2_Final_Presentation.pptx` and writes a **new** 23-slide file (the input is never modified). It adds the template items the deck was missing: noisy labels, the 11-recipe tuning screen, the final model's kinds of errors, best/worst validation cases (before/after cleanup), what else we tried (including the ensemble and preprocessing retrains), formulation and reasons for choices, resolution/scaling/augmentation details, updated conclusions, and a data/libraries bibliography slide. New slides copy the deck's own shapes, header and Morph transition. Kaggle placeholders are left for the team. Speaker notes total about 1,130 words (roughly 8 minutes).

```powershell
.\.venv\Scripts\python.exe delivery\final_deck_figures.py
.\.venv\Scripts\python.exe delivery\update_final_presentation.py --input C:\path\COS711_A2_Final_Presentation.pptx --output C:\path\COS711_A2_Final_Presentation_updated.pptx
```

`final_deck_figures.py` renders the best/worst panels from the final model's saved validation probabilities and the noisy/faint/disagreeing panels chosen by `eda.ipynb`. The before-cleanup per-image scores come from `offline_eval.py score` on the same probabilities (`runs/improvements/deck/final_model_before_cleanup`), which reproduces the recorded 0.3288.

## New team deck using the supplied university template

**Use `COS711_A2_team_template.pptx` for the template-based team presentation.**
It preserves the supplied template's masters, university branding, blue headings,
red bullets, cover artwork, concluding photo panels and required section order.
The template's cover typo ("Assignment 3") is corrected to "Assignment 2".
Fourteen slides expand the template's sections for readable results and examples.
Existing Person B/C and combined decks remain unchanged.

The supplied original is retained unchanged in
`templates/COS711_A2_Presentation_Template.pptx`. Rebuild to a **new** filename:

```powershell
.\.venv\Scripts\python.exe delivery\build_template_deck.py --output delivery\COS711_A2_team_template_v2.pptx
```

The builder requires `python-pptx`, Pillow, matplotlib, NumPy and pycocotools,
plus the local test images and checked final CSV. It refuses to overwrite an
existing deck and rejects a final CSV whose SHA-256 differs from its recorded
receipt. It neither trains nor uploads. Test illustrations show original inputs
beside masks decoded from the **final** CSV; validation examples are explicitly
labelled teacher-before/after-cleanup illustrations.

Every slide has PowerPoint speaker notes with plain-language explanations,
evidence sources and a suggested duration. The adjacent
`COS711_A2_team_template.json` preserves the script, provenance and test-image IDs.
Suggested narration totals **8 minutes 30 seconds**, leaving rehearsal margin
under the 10-minute limit; actual recorded duration must still be checked.

Before recording, add names/student numbers on slides 1-2 and the actual Kaggle
score, rank/date and screenshot on slide 7. The displayed 0.4031 is **local
validation PQ**, not a Kaggle result. Keep the speaker webcam in the upper-right
corner; do not cover the university branding at the bottom right. Bibliography
slides need not be narrated. The final compressed score is one run, separate
from Person B's three-run mean and sample standard deviation.

Use this new deck in the existing final ZIP command:

```powershell
.\.venv\Scripts\python.exe make_submission_zip.py --submission runs\person_c\final\submission.csv --slides delivery\COS711_A2_team_template.pptx --video presentation.mp4 --output COS711_A2_submission.zip
```

The completed deck is `person_b_results.pptx`: ten slides using the actual 17-run study, three fresh-seed results, runtime measurements and labelled validation examples. `speaker_notes.md` contains the narration and supporting evidence, also embedded in the PPTX. `references.md` lists primary sources. `qa_summary.json` records package checks and visual review; `slide_previews/` contains final rendered slides.

`presentation_plan.md` maps the assignment and team plan to the modelling slides. `slide_content.json` holds editable slide text and narration; `references.json` holds the bibliography. `result_sources.json` points to the real study evidence. The builder uses the bundled Artifact Tool to produce editable PowerPoint text and tables and embedded speaker notes. It never trains, changes the dataset/split, or uploads anything.

From the repository root in PowerShell:

```powershell
./delivery/build_person_b_slides.ps1
```

The build uses a new timestamped PPTX name in `delivery/output/`, which is ignored by Git. A chosen new filename can be supplied with `-OutputName person_b_results_v2.pptx`. Existing finalized files are never overwritten. The bundled runtime paths in the PowerShell launcher are specific to this installation; update them from Codex's workspace-dependencies tool if the bundled runtime moves. No dependency installation is needed.

The builder also writes editable `speaker_notes.md` and `references.md`. Private previews and validation reports live in `.build/`. Review every preview after changing content or supplying new results. The delivered deck uses the completed `results/person_b_study/summary.json`, generated by `study_report.py`. Explicit preparation fallbacks remain in the reusable builder for studies that have not finished; they are absent from the delivered deck.

Only completed full-budget comparisons enter the result table. The selected configuration's fresh-seed statistics remain separate from the configuration-selection seed. Image-level variability must not stand in for run-to-run variability. The current work does not upload to Kaggle, so a leaderboard score remains unavailable.

The ten-slide insert has seven spoken slides (allow about 3–4 minutes), one optional unlabelled-test illustration, and two bibliography slides. Highest/lowest validation cases use explicit selection roles, not list positions. Their close-ups include whole-image locators, and displayed PQ scores apply to whole observations. The upper-right corner is reserved for the speaker's camera. Combine this with Person A and C's contributions and any optional ClickUP template, add the real names, and rehearse. Human recording, authorization for any later Kaggle upload, and final account submission remain outside this generator.

`configuration_selection.json` provides the frozen selection decision. The builder preserves its near-tie interpretation and does not treat a single selection seed as evidence of statistical superiority. The `no_augmentation` label is explained as disabling added transforms while keeping random foreground-biased crops. Pixel Dice/IoU are identified as foreground coverage scores, and timing separates the training loop from the latest subprocess wall time and complete test export. `presentation_evidence.mjs` contains lightweight selection/formatting helpers; its Node tests require no model or rendering runtime.

The main pooled validation PQ is **0.33695 ± 0.00758**, the mean and sample standard deviation across seeds 43, 44 and 45. The representative seed-45 checkpoint scores 0.34481 and supplies the pictures; it is not the average run. It makes fewer extra predictions than baseline (1007 versus 1416 pooled false positives), but misses more reference instances (507 versus 449 false negatives). The notes preserve that tradeoff. The checked no-archive image-to-CSV pipeline averages 536.3 ms/image, including the first image and excluding model loading and probability-archive saving. The separate canonical export includes full-resolution probability and instance arrays for Person C.

## Person C additions: team deck, video and final ZIP

`build_team_deck.py` (python-pptx) builds Person C's slides from the recorded Person C results and assembles the team presentation. It opens `person_b_results.pptx` read-only as the style template and never overwrites it.

```powershell
.\.venv\Scripts\python.exe person_c_report.py          # refresh results/person_c and figures/person_c
.\.venv\Scripts\python.exe delivery\build_team_deck.py
```

| Output | Contents |
|---|---|
| `person_c_enhancement.pptx` | Person C section only: error analysis, post-processing study, selected rule, best/worst cases, negative results, compression, final pipeline, conclusions, references |
| `COS711_A2_team_presentation.pptx` | Title and roles, three dataset slides drafted from Person A's recorded EDA, Person B's seven spoken slides and test illustration, the Person C section, then all references |
| `person_c_speaker_notes.md` | Narration for the Person C slides (also embedded as PowerPoint notes) |

Every new slide keeps the upper-right corner free for the speaker's camera, as on Person B's slides. Person A should check the three dataset slides (they only use A's recorded `results/eda_summary.json` and `figures/eda_*.png`). Before recording: replace "Person A/B/C" with names, add the Kaggle score to the final-pipeline slide after uploading, and rehearse with a timer (the assignment limit is 10 minutes).

Record each section with a screen recorder that places your webcam (head and shoulders) in the upper-right corner, then merge and compress below the limits with the bundled ffmpeg (`pip install imageio-ffmpeg`):

```powershell
.\.venv\Scripts\python.exe delivery\prepare_video.py part_A.mp4 part_B.mp4 part_C.mp4 --output presentation.mp4
```

Finally build the ZIP from the exact CSV uploaded to Kaggle (it checks the CSV header, the 50 MB / 10 minute video limits, and excludes data, caches, weights and run outputs):

```powershell
.\.venv\Scripts\python.exe make_submission_zip.py --submission runs\person_c\final\submission.csv --slides delivery\COS711_A2_team_presentation.pptx --video presentation.mp4 --output COS711_A2_submission.zip
```
