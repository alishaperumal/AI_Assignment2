# Repository guidance for coding assistants

## Start with the current state

Read [PERSON_C_START_HERE.md](PERSON_C_START_HERE.md), [README.md](README.md), [HANDOFF.md](HANDOFF.md), then the linked frozen Person C handoff and results interpretation. The assignment PDF and Word team guide define requirements and responsibilities. The current user request determines the scope of work.

Person A fixes and Person B's local research/deliverables were completed on 1 November 2026: 17 research runs, 186 epochs, three fresh seeds, verified model/prediction package and presentation content. [Completion evidence](results/person_b_completion_verification.json) records that historical state. Person C's enhancement, compression and delivery work followed the same day; its results, configs and provenance are in [results/person_c/person_c_results.md](results/person_c/person_c_results.md) and `results/person_c/`. Do not rerun completed training as routine setup. Inspect actual files, summary/history records, active processes and available disk space before starting jobs.

## Preserve evidence and user constraints

- Keep raw data and `splits/split_v1.json` unchanged. All annotations for one observation belong to the same partition.
- Treat `runs/person_b_final/`, existing B run outputs and `results/person_b_study/` as preserved evidence. Put C experiments in new directories such as `runs/person_c/<experiment>/` and `results/person_c/`. Evaluation/export commands can overwrite outputs; use fresh destinations.
- Develop in the working repository or a separate copy, not inside the verified package. Working source can evolve with recorded provenance; do not rewrite past results to imply they used new code.
- No Kaggle/ClickUP uploads unless the user explicitly changes the existing no-upload instruction. No external test labels or annotation spine/bbox/category supervision. Do not use time/instrument metadata as model inputs.
- Run GPU jobs sequentially, check for other jobs and do not interfere with them. Use the existing `.venv\Scripts\python.exe` here; read README for other machines. Preserve important artifacts and monitor storage. D: is an authorized backup option if storage becomes low.
- Keep implementation understandable. Use a bounded experiment plan, justify changes with validation evidence, retain failed trials, and do not reduce scope because of the deadline.

## Avoid these interpretation and implementation mistakes

- Main result: PQ 0.33695 +/- 0.00758, sample SD across fresh seeds 43/44/45. Selection seed 42 is excluded. Exported seed 45 scores 0.34481; it is not the mean run. Per-image variability is not across-run variability.
- The selected recipe disables added transforms, not random foreground-biased crops. Its near-tie with 1024 whole images does not establish statistical superiority.
- Extra misses are especially pronounced in exported seed 45: FN 507 versus baseline 449. The controlled seed-42 recipe comparison has FN 451 versus 449. Do not attribute all 58 extra misses to augmentation removal.
- `evaluation.threshold` controls pixel inclusion; `evaluation.iou_threshold` controls object matching. Preserve official strict IoU > 0.5 matching and pooled `PQ_dataset` over all annotation sets. Do not substitute a mean of image PQ values.
- Preserve `evaluate.py`'s handling of original overlapping/empty annotation layers. Only evaluating integer-cache maps loses some reference information.
- Saved NPZ probabilities are scored offline with `offline_eval.py` (it reproduced the seed-45 baseline exactly before any modified processing was trusted). `evaluate.py` still runs checkpoint inference; do not invent `--predictions-dir`. Post-processing is configured by the `postprocess` config section (defaults reproduce the baseline) and applied identically by `evaluate.py`, `predict.py` and the offline runner.
- Mixed-precision inference and omission of optimizer state do not by themselves prove a compression improvement. Person C's compression study compares quality, deployable size and consistently scoped end-to-end runtime per variant (`compress.py`, `compress_onnx.py`).
- Three zero-detection test images are recorded in the sidecar. Do not fabricate CSV objects or imply Kaggle acceptance. Test images have no supplied labels, so no accuracy ranking is available.

## Verification and handoff hygiene

Run tests appropriate to changed behavior. The last completed full Python suite passed 101 tests; this is evidence for that version, not a promise about future edits. Use `python -m pytest -q` with the selected interpreter for a full check. Documentation-only changes need factual/link checks, not another training run.

Read existing tests and reuse the metric/export APIs rather than duplicating their policies. Record configs, checkpoint/split/source provenance, seeds/budgets, metrics, timing boundaries and actual completion status for new experiments. First reproduce the unchanged saved-probability result before trusting any offline post-processing runner.

Keep the start guide/current documentation updated as C work progresses. Preserve historical B reports and manifests; put new conclusions in C records. Inspect `git status`: the current working copy may contain code/docs not yet shared through Git. Git also does not transfer ignored weights/predictions: hand over the current repository plus the verified package, and any explicitly needed alternative weights. Do not claim a long task is complete merely because its process was launched.
