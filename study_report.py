"""Summarize the staged Person B study from saved evidence, without training.

Inputs are protocol.json and selection.json. Relative paths refer to the
repository root. Selection uses seed 42; fresh seeds are a separate estimate
of run variability on the same validation split, not an independent test set.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import statistics

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from report import _evaluation_provenance

ROOT = Path(__file__).resolve().parent
METRICS = ("PQ_dataset", "SQ_dataset", "RQ_dataset", "tp", "fp", "fn", "sum_iou", "filament_dice_mean",
           "filament_iou_mean", "pixel_dice_mean", "pixel_iou_mean", "observation_PQ_mean", "any_overlap_one_to_many",
           "any_overlap_many_to_one", "custom_one_to_many", "custom_many_to_one",
           "baseline_prediction_and_components_ms_per_image")
LIMITATIONS = [
    "Screening and full-budget confirmation are separate comparisons; compare only matching budgets.",
    "Seed 42 selected the method. Fresh seeds estimate training variability on the same validation "
    "split; they do not remove validation-selection bias or provide an independent test score.",
    "Across-run standard deviations use ddof=1. Per-observation/annotation-set variation is a "
    "different statistic and is never substituted for run variability.",
    "pixel_dice_mean and pixel_iou_mean average foreground-mask scores across annotation sets. "
    "They measure semantic pixel coverage, distinct from pooled instance PQ and best-per-GT "
    "filament overlap. Good foreground coverage can still merge or split individual filaments.",
    "The 1024 trial changes resolution, field of view, pixels per update and foreground sampling: "
    "whole reduced images are compared with full-resolution 512-pixel crops.",
    "Consensus changes target construction and observation weighting; with two annotators its "
    "tie-inclusive vote is a union. It is not a pure voting-only experiment.",
    "Tversky weights missed foreground more heavily; it does not directly enforce thin structures "
    "or topology. Any continuity benefit must be supported by observed errors.",
    "Official overlap distributions include every positive-overlap pair. Best-per-GT distributions "
    "include missed objects as zero. Relation counts repeat predictions across annotation sets.",
    "Read fragmentation/merging counts alongside PQ, RQ, matched objects (TP), missed objects (FN) "
    "and extra predictions (FP). Fewer relation errors can result from missing more filaments, "
    "rather than improved segmentation. All these counts are pooled over annotation-set comparisons.",
    "Training seconds cover the training loop; epoch totals include validation but exclude setup "
    "and checkpoint serialization. Job sidecars, when supplied, record invocation wall time.",
    "Experiment-record subprocess wall time includes setup, epoch loops and checkpoint writes. "
    "It covers the latest recorded train.py invocation, not necessarily every invocation of a "
    "resumed run. Baseline job-sidecar wall time likewise remains scoped to its resumed invocation.",
    "Training budgets match samples and planned optimizer-update attempts. Mixed-precision gradient "
    "scaling may skip an update on overflow; completed updates and AMP skips are reported separately. "
    "A few recorded skips are not by themselves a failed run.",
    "Default inference uses mixed precision with float32 stored weights. Compact checkpoints omit "
    "optimizer state; that alone is not model compression.",
    "Test images have no supplied labels. Their panels illustrate predictions without accuracy "
    "rankings. Kaggle scores are unavailable because no upload is requested.",
]


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf8"))


def resolve_path(path):
    path = Path(path)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def _config(path):
    return yaml.safe_load(Path(path).read_text(encoding="utf8")) or {}


def _hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _prediction_settings(cfg):
    """Settings that can change scores while keeping model weights identical."""
    return dict(inference=cfg.get("inference", {}),
                preprocess=cfg.get("data", {}).get("preprocess", {}),
                threshold=cfg.get("evaluation", {}).get("threshold", 0.5),
                iou_threshold=cfg.get("evaluation", {}).get("iou_threshold", 0.5))


def _comparable_config(cfg):
    cfg = deepcopy(cfg)
    cfg.pop("run", None)
    cfg.pop("seed", None)
    cfg.get("data", {}).pop("seed", None)
    return cfg


def _update_accounting(directory, cfg, planned_epochs, summary):
    """Report attempted versus applied updates; absent evidence remains unknown."""
    data, train = cfg.get("data", {}), cfg.get("train", {})
    samples, batch = data.get("epoch_length"), data.get("batch_size")
    accumulation = train.get("accumulation_steps", 1)
    planned = None
    if samples is not None and batch is not None:
        batches = int(samples) // int(batch) if data.get("drop_last", True) else math.ceil(int(samples) / int(batch))
        planned = int(planned_epochs) * math.ceil(batches / int(accumulation))
    result = dict(planned_update_attempts=planned, recorded_update_attempts=None,
                  amp_skipped_steps=None, update_accounting="unavailable_missing_history_counts")
    history_path = directory / "history.csv"
    if not history_path.exists():
        return result
    history = pd.read_csv(history_path)
    columns = ["optimizer_steps", "skipped_optimizer_steps"]
    if not set(columns) <= set(history.columns) or history[columns].isna().any().any():
        return result
    values = history[columns].to_numpy(dtype=float)
    if not np.isfinite(values).all() or np.any(values < 0) or not np.equal(values, np.floor(values)).all():
        raise ValueError(f"Invalid optimizer update counts: {history_path}")
    applied, skipped = map(int, values.sum(axis=0))
    if summary.get("optimizer_steps") is not None and int(summary["optimizer_steps"]) != applied:
        raise ValueError(f"Optimizer step summary disagrees with history: {directory}")
    result.update(optimizer_steps=applied, recorded_update_attempts=applied + skipped,
                  amp_skipped_steps=skipped,
                  update_accounting="recorded" if planned is not None else "recorded_counts_unknown_planned_budget")
    return result


def _experiment_timing(path, directory, seed, cfg):
    """Read measured subprocess time only after matching the run, seed and config."""
    path = resolve_path(path)
    record = read_json(path)
    if (resolve_path(record["run_dir"]) != directory.resolve()
            or int(record["seed"]) != int(seed)):
        raise ValueError(f"Experiment timing record has different run/seed provenance: {path}")
    config_hash = hashlib.sha256(json.dumps(cfg, sort_keys=True, separators=(",", ":"),
                                           allow_nan=False).encode()).hexdigest()
    if record.get("config_sha256") != config_hash:
        raise ValueError(f"Experiment timing record has different or missing configuration hash: {path}")
    seconds = record.get("invocation_seconds")
    if seconds is not None and (not math.isfinite(float(seconds)) or float(seconds) < 0):
        raise ValueError(f"Invalid measured subprocess wall time: {path}")
    return dict(subprocess_wall_seconds=float(seconds) if seconds is not None else None,
                subprocess_wall_scope="latest recorded train.py subprocess invocation, including setup, "
                    "epoch loops and checkpoint writes; resumed runs may have earlier invocations",
                subprocess_wall_record=str(path), subprocess_wall_status=record.get("status"),
                subprocess_wall_started_utc=record.get("last_started_utc", record.get("started_utc")),
                subprocess_wall_finished_utc=record.get("finished_utc"))


def load_run(item, planned_epochs, expected_split=None):
    """Read a run and verify any accompanying evaluation uses the same pipeline."""
    directory = resolve_path(item["run_dir"])
    row = dict(variant=item.get("variant", item.get("experiment", directory.name)),
               seed=item.get("seed"), run_dir=str(directory), status="pending",
               epochs_planned=int(planned_epochs), epochs_completed=0,
               Kaggle_PQ=None, evaluation_provenance="unavailable")
    path = directory / "summary.json"
    if not path.exists():
        return row
    summary = read_json(path)
    cfg = _config(directory / "config.yaml")
    seed = int(summary["seed"])
    if row["seed"] is not None and int(row["seed"]) != seed:
        raise ValueError(f"Seed disagrees with study declaration: {directory}")
    if summary.get("overfit_images", 0):
        raise ValueError(f"Diagnostic run cannot enter study results: {directory}")
    split_hash = summary.get("split_sha256")
    if expected_split and split_hash != expected_split:
        raise ValueError(f"Frozen split hash differs or is missing: {directory}")
    actual_budget = int(cfg["train"]["epochs"])
    completed = int(summary["epochs_completed"])
    reason = summary.get("stop_reason", "unknown")
    status = "completed" if completed >= planned_epochs and actual_budget == planned_epochs else "partial"
    if reason == "invocation_limit":
        status = "partial"
    values = summary["best_validation"]
    if not math.isfinite(float(values["PQ_dataset"])):
        raise ValueError(f"Non-finite primary score: {directory}")
    row.update(seed=seed, status=status, epochs_completed=completed,
               configured_epochs=actual_budget, stop_reason=reason,
               best_epoch=summary["best_epoch"], split_sha256=split_hash,
               parameters=summary.get("parameters"), weights_file_bytes=summary.get("weights_file_bytes"),
               weights_MiB=(summary["weights_file_bytes"] / 1024**2 if summary.get("weights_file_bytes") else None),
               training_seconds=summary.get("training_seconds"),
               epoch_loop_seconds=summary.get("total_seconds"),
               optimizer_steps=summary.get("optimizer_steps"),
               evaluation_provenance=_evaluation_provenance(directory, summary))
    row.update({key: values.get(key) for key in METRICS})
    row.update(_update_accounting(directory, cfg, planned_epochs, summary))
    metadata = directory / "run_metadata.json"
    sources = read_json(metadata).get("source_sha256") if metadata.exists() else None
    row["source_fingerprint"] = (hashlib.sha256(json.dumps(sources, sort_keys=True).encode()).hexdigest()
                                  if sources else None)
    # Do not silently mix the training-summary table with re-evaluated settings.
    evaluation = directory / "evaluation"
    if (evaluation / "config.yaml").exists():
        if _prediction_settings(_config(evaluation / "config.yaml")) != _prediction_settings(cfg):
            raise ValueError(f"Evaluation settings differ from training checkpoint: {directory}; "
                             "keep tuned pipelines in a separate comparison")
    if (evaluation / "summary.json").exists():
        scored = read_json(evaluation / "summary.json")
        if not math.isclose(float(scored["PQ_dataset"]), float(values["PQ_dataset"]), rel_tol=1e-6, abs_tol=1e-8):
            raise ValueError(f"Saved evaluation PQ differs from training summary: {directory}; rerun evaluation")
        for key in ("n_observations", "n_annotation_sets", "observation_PQ_std", "annotation_set_PQ_std"):
            row[key] = scored.get(key)
    if item.get("job_sidecar"):
        job = read_json(resolve_path(item["job_sidecar"]))
        row["invocation_wall_seconds"] = job.get("elapsed_seconds")
        row["invocation_wall_scope"] = "one recorded job invocation, not necessarily the whole resumed run"
    if item.get("experiment_record"):
        row.update(_experiment_timing(item["experiment_record"], directory, seed, cfg))
    return row


def fresh_seed_statistics(rows, expected_seeds):
    """Aggregate only completed independent seeds, keeping example variability separate."""
    expected_seeds = list(map(int, expected_seeds))
    if len(expected_seeds) != len(set(expected_seeds)):
        raise ValueError("Expected fresh seeds must be unique")
    seeds = [int(r["seed"]) for r in rows]
    if len(seeds) != len(set(seeds)) or set(seeds) - set(expected_seeds):
        raise ValueError("Fresh-seed declarations contain duplicates or unexpected/selection seeds")
    completed = [r for r in rows if r["status"] == "completed"]
    configurations = [_comparable_config(_config(Path(r["run_dir"]) / "config.yaml")) for r in completed]
    if configurations and any(c != configurations[0] for c in configurations[1:]):
        raise ValueError("Fresh seeds do not share the same method and training budget")
    sources = {r.get("source_fingerprint") for r in completed}
    if len(sources) > 1:
        raise ValueError("Fresh seeds have different or inconsistently recorded source fingerprints")
    result = dict(n_runs=len(completed), seeds=[r["seed"] for r in completed],
                  expected_seeds=expected_seeds, std_definition="sample standard deviation across runs; ddof=1",
                  source_provenance="recorded" if sources and None not in sources else "unverified")
    for metric in METRICS:
        values = [float(r[metric]) for r in completed if r.get(metric) is not None]
        result[metric] = dict(n=len(values), mean=statistics.mean(values) if values else None,
                              std_across_runs=statistics.stdev(values) if len(values) > 1 else None)
    return result


def _save_csv(rows, path):
    pd.DataFrame(rows).to_csv(path, index=False)


def _stage_plot(rows, path, title):
    complete = [r for r in rows if r["status"] == "completed" and r.get("PQ_dataset") is not None]
    if not complete:
        return None
    complete.sort(key=lambda r: r["PQ_dataset"])
    fig, ax = plt.subplots(figsize=(10, max(3, 0.45 * len(complete) + 1.6)))
    ax.barh([r["variant"] for r in complete], [r["PQ_dataset"] for r in complete])
    for i, row in enumerate(complete):
        ax.text(row["PQ_dataset"] + 0.002, i, f"{row['PQ_dataset']:.4f}", va="center")
    ax.set(xlabel="Validation PQ (pooled annotation sets)", title=title)
    ax.set_xlim(0, min(1, max(r["PQ_dataset"] for r in complete) * 1.2 + 0.02))
    fig.tight_layout(); fig.savefig(path); plt.close(fig)
    return str(path)


def _learning_curves(confirmation, fresh, path):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    used = False
    for row_axes, rows, stage in zip(axes, (confirmation, fresh), ("Full-budget confirmation", "Fresh-seed estimation")):
        for row in rows:
            history = Path(row["run_dir"]) / "history.csv"
            if not history.exists():
                continue
            table = pd.read_csv(history)
            label = f"{row['variant']} / seed {row['seed']}"
            row_axes[0].plot(table["epoch"], table["train_loss"], label=label)
            measured = table.dropna(subset=["val_PQ_dataset"])
            row_axes[1].plot(measured["epoch"], measured["val_PQ_dataset"], label=label)
            used = True
        for ax, ylabel in zip(row_axes, ("Training loss", "Pooled validation PQ")):
            ax.set(xlabel="Epoch", ylabel=ylabel, title=stage)
            if ax.lines:
                ax.legend(fontsize=8)
            ax.grid(alpha=0.2)
    if used:
        fig.tight_layout(); fig.savefig(path)
    plt.close(fig)
    return str(path) if used else None


def _distributions(baseline, representative, output):
    figures = {}
    comparisons = [("Baseline", Path(baseline["run_dir"])), ("Selected model", Path(representative["run_dir"]))]
    for filename, key, title, xlabel in [
        ("overlap_scores.npz", "official_overlap_distributions", "Official: every positive-overlap pair", "Pair overlap score"),
        ("filament_scores.npz", "best_gt_distributions", "Diagnostic: best match per GT, including missed objects", "Best GT overlap score")]:
        paths = [(label, d / "evaluation" / filename) for label, d in comparisons]
        if not all(p.exists() for _, p in paths):
            figures[key] = None
            continue
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        for label, path in paths:
            with np.load(path) as arrays:
                for ax, metric in zip(axes, ("dice", "iou")):
                    ax.hist(arrays[metric], bins=np.linspace(0, 1, 21), alpha=0.45, label=label)
                    ax.set(xlabel=xlabel, ylabel="Count", title=metric.upper())
        for ax in axes:
            ax.legend()
        fig.suptitle(title); fig.tight_layout()
        path = output / f"{key}.png"
        fig.savefig(path); plt.close(fig); figures[key] = str(path)
    values = [[r.get(k) for k in ("any_overlap_one_to_many", "any_overlap_many_to_one")]
              for r in (baseline, representative)]
    if all(v is not None for pair in values for v in pair):
        fig, ax = plt.subplots(figsize=(8, 4))
        x = np.arange(2)
        for offset, value, label in zip((-.18, .18), values, ("Baseline", "Selected model")):
            ax.bar(x + offset, value, width=.36, label=label)
        ax.set_xticks(x, ["Fragmented GT", "Merging predictions"])
        ax.set(ylabel="Count over all annotation-set comparisons", title="Relations using any positive overlap")
        ax.legend(); fig.tight_layout()
        path = output / "overlap_relationships.png"
        fig.savefig(path); plt.close(fig); figures["overlap_relationships"] = str(path)
    else:
        figures["overlap_relationships"] = None
    return figures


def _npz(path, observation):
    with np.load(path) as value:
        if str(value["observation"]) != observation:
            raise ValueError(f"Prediction observation identity mismatch: {path}")
        return value["probability"], value["instances"]


def _outline(ax, image, instances, title):
    ax.imshow(image, cmap="gray", vmin=0, vmax=1)
    if np.any(instances):
        ax.contour(instances > 0, levels=[.5], colors=["cyan"], linewidths=.6)
    ax.set_title(title, fontsize=9); ax.axis("off")


def _individual_panels(fig, axes, rows, output, prefix):
    """Export a readable single-example row for slide layouts as well as the overview."""
    from matplotlib.transforms import Bbox
    title = fig._suptitle
    if title is not None:
        title.set_visible(False)
    try:
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        for row_axes, row in zip(axes, rows):
            bounds = Bbox.union([ax.get_tightbbox(renderer) for ax in row_axes])
            bounds = bounds.transformed(fig.dpi_scale_trans.inverted()).padded(.12)
            path = output / f"{prefix}_{row['observation']}.png"
            fig.savefig(path, bbox_inches=bounds)
            row["panel"] = str(path)
    finally:
        if title is not None:
            title.set_visible(True)


def select_validation_examples(table):
    """Select global score/delta extremes and a typical case, without duplicates.

    Ties prefer an unused observation, then alphabetical observation ID. If one
    image meets multiple criteria, retain every reason on that single example.
    """
    if table.empty:
        return []
    if table.observation.duplicated().any():
        raise ValueError("Validation example table contains duplicate observations")
    table = table.copy()
    table["delta"] = table.PQ_final - table.PQ_baseline
    table["typical_distance"] = abs(table.PQ_final - table.PQ_final.median())
    chosen = {}
    criteria = [("PQ_final", False, "Highest selected-model observation mean PQ"),
                ("PQ_final", True, "Lowest selected-model observation mean PQ"),
                ("delta", False, "Largest observed gain (or smallest drop)"),
                ("delta", True, "Largest observed drop (or smallest gain)"),
                ("typical_distance", True, "Typical selected-model observation score")]
    for column, ascending, reason in criteria:
        extreme = table[column].min() if ascending else table[column].max()
        candidates = table[table[column] == extreme].sort_values("observation")
        unused = candidates[~candidates.observation.isin(chosen)]
        row = (unused if not unused.empty else candidates).iloc[0]
        obs = str(row.observation)
        if obs in chosen:
            chosen[obs]["reasons"].append(reason)
        else:
            chosen[obs] = dict(observation=obs, baseline_PQ=float(row.PQ_baseline),
                               final_PQ=float(row.PQ_final), delta=float(row.delta),
                               reason=reason, reasons=[reason])
    return list(chosen.values())


def mask_zoom_bounds(mask, minimum_size=256):
    """Square detail window derived only from mask pixels, kept inside the image."""
    ys, xs = np.nonzero(mask)
    if not len(xs):
        return None
    height, width = mask.shape
    size = min(height, width, max(minimum_size, math.ceil(1.7 * max(xs.max() - xs.min() + 1,
                                                                 ys.max() - ys.min() + 1))))
    x = max(0, min(width - size, int(round((xs.min() + xs.max() + 1 - size) / 2))))
    y = max(0, min(height - size, int(round((ys.min() + ys.max() + 1 - size) / 2))))
    return [int(x), int(y), int(x + size), int(y + size)]


def _validation_zoom(row, image, reference_rgb, reference_mask, predictions, output):
    """Keep full-image context beside mask-derived paired detail views."""
    from matplotlib.patches import Rectangle
    bounds = mask_zoom_bounds(reference_mask)
    if bounds is None:
        return
    x0, y0, x1, y1 = bounds
    region = np.s_[y0:y1, x0:x1]
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.5))
    axes[0].imshow(reference_rgb)
    axes[0].add_patch(Rectangle((x0, y0), x1-x0, y1-y0, fill=False, edgecolor="yellow", linewidth=1.2))
    axes[0].set_title("Whole-image context; yellow detail window", fontsize=9)
    axes[1].imshow(reference_rgb[region])
    axes[1].set_title("Human foreground outlines: detail", fontsize=9)
    for ax in axes[:2]:
        ax.axis("off")
    for ax, prediction, label, score in zip(axes[2:], predictions, ("Baseline", "Selected model"),
                                           (row["baseline_PQ"], row["final_PQ"])):
        _outline(ax, image[region], prediction[region], f"{label}: detail\nWhole-observation mean PQ={score:.3f}")
    fig.suptitle(f"{row['observation']} — mask-derived detail; this crop is not separately scored", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, .9))
    path = output / f"validation_zoom_{row['observation']}.png"
    fig.savefig(path); plt.close(fig)
    row.update(zoom_panel=str(path), zoom_bounds_xyxy=bounds,
               zoom_reason="Largest cached reference-instance mask by pixel area; padded square crop from mask pixels only")


def paired_validation_examples(baseline, representative, output):
    """Compare both models on identical deliberately selected labelled images."""
    import data as D
    from skimage.segmentation import find_boundaries
    directories = [Path(r["run_dir"]) for r in (baseline, representative)]
    paths = [d / "evaluation/observations.csv" for d in directories]
    if not all(p.exists() for p in paths):
        return None, []
    a, b = [pd.read_csv(p) for p in paths]
    if set(a.observation) != set(b.observation):
        raise ValueError("Paired examples require identical validation observations")
    table = a[["observation", "PQ"]].merge(b[["observation", "PQ"]], on="observation", suffixes=("_baseline", "_final"))
    chosen = select_validation_examples(table)
    for row in chosen:
        obs = row["observation"]
        if not all((d / "evaluation/predictions" / f"{obs}.npz").exists() for d in directories):
            return None, []
    if not chosen:
        return None, []
    cfg = _config(directories[0] / "config.yaml")
    catalog = D.load_catalog(Path(cfg["data"]["cache_dir"]))
    fig, axes = plt.subplots(len(chosen), 3, figsize=(13, 4 * len(chosen)), squeeze=False)
    for i, row in enumerate(chosen):
        obs = row["observation"]
        entries = catalog[catalog.obs == obs]
        image = D.load_image(Path(cfg["data"]["data_root"]) / "train/train_images" / entries.file_name.iloc[0])
        rgb = np.repeat(image[..., None], 3, axis=2)
        largest_mask, largest_area = np.zeros(image.shape, bool), 0
        for sid, color in zip(entries.set_id, [(1, .25, .25), (.25, 1, .25), (.3, .5, 1)] * len(entries)):
            gt = D.load_instance_map(sid, Path(cfg["data"]["cache_dir"]))
            rgb[find_boundaries(gt > 0)] = color
            labels, areas = np.unique(gt[gt > 0], return_counts=True)
            if len(labels) and int(areas.max()) > largest_area:
                largest_area = int(areas.max())
                largest_mask = gt == labels[areas.argmax()]
        axes[i, 0].imshow(rgb); axes[i, 0].axis("off")
        axes[i, 0].set_title(f"{obs}\nHuman foreground outlines", fontsize=9)
        predictions = []
        for ax, directory, label, score in zip(axes[i, 1:], directories, ("Baseline", "Selected model"),
                                                (row["baseline_PQ"], row["final_PQ"])):
            _, instances = _npz(directory / "evaluation/predictions" / f"{obs}.npz", obs)
            predictions.append(instances)
            _outline(ax, image, instances, f"{label}: observation mean PQ={score:.3f}\n{row['reason']}")
        _validation_zoom(row, image, rgb, largest_mask, predictions, output)
    fig.suptitle("Paired validation examples (selection rules stated; not a random sample)")
    fig.tight_layout(); path = output / "validation_paired.png"
    _individual_panels(fig, axes, chosen, output, "validation")
    fig.savefig(path); plt.close(fig)
    return str(path), chosen


def test_examples(representative, selection, test_summary, output):
    """Illustrate unlabelled images without assigning best/worst accuracy ranks."""
    import data as D
    directory = Path(representative["run_dir"])
    predictions = resolve_path(selection.get("test_predictions_dir", directory / "test_predictions"))
    if not predictions.exists():
        return None, []
    chosen = deepcopy(selection.get("test_examples", []))
    if not chosen:
        rows = test_summary.get("images", [])
        groups = {}
        for row in rows:
            groups.setdefault(row["observation"][-2:], []).append(row)
        for instrument, values in sorted(groups.items(), key=lambda kv: (-len(kv[1]), kv[0]))[:3]:
            values.sort(key=lambda r: (r["n_instances"], r["observation"]))
            row = values[len(values) // 2]
            chosen.append(dict(observation=row["observation"],
                               reason=f"Median predicted-instance count for instrument {instrument}; visual coverage only"))
    if not chosen:
        return None, []
    cfg = _config(directory / "config.yaml")
    images = Path(cfg["data"]["data_root"]) / "test/test_images"
    lookup = {D.obs_id_from_name(p.name): p for p in images.iterdir() if p.is_file()}
    fig, axes = plt.subplots(len(chosen), 3, figsize=(13, 4 * len(chosen)), squeeze=False)
    for row_axes, row in zip(axes, chosen):
        obs = row["observation"]
        image = D.load_image(lookup[obs])
        probability, instances = _npz(predictions / f"{obs}.npz", obs)
        row.update(unlabeled=True, n_predicted_instances=int(np.unique(instances[instances > 0]).size))
        row_axes[0].imshow(image, cmap="gray", vmin=0, vmax=1)
        row_axes[0].set_title(f"{obs}\nUnlabelled test image", fontsize=9)
        row_axes[1].imshow(probability, vmin=0, vmax=1, cmap="magma")
        row_axes[1].set_title("Predicted probability")
        _outline(row_axes[2], image, instances, f"Predicted foreground; instances: {row['n_predicted_instances']}\nAccuracy unknown")
        for ax in row_axes:
            ax.axis("off")
    fig.suptitle("Representative test predictions — no ground truth or accuracy ranking")
    fig.tight_layout(); path = output / "test_examples.png"
    _individual_panels(fig, axes, chosen, output, "test")
    fig.savefig(path); plt.close(fig)
    return str(path), chosen


def summarize_test_timing(sidecar, source):
    """Use complete CSV-operation time, keeping generator-timer averages separate."""
    seconds, count = sidecar.get("end_to_end_seconds"), sidecar.get("n_images")
    valid_seconds = isinstance(seconds, (int, float)) and not isinstance(seconds, bool) and math.isfinite(seconds) and seconds >= 0
    valid_count = isinstance(count, (int, float)) and not isinstance(count, bool) and math.isfinite(count) and count > 0 and count == int(count)
    derived = 1000 * seconds / count if valid_seconds and valid_count else None
    return dict(seconds=seconds if valid_seconds else None, ms_per_image=derived,
                n_images=int(count) if valid_count else None,
                ms_per_image_derivation="1000 * end_to_end_seconds / n_images; complete CSV-operation average"
                    if derived is not None else "unavailable: valid complete-operation seconds and image count required",
                exporter_per_image_mean_ms=sidecar.get("end_to_end_ms_per_image"),
                timing_scope=sidecar.get("timing_scope"), checkpoint_sha256=sidecar.get("checkpoint_sha256"),
                source=str(source) if sidecar else None)


def figure_gallery(figures, output_dir):
    """Portable Markdown links; JSON keeps absolute paths as source provenance."""
    captions = [
        ("screening", "Screening comparisons", "Short equal-budget runs using the selection seed."),
        ("confirmation", "Full-budget comparisons", "Promoted candidates and baseline using the selection seed."),
        ("learning_curves", "Learning curves", "Training loss and validation PQ; loss scales can differ between methods."),
        ("official_overlap_distributions", "Official overlap-pair distributions", "Every positive-overlap pair; missed objects do not contribute zeros."),
        ("best_gt_distributions", "Best-per-GT overlap distributions", "Each reference filament contributes its best match, including zero for a miss."),
        ("overlap_relationships", "Fragmentation and merging", "Any-positive-overlap counts pooled across annotation sets; read alongside PQ, RQ, TP, FP and FN."),
        ("validation_paired", "Paired validation examples", "Same labelled images for baseline and selected model; highest/lowest scores, changes and a typical case are deliberately selected."),
        ("test_examples", "Representative test predictions", "Unlabelled visual examples; no best/worst accuracy ranking."),
    ]
    lines = ["## Figures", "", "Links stay within this study folder when it is copied into a handoff package.", ""]
    for key, label, caption in captions:
        path = figures.get(key)
        if path:
            relative = Path(path).resolve().relative_to(Path(output_dir).resolve()).as_posix()
            lines.append(f"- [{label}](<{relative}>): {caption}")
        else:
            lines.append(f"- {label}: not generated; saved artifacts may be unavailable or examples disabled.")
    return lines


def build_study(protocol_path, selection_path, output_dir, examples=True, require_complete=False):
    protocol_path, selection_path = resolve_path(protocol_path), resolve_path(selection_path)
    protocol, selection = read_json(protocol_path), read_json(selection_path)
    if protocol["primary_metric"] != "PQ_dataset":
        raise ValueError("This study reports official pooled PQ_dataset as its primary metric")
    output = resolve_path(output_dir); output.mkdir(parents=True, exist_ok=True)
    figures_dir = output / "figures"; figures_dir.mkdir(exist_ok=True)
    split = protocol.get("split_sha256")
    expected = protocol["screen"]["expected_variants"]
    records_dir = resolve_path(protocol["screen"]["records_dir"])
    records = {}
    for path in records_dir.glob("*.json"):
        record = read_json(path)
        records[record["experiment"]] = {**record, "source_record": str(path)}
    screening = []
    for variant in expected:
        record = records.get(variant, {})
        item = dict(variant=variant, seed=protocol["screen"]["seed"],
                    run_dir=record.get("run_dir", records_dir.parent / f"{variant}_seed{protocol['screen']['seed']}"))
        if record.get("source_record"):
            item["experiment_record"] = record["source_record"]
        row = load_run(item, protocol["screen"]["epochs"], split)
        if not (Path(row["run_dir"]) / "summary.json").exists():
            row["status"] = record.get("status", "pending")
        screening.append(row)
    confirmation = [load_run(r, protocol["confirmation"]["epochs"], split) for r in selection["confirmation_runs"]]
    fresh = [load_run(r, protocol["final"]["epochs"], split) for r in selection["fresh_seed_runs"]]
    if selection["selection_seed"] in protocol["final"]["seeds"]:
        raise ValueError("Selection seed must be separate from fresh estimation seeds")
    if any(r["seed"] != protocol["confirmation"]["seed"] for r in confirmation):
        raise ValueError("Confirmation runs must use the declared selection seed")
    if len({r["variant"] for r in confirmation}) != len(confirmation):
        raise ValueError("Confirmation should contain one run per candidate variant")
    if any(r["variant"] != selection["selected_variant"] for r in fresh):
        raise ValueError("Fresh-seed runs must use the selected method")
    aggregate = fresh_seed_statistics(fresh, protocol["final"]["seeds"])
    all_rows = confirmation + fresh
    def find_run(path):
        absolute = str(resolve_path(path))
        found = next((r for r in all_rows if r["run_dir"] == absolute), None)
        if found is None:
            raise ValueError(f"Baseline/representative must be declared in the study runs: {absolute}")
        return found
    baseline, representative = find_run(selection["baseline_run"]), find_run(selection["representative_run"])
    if representative not in fresh or representative["status"] != "completed":
        raise ValueError("Representative model must be a completed fresh-seed run")
    complete = dict(screening_completed=sum(r["status"] == "completed" for r in screening),
                    screening_expected=len(expected), confirmation_completed=sum(r["status"] == "completed" for r in confirmation),
                    confirmation_expected=(1 + protocol["confirmation"]["promote_nonbaseline_count"]
                        if "promote_nonbaseline_count" in protocol["confirmation"] else len(confirmation)),
                    fresh_seeds_completed=aggregate["n_runs"],
                    fresh_seeds_expected=len(protocol["final"]["seeds"]))
    complete["all_training_complete"] = all(complete[f"{k}_completed"] == complete[f"{k}_expected"]
                                            for k in ("screening", "confirmation", "fresh_seeds"))
    if require_complete and not complete["all_training_complete"]:
        raise ValueError("Study still contains pending/partial runs; do not report it as complete")
    # Test timing is from its explicit end-to-end export sidecar, never validation latency.
    test_path = resolve_path(selection.get("test_summary", Path(representative["run_dir"]) / "submission.summary.json"))
    test_summary = read_json(test_path) if test_path.exists() else {}
    if test_summary.get("checkpoint_sha256"):
        weights = Path(representative["run_dir"]) / "best_weights.pt"
        if not weights.exists() or _hash(weights) != test_summary["checkpoint_sha256"]:
            raise ValueError("Test export is stale or lacks its matching current checkpoint")
    if test_summary.get("config") and _prediction_settings(test_summary["config"]) != _prediction_settings(_config(Path(representative["run_dir"]) / "config.yaml")):
        raise ValueError("Test export uses different inference/preprocessing settings")
    representative = deepcopy(representative)
    representative["end_to_end_test"] = summarize_test_timing(test_summary, test_path)
    plt.rcParams.update({"figure.dpi": 130, "font.size": 10})
    figures = dict(screening=_stage_plot(screening, figures_dir / "screening.png", "Screening — one selection seed, equal short budget"),
        confirmation=_stage_plot(confirmation, figures_dir / "confirmation.png", "Full-budget confirmation — selection seed only"),
        learning_curves=_learning_curves(confirmation, fresh, figures_dir / "learning_curves.png"))
    figures.update(_distributions(baseline, representative, figures_dir))
    validation_examples, selected_test = [], []
    figures.update(validation_paired=None, test_examples=None)
    if examples:
        figures["validation_paired"], validation_examples = paired_validation_examples(baseline, representative, figures_dir)
        figures["test_examples"], selected_test = test_examples(representative, selection, test_summary, figures_dir)
    result = dict(schema_version=1,
        protocol=dict(path=str(protocol_path), primary_metric=protocol["primary_metric"], split_sha256=split,
                      selection_seed=selection["selection_seed"], fresh_seeds=protocol["final"]["seeds"]),
        selection={k: selection[k] for k in ("selected_variant", "selection_rule", "representative_run", "baseline_run")},
        screening=screening, confirmation=confirmation, fresh_seed_runs=fresh, fresh_seed_summary=aggregate,
        representative=representative, figures=figures,
        examples=dict(validation=validation_examples, test=selected_test),
        kaggle=dict(status="not_submitted_by_user_choice", score=None),
        limitations=LIMITATIONS + protocol.get("notes", []), completeness=complete)
    result["selection"]["representative_rule"] = protocol["final"].get("representative_rule", "Declared in selection.json")
    result["limitations"].append("The representative checkpoint may be the best fresh seed by validation PQ; "
        "its individual score/plots do not replace the mean over all fresh seeds or establish significance over a single baseline run.")
    for name, rows in (("screening", screening), ("confirmation", confirmation), ("fresh_seed_runs", fresh)):
        _save_csv(rows, output / f"{name}.csv")
    _save_csv([dict(metric=k, **aggregate[k]) for k in METRICS], output / "fresh_seed_summary.csv")
    (output / "summary.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf8")
    lines = ["# Person B staged modelling study", "", "## Study design", "",
        "Screening identifies promising methods, full-budget runs confirm candidates, and fresh seeds "
        "estimate training variability. Seed 42 is used for selection and excluded from the fresh-seed mean.", "",
        f"Selected method: **{selection['selected_variant']}**. {selection['selection_rule']}", "",
        f"Completed screening / confirmation / fresh runs: {complete['screening_completed']}/{complete['screening_expected']}, "
        f"{complete['confirmation_completed']}/{complete['confirmation_expected']}, "
        f"{complete['fresh_seeds_completed']}/{complete['fresh_seeds_expected']}.", "",
    ]
    for title, rows in (("Screening results", screening), ("Full-budget confirmation", confirmation)):
        lines += [f"## {title}", "", "| Variant | Seed | PQ | Epochs completed / planned | Status |",
                  "|---|---:|---:|---:|---|"]
        for row in rows:
            score = f"{row['PQ_dataset']:.5f}" if row.get("PQ_dataset") is not None else "unavailable"
            lines.append(f"| {row['variant']} | {row['seed']} | {score} | "
                         f"{row['epochs_completed']} / {row['epochs_planned']} | {row['status']} |")
        lines.append("")
    lines += ["## Fresh-seed results", "", "| Metric | Mean | Sample SD across runs | Runs |", "|---|---:|---:|---:|"]
    for metric in METRICS:
        value = aggregate[metric]
        mean = f"{value['mean']:.5f}" if value["mean"] is not None else "unavailable"
        std = f"{value['std_across_runs']:.5f}" if value["std_across_runs"] is not None else "unavailable"
        lines.append(f"| {metric} | {mean} | {std} | {value['n']} |")
    lines += ["", "## Representative model and timing", "",
              f"Model: {representative['variant']}, seed {representative['seed']}. "
              f"{result['selection']['representative_rule']}", "",
              "| Quantity | Recorded value |", "|---|---:|"]
    for label, value in [("Parameters", representative.get("parameters")),
                         ("Compact weights MiB", representative.get("weights_MiB")),
                         ("Training-loop seconds", representative.get("training_seconds")),
                         ("Epoch-loop seconds including validation", representative.get("epoch_loop_seconds")),
                         ("Latest training subprocess wall seconds", representative.get("subprocess_wall_seconds")),
                         ("Planned optimizer-update attempts", representative.get("planned_update_attempts")),
                         ("Recorded optimizer-update attempts", representative.get("recorded_update_attempts")),
                         ("Applied optimizer updates", representative.get("optimizer_steps")),
                         ("AMP-skipped optimizer updates", representative.get("amp_skipped_steps")),
                         ("Test export milliseconds per image", representative["end_to_end_test"]["ms_per_image"]),
                         ("Test images", representative["end_to_end_test"]["n_images"])]:
        rendered = f"{value:.3f}" if isinstance(value, (int, float)) else "unavailable"
        lines.append(f"| {label} | {rendered} |")
    lines += ["", "Test timing scope: " + str(representative["end_to_end_test"]["timing_scope"] or "unavailable"),
              "", "Reported test milliseconds per image: " + representative["end_to_end_test"]["ms_per_image_derivation"],
              "", "Training subprocess timing scope: " + str(representative.get("subprocess_wall_scope", "unavailable")),
              "", "Kaggle score: **unavailable — no submission requested**. Test examples have no accuracy labels.",
              "", "## Interpretation and limitations", ""] + [f"- {s}" for s in result["limitations"]]
    lines += [""] + figure_gallery(figures, output)
    zooms = [row for row in validation_examples if row.get("zoom_panel")]
    if zooms:
        lines += ["", "### Validation details", "",
                  "Each panel retains the whole image and marks a crop derived from reference-mask pixels. "
                  "PQ labels refer to the whole observation, not the displayed crop.", ""]
        for row in zooms:
            relative = Path(row["zoom_panel"]).relative_to(output).as_posix()
            lines.append(f"- [{row['observation']}](<{relative}>): {row['reason']}.")
    (output / "study_report.md").write_text("\n".join(lines) + "\n", encoding="utf8")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", default="results/person_b_study/protocol.json")
    parser.add_argument("--selection", default="results/person_b_study/selection.json")
    parser.add_argument("--output", default="results/person_b_study")
    parser.add_argument("--no-examples", action="store_true")
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args(argv)
    result = build_study(args.protocol, args.selection, args.output, not args.no_examples, args.require_complete)
    print(json.dumps(dict(completeness=result["completeness"], fresh_seed_summary=result["fresh_seed_summary"]), indent=2))


if __name__ == "__main__":
    main()
