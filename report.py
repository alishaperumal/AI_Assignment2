"""Create reproducible figures and a concise modelling report from saved runs.

This reads recorded results, never fabricates missing scores, and never ranks
unlabelled test images as quantitatively best or worst.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml


def _load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _training_status(directory, summary):
    """Describe recorded completion without inventing an absent epoch budget."""
    planned = summary.get("epochs_planned")
    config_path = directory / "config.yaml"
    if config_path.exists():
        saved_config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        planned = saved_config.get("train", {}).get("epochs", planned)
    reason = summary.get("stop_reason", "unknown")
    completed = int(summary["epochs_completed"])
    if reason == "invocation_limit":
        status = "preliminary_partial"
    elif reason == "early_stopping":
        status = "completed_early_stopping"
    elif planned is not None and completed < int(planned):
        status = "preliminary_partial"
    elif reason == "completed" or (planned is not None and completed >= int(planned)):
        status = "completed"
    else:
        status = "unknown_completion"
    return dict(epochs_planned=planned, stop_reason=reason, training_status=status)


def _evaluation_provenance(directory, summary):
    """Reject evaluation artifacts from replaced weights; label missing proof."""
    evaluation = directory / "evaluation"
    if not evaluation.exists():
        return "unavailable"
    metadata = evaluation / "summary.json"
    if not metadata.exists():
        return "unverified_missing_evaluation_metadata"
    expected = _load(metadata).get("checkpoint_sha256")
    if not expected:
        return "unverified_missing_checkpoint_hash"
    # Use this run directory, including when a run was moved from its saved path.
    weights = directory / Path(summary.get("weights_checkpoint", "best_weights.pt")).name
    if not weights.is_file():
        return "unverified_missing_current_weights"
    digest = hashlib.sha256()
    with weights.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    if digest.hexdigest() != expected:
        raise ValueError(
            f"Saved evaluation checkpoint hash does not match current weights: {weights}. "
            f"Training may have resumed since evaluation. Rerun evaluate.py --checkpoint "
            f'"{weights}" --output "{evaluation}" --save-predictions before reporting.'
        )
    return "verified_current_weights_sha256"


def create_report(run_dirs, output_dir):
    run_dirs = [Path(d) for d in run_dirs]
    if not run_dirs:
        raise ValueError("Supply at least one saved training run")
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows, histories = [], []
    for directory in run_dirs:
        s = _load(directory / "summary.json")
        if s.get("overfit_images", 0):
            raise ValueError("Diagnostic overfit runs cannot be reported as benchmark results")
        provenance = _evaluation_provenance(directory, s)
        v = s["best_validation"]
        status = _training_status(directory, s)
        rows.append({"run": s["run_name"], "seed": s["seed"], "PQ": v["PQ_dataset"],
                     "SQ": v["SQ_dataset"], "RQ": v["RQ_dataset"],
                     "checkpoint_selection_metric": s.get("primary_metric", "PQ_dataset"),
                     "checkpoint_selection_score": s["best_score"],
                     "iou_matching_threshold": v.get("iou_matching_threshold", 0.5),
                     "foreground_threshold": v.get("foreground_threshold", 0.5),
                     "per_filament_Dice": v.get("filament_dice_mean"),
                     "per_filament_IoU": v.get("filament_iou_mean"),
                     "fragmented_GT_any_overlap": v.get("any_overlap_one_to_many"),
                     "merged_predictions_any_overlap": v.get("any_overlap_many_to_one"),
                     "parameters": s["parameters"],
                     "weights_MiB": s["weights_file_bytes"] / 1024**2,
                     "training_minutes": s["training_seconds"] / 60,
                     "prediction_and_components_ms": v.get("baseline_prediction_and_components_ms_per_image"),
                     "best_epoch": s["best_epoch"], "epochs_completed": s["epochs_completed"],
                     **status,
                     "evaluation_provenance": provenance,
                     "Kaggle_PQ": None, "run_dir": str(directory.resolve())})
        label = s["run_name"]
        if status["training_status"] == "preliminary_partial":
            label += f" (partial: {s['epochs_completed']}/{status['epochs_planned'] or '?'} epochs)"
        histories.append((label, pd.read_csv(directory / "history.csv")))
    frame = pd.DataFrame(rows)
    frame.to_csv(out / "model_comparison.csv", index=False)
    plt.rcParams.update({"figure.dpi": 130, "font.size": 10})
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for label, history in histories:
        axes[0].plot(history.epoch, history.train_loss, label=label)
        score_column = "val_PQ_dataset"
        evaluated = history.dropna(subset=[score_column])
        axes[1].plot(evaluated.epoch, evaluated[score_column], marker="o", markersize=3, label=label)
    axes[0].set(xlabel="Epoch", ylabel="Training loss", title="Learning on training crops")
    axes[1].set(xlabel="Epoch", ylabel="Validation PQ", title="Full validation: pooled annotation sets")
    for ax in axes:
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "learning_curves.png")
    plt.close(fig)

    available = [(Path(d), Path(d) / "evaluation" / "filament_scores.npz") for d in run_dirs]
    available = [(d, p) for d, p in available if p.exists()]
    if available:
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        for directory, path in available:
            with np.load(path) as scores:
                for ax, key in zip(axes, ("dice", "iou")):
                    ax.hist(scores[key], bins=np.linspace(0, 1, 21), alpha=0.45, label=directory.name)
        for ax, title in zip(axes, ("Per-GT-filament Dice", "Per-GT-filament IoU")):
            ax.set(xlabel="Score (missed filaments included as zero)", ylabel="Count", title=title)
            ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(out / "filament_distributions.png")
        plt.close(fig)

    # These samples differ deliberately from best-per-GT values above:
    # every intersecting GT/prediction pair contributes, with no missed-object zeros.
    available_pairs = [(d, d / "evaluation" / "overlap_scores.npz") for d in run_dirs]
    available_pairs = [(d, p) for d, p in available_pairs if p.exists()]
    if available_pairs:
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        for directory, path in available_pairs:
            with np.load(path) as scores:
                for ax, key in zip(axes, ("dice", "iou")):
                    ax.hist(scores[key], bins=np.linspace(0, 1, 21), alpha=0.45, label=directory.name)
        for ax, title in zip(axes, ("All positive-overlap pair Dice", "All positive-overlap pair IoU")):
            ax.set(xlabel="Score (intersecting pairs only)", ylabel="Pair count", title=title)
            ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(out / "official_overlap_distributions.png")
        plt.close(fig)

        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        for directory, path in available_pairs:
            with np.load(path) as scores:
                for ax, key in zip(axes, ("gt_degree", "pred_degree")):
                    counts = np.bincount(scores[key].astype(np.int64))
                    ax.step(np.arange(len(counts)), counts, where="mid", marker="o", label=directory.name)
        for ax, title, xlabel in zip(axes, ("GT fragmentation", "Prediction merging"),
                                    ("Predictions overlapping each GT", "GT objects overlapping each prediction")):
            ax.set(xlabel=xlabel, ylabel="Object count", title=title)
            ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
            ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(out / "overlap_relationships.png")
        plt.close(fig)

    lines = ["# Person B modelling results", "",
             "These figures and tables are generated from saved run records. "
             "PQ pools matches and errors over all annotation sets, following the organizers' "
             "published self-evaluation notebook. The leaderboard uses a separate hidden test set.", ""]
    if any(r["training_status"] == "preliminary_partial" for r in rows):
        lines += ["**Preliminary partial training is included below.** A run stopped by "
                  "`invocation_limit` has only completed part of its intended training. Its current "
                  "score verifies progress; it is not a completed training experiment. Evaluating "
                  "the full validation set does not mean training has finished.", ""]
    lines += ["| Run | Training status | Epochs completed / planned | Stop reason | Evaluation provenance |",
              "|---|---|---:|---|---|"]
    labels = {"preliminary_partial": "**Preliminary partial**",
              "completed": "Completed planned training",
              "completed_early_stopping": "Finished by early stopping",
              "unknown_completion": "Completion not recorded"}
    for r in rows:
        planned = r["epochs_planned"] if r["epochs_planned"] is not None else "unknown"
        lines.append(f"| {r['run']} | {labels[r['training_status']]} | "
                     f"{r['epochs_completed']} / {planned} | {r['stop_reason']} | "
                     f"{r['evaluation_provenance'].replace('_', ' ')} |")
    lines += ["", "| Run | Seed | Validation PQ | SQ | RQ | Weights MiB | Training minutes |",
              "|---|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        lines.append(f"| {r['run']} | {r['seed']} | {r['PQ']:.4f} | {r['SQ']:.4f} | "
                     f"{r['RQ']:.4f} | {r['weights_MiB']:.1f} | {r['training_minutes']:.1f} |")
    lines += ["", "## How to explain the method", "",
              "- The baseline uses a grayscale ResNet34 U-Net and BCE plus Dice loss; "
              "controlled variants may change architecture, loss, or preprocessing.",
              "- The baseline learns from full-resolution crops. The resolution variant downsamples "
              "before cropping/inference; all final masks are evaluated at the original image size.",
              "- Group predicted foreground pixels into instances using connected components.",
              "- Compare one changed setting at a time on the same observation-grouped split.",
              "- By default, choose checkpoints using pooled instance PQ. The table always reports "
              "that PQ; its CSV also records the actual checkpoint-selection metric.", "",
              "## Reporting limits", "",
              "- Kaggle scores are unfilled until an actual submission is scored.",
              "- A single run does not establish run-to-run variability; use experiments.py's seed_summary.csv.",
              "- filament_distributions.png uses each GT filament's best overlap, including missed "
              "objects as zero. official_overlap_distributions.png follows the organizer reference: "
              "every positive-overlap GT/prediction pair contributes, with no missed-object zeros. "
              "These distributions are different diagnostics and must not be combined.",
              "- overlap_relationships.png counts any positive overlap: zero means no contact, "
              "one means one contact, and two or more indicate fragmentation/merging. The same "
              "prediction is compared separately against each annotation set.",
              "- Distribution figures require saved evaluation artifacts. Training-loss scales "
              "can differ between loss variants and do not directly rank model quality.",
              "- Evaluate each run's selected checkpoint into its evaluation directory before "
              "drawing distribution/example figures. The comparison table comes from the training "
              "summary; those figures come from the separately saved evaluation.",
              "- When evaluation metadata contains a checkpoint hash and current weights exist, "
              "the report requires that hash to match the run's current weights. A mismatch "
              "requires rerunning evaluation. Older or synthetic artifacts without provenance "
              "remain readable but are explicitly marked unverified.",
              "- Official matching requires IoU strictly greater than 0.5. The comparison CSV "
              "records the matching and foreground thresholds so changed settings are visible.",
              "- Prediction timing in this table excludes image loading and final CSV encoding. "
              "The submission sidecar records the complete exported pipeline's timing.",
              "- Quantitative best/worst examples must come from labelled validation data. "
              "Test examples are visual demonstrations only.", "",
              "## References", "",
              "- [Assignment evaluation reference](https://www.kaggle.com/code/azimahmadzadeh/self-evaluation-notebook)",
              "- [U-Net](https://arxiv.org/abs/1505.04597)",
              "- [UNet++](https://arxiv.org/abs/1807.10165)",
              "- [Tversky loss](https://arxiv.org/abs/1706.05721)",
              "- [ResNet](https://arxiv.org/abs/1512.03385)",
              "- [Panoptic Segmentation](https://arxiv.org/abs/1801.00868)",
              "- [segmentation_models_pytorch](https://github.com/qubvel-org/segmentation_models.pytorch)", ""]
    (out / "person_b_results.md").write_text("\n".join(lines), encoding="utf-8")
    return frame


def prediction_examples(run_dir, output_dir, count=3):
    """Render best/worst labelled validation observations from saved predictions."""
    import data as D
    from config import load_config
    from skimage.segmentation import find_boundaries

    if count < 1:
        raise ValueError("example count must be positive")
    directory, out = Path(run_dir), Path(output_dir)
    _evaluation_provenance(directory, _load(directory / "summary.json"))
    cfg = load_config(directory / "config.yaml")
    table = pd.read_csv(directory / "evaluation" / "observations.csv")
    if len(table) == 0:
        return
    score_col = "PQ" if "PQ" in table.columns else "PQ_mean"
    id_col = "observation" if "observation" in table.columns else "name"
    selected = pd.concat([table.nlargest(count, score_col).assign(group="Highest observation mean PQ"),
                          table.nsmallest(count, score_col).assign(group="Lowest observation mean PQ")])
    catalog = D.load_catalog(Path(cfg["data"]["cache_dir"]))
    out.mkdir(parents=True, exist_ok=True)
    for group, subset in selected.groupby("group", sort=False):
        fig, axes = plt.subplots(len(subset), 3, figsize=(12, 4 * len(subset)), squeeze=False)
        for row_index, (_, row) in enumerate(subset.iterrows()):
            obs = str(row[id_col])
            entries = catalog[catalog.obs == obs]
            raw = D.load_image(Path(cfg["data"]["data_root"]) / "train/train_images" / entries.file_name.iloc[0])
            path = directory / "evaluation/predictions" / f"{obs}.npz"
            if not path.exists():
                plt.close(fig)
                raise FileNotFoundError(f"Save validation predictions before plotting examples: {path}")
            with np.load(path) as prediction:
                probability, instances = prediction["probability"], prediction["instances"]
            rgb = np.repeat(raw[..., None], 3, axis=2)
            for set_index, sid in enumerate(entries.set_id):
                gt = D.load_instance_map(sid, Path(cfg["data"]["cache_dir"]))
                rgb[find_boundaries(gt > 0)] = [(1, 0.2, 0.2), (0.2, 1, 0.2), (0.3, 0.5, 1)][set_index % 3]
            axes[row_index, 0].imshow(rgb)
            axes[row_index, 1].imshow(probability, vmin=0, vmax=1, cmap="magma")
            axes[row_index, 2].imshow(raw, cmap="gray", vmin=0, vmax=1)
            axes[row_index, 2].contour(instances > 0, levels=[0.5], colors=["cyan"], linewidths=0.6)
            axes[row_index, 0].set_title(f"{obs}\nHuman foreground outlines; mean PQ={row[score_col]:.3f}", fontsize=9)
            axes[row_index, 1].set_title("Predicted filament probability")
            axes[row_index, 2].set_title("Predicted foreground outline")
            for ax in axes[row_index]:
                ax.axis("off")
        fig.suptitle(group + " (labelled validation images)")
        fig.tight_layout()
        fig.savefig(out / ("validation_best.png" if group.startswith("Highest") else "validation_worst.png"))
        plt.close(fig)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs", nargs="+", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--examples", action="store_true", help="Requires evaluation --save-predictions for each run")
    args = p.parse_args(argv)
    frame = create_report(args.runs, args.output)
    if args.examples:
        for directory in args.runs:
            prediction_examples(directory, Path(args.output) / Path(directory).name)
    print(frame.to_string(index=False))


if __name__ == "__main__":
    main()
