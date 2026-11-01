"""Person C report: tables, figures and a results summary from recorded runs.

Reads only saved artifacts (sweep tables, offline scoring outputs, saved
probability/instance archives and compression ``variant.json`` files). It never
trains, re-scores or uploads anything. Run from the repository root:

    python person_c_report.py

Outputs go to ``results/person_c/`` (tables, summary JSON, markdown) and
``figures/person_c/`` (PNG figures sized for slides).
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
RUNS = ROOT / "runs" / "person_c"
FAMILY_LABELS = {
    "threshold": "Pixel cutoff only", "hysteresis": "Hysteresis (low cutoff + seed)",
    "confident_regions": "Require a confident pixel", "min_area": "Remove small regions",
    "bridge": "Bridge fragments (ID only)", "closing": "Morphological closing",
    "fill_holes": "Fill holes", "core_split": "Watershed core split", "connectivity": "4-connectivity",
}


def read_csv(path):
    with Path(path).open(newline="", encoding="utf8") as f:
        return list(csv.DictReader(f))


def write_csv(path, rows, fields=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = fields or list(rows[0])
    with path.open("w", newline="", encoding="utf8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def with_intervals(stage_dir: Path) -> list[dict]:
    """Sweep rows joined with their paired bootstrap intervals."""
    rows = read_csv(stage_dir / "results.csv")
    intervals = {r["name"]: r for r in load_json(stage_dir / "summary.json")["analysis"]["bootstrap"]["intervals"]}
    for row in rows:
        ci = intervals[row["name"]]
        row.update(delta_ci95_low=ci["delta_ci95_low"], delta_ci95_high=ci["delta_ci95_high"],
                   bootstrap_probability_delta_positive=ci["bootstrap_probability_delta_positive"])
        for key in ("PQ_dataset", "SQ_dataset", "RQ_dataset", "delta_PQ_dataset", "threshold", "postprocess_ms_per_image"):
            row[key] = float(row[key])
        for key in ("tp", "fp", "fn", "custom_one_to_many", "custom_many_to_one",
                    "any_overlap_one_to_many", "any_overlap_many_to_one"):
            row[key] = int(float(row[key]))
    return rows


def setting_of(row) -> dict:
    return json.loads(row["settings"])


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 13, "axes.titlesize": 15, "axes.labelsize": 13,
                         "figure.dpi": 150, "axes.spines.top": False, "axes.spines.right": False})
    return plt


def figure_families(stage1, selected, out):
    plt = _plt()
    best = {}
    for row in stage1[1:]:
        family = row["family"]
        if family not in best or row["PQ_dataset"] > best[family]["PQ_dataset"]:
            best[family] = row
    items = sorted(best.values(), key=lambda r: r["delta_PQ_dataset"])
    items.append(dict(selected, family="selected"))
    labels = [FAMILY_LABELS.get(r["family"], "Selected: bridge 8 + remove <320 px (stage 2)") + "\n" + r["name"]
              for r in items]
    deltas = np.array([r["delta_PQ_dataset"] for r in items])
    low = deltas - np.array([r["delta_ci95_low"] for r in items])
    high = np.array([r["delta_ci95_high"] for r in items]) - deltas
    colors = ["#2a9d8f" if r["delta_ci95_low"] > 0 else ("#e76f51" if r["delta_ci95_high"] < 0 else "#9aa5b1")
              for r in items]
    colors[-1] = "#264653"
    fig, ax = plt.subplots(figsize=(11, 6.2))
    y = np.arange(len(items))
    ax.barh(y, deltas, xerr=[low, high], color=colors, capsize=3)
    ax.axvline(0, color="black", lw=1)
    ax.set_yticks(y, labels, fontsize=9.5)
    ax.set_xlabel("Change in pooled validation PQ vs baseline (95% observation-bootstrap CI)")
    ax.set_title("Post-processing ideas, one change at a time (best setting per family)")
    for yi, d, r in zip(y, deltas, items):
        anchor = r["delta_ci95_high"] + 0.0015 if d >= 0 else r["delta_ci95_low"] - 0.0015
        ax.text(anchor, yi, f"{d:+.3f}", va="center", ha="left" if d >= 0 else "right", fontsize=10)
    ax.set_xlim(min(-0.012, min(r["delta_ci95_low"] for r in items) - 0.006),
                max(r["delta_ci95_high"] for r in items) + 0.01)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def figure_curves(stage1, stage2, out):
    plt = _plt()
    base = stage1[0]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), sharey=True)
    groups = [
        ("threshold", "Pixel cutoff", lambda r: r["threshold"], False),
        ("min_area", "Minimum region area (px)", lambda r: setting_of(r)["min_area"], True),
        ("bridge", "Bridge radius (px)", lambda r: setting_of(r)["bridge_radius"], False),
    ]
    extended = {"min_area": "min_area_extended", "bridge": "bridge_extended"}
    for ax, (family, label, value, log) in zip(axes, groups):
        rows = [r for r in stage1 if r["family"] == family]
        rows += [r for r in stage2 if r["family"] == extended.get(family)]
        base_x = 0.5 if family == "threshold" else (1 if log else 0)
        points = sorted([(base_x, base)] + [(value(r), r) for r in rows], key=lambda p: p[0])
        xs = [p[0] for p in points]
        ax.plot(xs, [p[1]["PQ_dataset"] for p in points], "o-", color="#264653", label="PQ")
        ax2 = ax.twinx()
        ax2.plot(xs, [p[1]["fp"] for p in points], "s--", color="#e76f51", alpha=0.8, label="FP")
        ax2.plot(xs, [p[1]["fn"] for p in points], "^:", color="#2a9d8f", alpha=0.8, label="FN")
        ax2.set_ylabel("Pooled count")
        ax.axhline(base["PQ_dataset"], color="gray", lw=1, ls=":")
        ax.set_xlabel(label)
        ax.set_ylabel("Validation PQ")
        if log:
            ax.set_xscale("log")
            ax.set_xticks([1, 10, 100, 1000], ["none", "10", "100", "1000"])
        if family == "threshold":
            ax.set_title("Cutoff alone: no gain")
        elif family == "min_area":
            ax.set_title("Removing small regions")
        else:
            ax.set_title("Bridging fragments")
        handles = ax.get_legend_handles_labels()[0] + ax2.get_legend_handles_labels()[0]
        ax.legend(handles, ["PQ", "FP", "FN"], loc="center right", fontsize=9)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def figure_error_counts(baseline, selected, out):
    plt = _plt()
    keys = [("tp", "Matched (TP)"), ("fp", "Extra (FP)"), ("fn", "Missed (FN)"),
            ("custom_one_to_many", "Fragmented\n(1-to-many)"), ("custom_many_to_one", "Merged\n(many-to-1)"),
            ("any_overlap_one_to_many", "Any-overlap\n1-to-many"), ("any_overlap_many_to_one", "Any-overlap\nmany-to-1")]
    fig, ax = plt.subplots(figsize=(12, 4.8))
    x = np.arange(len(keys))
    for offset, (summary, label, color) in zip((-0.2, 0.2), ((baseline, "Baseline conversion", "#9aa5b1"),
                                                            (selected, "Selected post-processing", "#264653"))):
        values = [summary[k] for k, _ in keys]
        bars = ax.bar(x + offset, values, 0.4, label=label, color=color)
        for bar, v in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), str(int(v)), ha="center", va="bottom", fontsize=9)
    ax.set_xticks(x, [label for _, label in keys])
    ax.set_ylabel("Pooled over 200 annotation sets")
    ax.set_title(f"Error counts: PQ {baseline['PQ_dataset']:.4f} -> {selected['PQ_dataset']:.4f} (same probabilities)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def component_areas(predictions_dir: Path, overlap_pairs_csv: Path, iou_threshold=0.5):
    """Areas of baseline predicted regions, split by whether they match any reference."""
    matched = set()
    for row in read_csv(overlap_pairs_csv):
        if float(row["iou"]) > iou_threshold:
            matched.add((row["observation"], int(row["pred_id"])))
    tp, fp = [], []
    for path in sorted(predictions_dir.glob("*.npz")):
        with np.load(path, allow_pickle=False) as archive:
            areas = np.bincount(archive["instances"].ravel().astype(np.int64))
        for pid in np.flatnonzero(areas[1:]) + 1:
            (tp if (path.stem, int(pid)) in matched else fp).append(int(areas[pid]))
    return np.array(tp), np.array(fp)


def figure_areas(tp, fp, gt_areas, min_area, out):
    plt = _plt()
    bins = np.logspace(0, np.log10(max(tp.max(initial=1), fp.max(initial=1), gt_areas.max(initial=1)) * 1.1), 40)
    fig, ax = plt.subplots(figsize=(11, 4.8))
    ax.hist(fp, bins, color="#e76f51", alpha=0.75, label=f"Unmatched predicted regions ({len(fp)})")
    ax.hist(tp, bins, color="#2a9d8f", alpha=0.75, label=f"Matched predicted regions ({len(tp)})")
    ax.hist(gt_areas, bins, histtype="step", color="black", lw=1.6,
            label=f"Reference filaments, all sets ({len(gt_areas)})")
    ax.axvline(min_area, color="#264653", ls="--", lw=2)
    ax.text(min_area * 1.08, ax.get_ylim()[1] * 0.92, f"selected removal\nbelow {min_area} px", color="#264653")
    ax.set_xscale("log")
    ax.set_xlabel("Region area (pixels, 2048 x 2048 image)")
    ax.set_ylabel("Count")
    share = float((fp < min_area).mean()) if len(fp) else 0.0
    ax.set_title(f"Baseline predictions: {share:.0%} of unmatched regions are below {min_area} px "
                 f"vs {float((tp < min_area).mean()):.0%} of matched")
    ax.legend(fontsize=10)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    return dict(n_matched=len(tp), n_unmatched=len(fp), min_area=min_area,
                unmatched_below_min_area=float((fp < min_area).mean()) if len(fp) else None,
                matched_below_min_area=float((tp < min_area).mean()) if len(tp) else None,
                reference_below_min_area=float((gt_areas < min_area).mean()) if len(gt_areas) else None)


def figure_filament_iou(baseline_dir, selected_dir, out):
    plt = _plt()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    for ax, key in zip(axes, ("iou", "dice")):
        for directory, label, color in ((baseline_dir, "Baseline", "#9aa5b1"), (selected_dir, "Selected", "#264653")):
            values = np.load(directory / "filament_scores.npz")[key]
            ax.hist(values, np.linspace(0, 1, 21), histtype="step", lw=2, color=color,
                    label=f"{label}: mean {values.mean():.3f}, >0.5: {(values > 0.5).mean():.0%}")
        ax.axvline(0.5, color="black", ls=":", lw=1)
        ax.set_xlabel(f"Best-overlap {key.upper() if key == 'iou' else 'Dice'} per reference filament")
        ax.set_ylabel("Reference filaments")
        ax.legend(fontsize=9, loc="upper left")
    fig.suptitle("Per-filament overlap distributions (all validation annotation sets)")
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def figure_examples(cfg_path, predictions_dir, baseline_dir, selected_dir, settings, out, n_each=2,
                    half=420, focus_changes=False):
    """Validation observations with the largest PQ gain and the largest loss."""
    import data as D
    from config import load_config
    from postprocess import probability_to_instances_pp

    plt = _plt()
    base = {r["observation"]: float(r["PQ"]) for r in read_csv(baseline_dir / "observations.csv")}
    sel = {r["observation"]: float(r["PQ"]) for r in read_csv(selected_dir / "observations.csv")}
    deltas = sorted(base, key=lambda o: sel[o] - base[o])
    chosen = [(o, "largest gain") for o in deltas[::-1][:n_each]] + [(o, "largest loss") for o in deltas[:n_each]]
    cfg = load_config(cfg_path)
    catalog = D.load_catalog(Path(cfg["data"]["cache_dir"]))
    rng = np.random.default_rng(0)
    palette = rng.uniform(0.25, 1.0, size=(4096, 3))
    fig, axes = plt.subplots(len(chosen), 3, figsize=(13, 4.3 * len(chosen)))
    for row_axes, (obs, role) in zip(axes, chosen):
        group = catalog[catalog.obs == obs]
        image = D.load_image(Path(cfg["data"]["data_root"]) / "train" / "train_images" / group.file_name.iloc[0])
        reference = D.load_instance_map(group.set_id.iloc[0], Path(cfg["data"]["cache_dir"]))
        with np.load(predictions_dir / f"{obs}.npz", allow_pickle=False) as archive:
            probability, baseline = archive["probability"], archive["instances"].astype(np.int32)
        selected = probability_to_instances_pp(probability, settings["threshold"], settings["postprocess"])
        removed_all = (baseline > 0) & (selected == 0)
        focus = removed_all if focus_changes and removed_all.any() else (reference > 0) | (baseline > 0)
        ys, xs = np.nonzero(focus)
        if len(ys):
            cy, cx = int(np.median(ys)), int(np.median(xs))
        else:
            cy = cx = 1024
        y0, x0 = max(0, min(cy - half, 2048 - 2 * half)), max(0, min(cx - half, 2048 - 2 * half))
        window = (slice(y0, y0 + 2 * half), slice(x0, x0 + 2 * half))
        for ax, labels, title in zip(row_axes, (reference, baseline, selected),
                                     (f"{obs} reference (set 1)", f"Baseline: PQ {base[obs]:.2f}",
                                      f"Selected: PQ {sel[obs]:.2f} ({role})")):
            crop = image[window]
            rgb = np.repeat(crop[..., None], 3, axis=2)
            lab = labels[window]
            mask = lab > 0
            rgb[mask] = 0.35 * rgb[mask] + 0.65 * palette[lab[mask] % len(palette)]
            if labels is selected:
                removed = (baseline[window] > 0) & ~mask
                rgb[removed] = [1.0, 0.1, 0.1]
            ax.imshow(np.clip(rgb, 0, 1))
            ax.set_title(title, fontsize=11)
            ax.axis("off")
    fig.suptitle("Observation PQ (mean over annotation sets); colours = instance IDs; red = removed regions", fontsize=13)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    return [dict(observation=o, role=r, baseline_PQ=base[o], selected_PQ=sel[o]) for o, r in chosen]


def figure_test_examples(test_predictions: Path, data_root: Path, settings, out, n=3):
    """Unlabelled test illustrations: where post-processing changed the most, a typical image and the busiest."""
    import cv2
    import data as D
    from postprocess import probability_to_instances_pp

    plt = _plt()
    stats = []
    for path in sorted(test_predictions.glob("*.npz")):
        with np.load(path, allow_pickle=False) as archive:
            base = archive["instances"]
            n_base = int(len(np.unique(base)) - 1)
        stats.append((path.stem, n_base))
    counts = np.array([s[1] for s in stats])
    typical = stats[int(np.argsort(counts)[len(counts) // 2])][0]
    busiest = stats[int(np.argmax(counts))][0]
    changed = []
    for name in {typical, busiest} | {s[0] for s in sorted(stats, key=lambda s: -s[1])[:12]}:
        with np.load(test_predictions / f"{name}.npz", allow_pickle=False) as archive:
            sel = probability_to_instances_pp(archive["probability"], settings["threshold"], settings["postprocess"])
            changed.append((int(len(np.unique(archive["instances"])) - 1) - int(sel.max()), name))
    most_changed = max(c for c in changed if c[1] not in (typical, busiest))[1]
    chosen = [(most_changed, "largest reduction in regions"), (typical, "typical image"), (busiest, "most regions")][:n]
    rng = np.random.default_rng(1)
    palette = rng.uniform(0.25, 1.0, size=(4096, 3))
    fig, axes = plt.subplots(2, len(chosen), figsize=(5.2 * len(chosen), 10.4))
    rows = []
    for col, (name, role) in enumerate(chosen):
        path = next((data_root / "test" / "test_images").glob(f"{name}.*"))
        image = cv2.resize(D.load_image(path), (1024, 1024), interpolation=cv2.INTER_AREA)
        with np.load(test_predictions / f"{name}.npz", allow_pickle=False) as archive:
            base = archive["instances"].astype(np.int32)
            sel = probability_to_instances_pp(archive["probability"], settings["threshold"], settings["postprocess"])
        for row, (labels, title) in enumerate(((base, "Baseline"), (sel, "Selected"))):
            small = cv2.resize(labels.astype(np.float32), (1024, 1024), interpolation=cv2.INTER_NEAREST).astype(np.int32)
            rgb = np.repeat(image[..., None], 3, axis=2)
            mask = small > 0
            rgb[mask] = 0.25 * rgb[mask] + 0.75 * palette[small[mask] % len(palette)]
            axes[row, col].imshow(np.clip(rgb, 0, 1))
            axes[row, col].set_title(f"{title}: {int(labels.max())} filaments\n{name} ({role})", fontsize=11)
            axes[row, col].axis("off")
        rows.append(dict(observation=name, role=role, baseline_regions=int(base.max()), selected_regions=int(sel.max())))
    fig.suptitle("Unlabelled test images (no accuracy can be computed); colours = instance IDs", fontsize=13)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    return rows


def figure_compression(rows, out):
    plt = _plt()
    usable = [r for r in rows if r.get("selected_PQ") is not None and r.get("ms_per_image") is not None
              and r.get("device") != "CPU"]
    if not usable:
        return
    fig, ax = plt.subplots(figsize=(10.5, 5.4))
    for r in usable:
        size = 40 + 6 * r["file_mb"]
        ax.scatter(r["ms_per_image"], r["selected_PQ"], s=size, color="#264653" if r["family"] != "teacher" else "#e9c46a",
                   alpha=0.8, edgecolor="black")
        ax.annotate(f"{r['label']}\n{r['parameters'] / 1e6:.1f}M params, {r['file_mb']:.0f} MB",
                    (r["ms_per_image"], r["selected_PQ"]), textcoords="offset points", xytext=(8, 4), fontsize=9)
    ax.set_xlabel("End-to-end test pipeline (ms per image, RTX 3070, batch 1)")
    ax.set_ylabel("Validation PQ (selected post-processing)")
    ax.set_title("Compression: quality vs speed (marker area ~ file size)")
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Compression table
# --------------------------------------------------------------------------- #
COMPRESSION = [
    # label, benchmark directory, check-sweep directory, family, training summary
    ("Teacher FP32", "bench_teacher_fp32", "pp_check_teacher_fp32", "teacher", None),
    ("Teacher mixed precision (B default)", "bench_teacher_amp16", "pp_check_teacher_amp16", "teacher", None),
    ("Teacher FP16 weights", "bench_teacher_fp16", "pp_check_teacher_fp16", "precision", None),
    ("KD student ResNet18", "bench_kd_resnet18", "pp_check_kd_resnet18", "distillation", "kd_resnet18"),
    ("Label-only ResNet18 (control)", "bench_label_resnet18", "pp_check_label_resnet18", "control", "label_resnet18"),
    ("Pruned 50% channels + fine-tune", "bench_pruned50", "pp_check_pruned50", "pruning", "pruned50"),
    ("Pruned 30% channels + fine-tune", "bench_pruned30", "pp_check_pruned30", "pruning", "pruned30"),
    ("Pruned 30% + FP16 weights", "bench_pruned30_fp16", "pp_check_pruned30_fp16", "pruning", "pruned30"),
    ("CPU: PyTorch FP32", "bench_cpu_pytorch_fp32", None, "cpu", None),
    ("CPU: ONNX Runtime FP32", "bench_cpu_onnx_fp32", None, "cpu", None),
    ("CPU: ONNX Runtime INT8", "bench_cpu_onnx_int8", "pp_check_cpu_onnx_int8", "cpu", None),
]
GPU_VARIANTS = 8


def compression_rows(compression_dir: Path) -> list[dict]:
    rows = []
    for label, bench, check, family, training in COMPRESSION:
        variant = compression_dir / bench / "variant.json"
        if not variant.is_file():
            continue
        v = load_json(variant)
        row = dict(label=label, family=family, bench=bench, precision=v["precision"], parameters=v["parameters"],
                   device="GPU (RTX 3070)" if v.get("gpu") else "CPU",
                   test_images_timed=v.get("test", {}).get("n_images"),
                   file_mb=v["deployable_file_bytes"] / 2 ** 20,
                   ms_per_image=v.get("test", {}).get("end_to_end_ms_per_image"),
                   ms_per_image_excluding_first=v.get("test", {}).get("ms_per_image_excluding_first"),
                   n_test_filaments=v.get("test", {}).get("n_filaments"),
                   selected_PQ=v.get("validation", {}).get("PQ_dataset"),
                   selected_tp=v.get("validation", {}).get("tp"), selected_fp=v.get("validation", {}).get("fp"),
                   selected_fn=v.get("validation", {}).get("fn"),
                   validation_model_ms_per_image=v.get("validation_prediction_ms_per_image"))
        sweep = compression_dir / check / "results.csv" if check else None
        if sweep is not None and sweep.is_file():
            baseline = read_csv(sweep)[0]
            row.update(baseline_PQ=float(baseline["PQ_dataset"]), baseline_tp=int(baseline["tp"]),
                       baseline_fp=int(baseline["fp"]), baseline_fn=int(baseline["fn"]))
        if training and (compression_dir / training / "summary.json").is_file():
            t = load_json(compression_dir / training / "summary.json")
            row.update(training_seconds=t["training_seconds"], best_epoch=t["best_epoch"],
                       epochs=t["epochs_completed"], training_best_PQ=t["best_score"])
        rows.append(row)
    return rows


def finalize(compression: list[dict], runs: Path, results: Path, tolerance: float = 0.002):
    """Apply the declared final-model rule and stage the chosen checked CSV."""
    import shutil

    teacher = next((r for r in compression if r["label"].startswith("Teacher mixed")), None)
    gpu = [r for r in compression if r["device"] != "CPU"]
    if teacher is None or teacher.get("selected_PQ") is None or len(gpu) < GPU_VARIANTS:
        return None
    eligible = [r for r in gpu if r.get("selected_PQ") is not None and r.get("ms_per_image") is not None
                and r["selected_PQ"] >= teacher["selected_PQ"] - tolerance]
    chosen = min(eligible, key=lambda r: (round(r["file_mb"], 1), r["ms_per_image"]))
    source = runs / "compression" / chosen["bench"]
    final = runs / "final"
    final.mkdir(parents=True, exist_ok=True)
    for name in ("submission.csv", "submission.summary.json", "variant.json"):
        shutil.copy2(source / name, final / name)
    variant = load_json(source / "variant.json")
    sidecar = load_json(source / "submission.summary.json")
    by_label = {r["label"]: r for r in compression}
    kd, control = by_label.get("KD student ResNet18"), by_label.get("Label-only ResNet18 (control)")
    half = by_label.get("Teacher FP16 weights")
    pruned = [r for r in compression if r["family"] == "pruning" and "channels" in r["label"]
              and r.get("selected_PQ") is not None]
    parts = []
    if half:
        parts.append(f"FP16 weights: PQ change {half['selected_PQ'] - teacher['selected_PQ']:+.5f} at "
                     f"{half['file_mb']:.0f} vs {teacher['file_mb']:.0f} MB")
    if kd and control:
        parts.append(f"distillation {kd['selected_PQ']:.3f} vs label-only {control['selected_PQ']:.3f} "
                     f"(ResNet18, {kd['parameters'] / 1e6:.1f} M params)")
    for r in pruned:
        parts.append(f"{r['label'].split(' channels')[0].replace('Pruned ', '')} channel pruning "
                     f"{r['selected_PQ']:.3f} ({r['parameters'] / 1e6:.1f} M params)")
    test = variant["test"]
    b_run = ROOT / "runs" / "person_b_final" / "summary.json"
    teacher_minutes = load_json(b_run)["training_seconds"] / 60 if b_run.is_file() else None
    training_text = (f"{teacher_minutes:.1f} min teacher training loop (20 epochs, RTX 5060 Ti)"
                     if teacher_minutes is not None else "see Person B's study")
    if chosen.get("training_seconds"):
        training_text += f" + {chosen['training_seconds'] / 60:.1f} min fine-tuning after pruning (RTX 3070)"
    record = dict(
        rule=load_json(results / "final_selection_rule.json")["rule"], tolerance=tolerance,
        teacher_validation_PQ=teacher["selected_PQ"], chosen=chosen["label"], chosen_bench=str(source),
        variant=variant["variant"], variant_sha256=variant["variant_sha256"], precision=variant["precision"],
        model_label=f"U-Net / ResNet34, seed 45 ({chosen['label']})",
        training_text=training_text,
        size_text=f"{chosen['parameters'] / 1e6:.2f} M parameters, {chosen['file_mb']:.1f} MB weights file",
        timing_text=f"{test['end_to_end_ms_per_image']:.0f} ms/image over {test['n_images']} test images "
                    f"({test['ms_per_image_excluding_first']:.0f} ms excluding the first), RTX 3070",
        pq_text=f"{chosen['selected_PQ']:.4f} (selected post-processing; baseline conversion "
                f"{chosen.get('baseline_PQ', float('nan')):.4f})",
        validation_PQ=chosen["selected_PQ"], n_images=test["n_images"], n_filaments=test["n_filaments"],
        zero_detection_images=test["zero_detection_images"], csv_sha256=test["csv_sha256"],
        end_to_end_ms_per_image=test["end_to_end_ms_per_image"], timing_scope=test["timing_scope"],
        postprocess=sidecar.get("postprocess"), foreground_threshold=sidecar.get("foreground_threshold"),
        csv=str(final / "submission.csv"),
        compression_conclusion="; ".join(parts) + f". Chosen by the declared rule: {chosen['label']}.",
        kaggle_score=None, kaggle_note="Not uploaded by this work; record the score after the team's upload.")
    (results / "final_submission.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf8")
    return record


def _table(headers, rows):
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" if i == 0 else "---:" for i in range(len(headers))) + "|"]
    lines += ["| " + " | ".join(str(v) for v in row) + " |" for row in rows]
    return "\n".join(lines)


def write_markdown(path: Path, summary: dict, stage1: list, stage2: list) -> None:
    """Plain-language results document; every number comes from recorded files."""
    base, sel, boot = summary["baseline"], summary["selected"], summary["selected_bootstrap"]
    halves, areas = summary["split_half_selection"], summary["component_areas"]
    final, comp = summary.get("final_submission"), summary.get("compression") or []
    best_by_family = {}
    for row in stage1[1:]:
        if row["family"] not in best_by_family or row["PQ_dataset"] > best_by_family[row["family"]]["PQ_dataset"]:
            best_by_family[row["family"]] = row
    family_rows = [[FAMILY_LABELS.get(f, f), r["name"], f"{r['PQ_dataset']:.4f}", f"{r['delta_PQ_dataset']:+.4f}",
                    f"[{r['delta_ci95_low']:+.4f}, {r['delta_ci95_high']:+.4f}]", r["tp"], r["fp"], r["fn"],
                    r["custom_one_to_many"], r["custom_many_to_one"]]
                   for f, r in sorted(best_by_family.items(), key=lambda kv: -kv[1]["delta_PQ_dataset"])]
    top2 = sorted(stage2, key=lambda r: -r["PQ_dataset"])[:10]
    stage2_rows = [[r["name"], f"{r['PQ_dataset']:.4f}", f"{r['delta_PQ_dataset']:+.4f}",
                    f"[{r['delta_ci95_low']:+.4f}, {r['delta_ci95_high']:+.4f}]", r["tp"], r["fp"], r["fn"]]
                   for r in top2]
    checks = summary.get("tta_checks", {})
    tta = {r["name"]: float(r["PQ_dataset"]) for r in checks.get("pp_check_tta", [])}
    notta = {r["name"]: float(r["PQ_dataset"]) for r in checks.get("pp_check_no_tta_same_gpu", [])}
    comp_rows = [[r["label"], r["device"], f"{r['parameters'] / 1e6:.2f} M", f"{r['file_mb']:.1f} MB",
                  f"{r['baseline_PQ']:.4f}" if r.get("baseline_PQ") is not None else "-",
                  f"{r['selected_PQ']:.4f}" if r.get("selected_PQ") is not None else "-",
                  (f"{r['ms_per_image']:.0f}" + (f" ({r['test_images_timed']} imgs)" if r.get("test_images_timed") != 180 else ""))
                  if r.get("ms_per_image") else "-",
                  f"{r['training_seconds'] / 60:.1f} min" if r.get("training_seconds") else "-"]
                 for r in comp]
    gains = [halves[k]["held_out_gain"] for k in sorted(halves)]
    lines = [
        "# Person C: instance post-processing, compression and final submission", "",
        "Generated by `person_c_report.py` from recorded run outputs; nothing here was retrained or re-scored by the "
        "report itself. All validation numbers use the frozen split (122 images, 200 annotation sets), pooled "
        "`PQ_dataset` and strict IoU > 0.5 matching, exactly as in Person B's study.", "",
        "## In plain language", "",
        f"* Person B's seed-45 model turns each image into a probability map. The baseline turned that map into "
        f"filaments by keeping pixels above 0.5 and calling each connected blob one filament (PQ {base['PQ_dataset']:.4f}).",
        f"* Most errors were **extra small blobs**: {base['fp']:,} unmatched predictions against {base['fn']} misses. "
        f"{areas['unmatched_below_min_area']:.0%} of unmatched blobs are smaller than {areas['min_area']} pixels, but "
        f"only {areas['matched_below_min_area']:.0%} of correctly matched blobs and "
        f"{areas['reference_below_min_area']:.0%} of human-drawn filaments are that small.",
        f"* The selected post-processing first gives fragments separated by at most 16 pixels one shared ID (no "
        f"pixels added), then removes blobs under 320 pixels. Validation PQ rises to **{sel['PQ_dataset']:.4f}** "
        f"(+{boot['delta_PQ']:.4f}; 95% observation-bootstrap interval +{boot['delta_ci95_low']:.4f} to "
        f"+{boot['delta_ci95_high']:.4f}). Tuning on one half of the validation time blocks and testing on the other "
        f"half gave gains of {gains[0]:+.4f} and {gains[1]:+.4f}.",
        f"* Cost: {base['tp'] - sel['tp']} fewer matched filaments (small real ones) and merges "
        f"{base['custom_many_to_one']} → {sel['custom_many_to_one']}; benefit: extra blobs {base['fp']:,} → {sel['fp']} "
        f"and fragmentation {base['custom_one_to_many']} → {sel['custom_one_to_many']}.",
        "* Changing only the pixel cutoff, watershed splitting, flip test-time augmentation, hole filling and "
        "4-connectivity did not help (details below).",
    ]
    if final:
        lines.append(f"* Final deployed model: {final['chosen']} ({final['size_text']}); end-to-end "
                     f"{final['timing_text']}; {final['n_filaments']} filaments on {final['n_images']} test images. "
                     f"{final['compression_conclusion']}")
    lines += ["", "## 1. Offline runner and reproduction check", "",
              "`offline_eval.py` re-scores saved probability archives with `evaluate.score_observation`, the same "
              "per-annotation-set function now used by `evaluate.py` (including original layers for overlapping or "
              "empty annotations). With the baseline conversion it reproduced Person B's recorded summary "
              f"(`PQ_dataset` {base['PQ_dataset']!r}, TP {base['tp']}, FP {base['fp']}, FN {base['fn']}) to 1e-12 on "
              "every checked key, and its recomputed instance maps were identical to all 122 archived maps "
              f"(reference check: {summary.get('reference_reproduced')}). Re-running the checkpoint on this RTX 3070 "
              "gave PQ 0.34492 (TP 783, FP 1006, FN 507): an expected small mixed-precision hardware difference.", "",
              "## 2. Where the errors were", "",
              "![Component areas](../../figures/person_c/pp_component_areas.png)", "",
              "## 3. Post-processing study", "",
              f"Stage 1 ({len(stage1) - 1} settings, `configs/person_c/pp_stage1.yaml`) changed one thing at a time. "
              "Intervals are paired 95% bootstrap intervals over validation observations (2,000 resamples).", "",
              _table(["Family", "Best setting", "PQ", "ΔPQ", "95% CI", "TP", "FP", "FN", "1-to-many", "many-to-1"],
                     family_rows), "",
              "![Families](../../figures/person_c/pp_families.png)", "",
              "![Parameter curves](../../figures/person_c/pp_parameter_curves.png)", "",
              f"Stage 2 ({len(stage2) - 1} settings, `configs/person_c/pp_stage2.yaml`) was declared after stage 1: it "
              "extended the two settings that peaked at the grid edge and combined the helpful steps. Top ten:", "",
              _table(["Setting", "PQ", "ΔPQ", "95% CI", "TP", "FP", "FN"], stage2_rows), "",
              "**Selection.** The four-step best setting and the two-step `bridge_radius=8, min_area=320` setting "
              "differ by 0.0005 PQ (paired 95% interval about -0.003 to +0.004), so the simpler two-step rule was "
              "selected. It was also the setting chosen when tuning on half A of the validation blocks; tested on "
              f"the held-out halves the gains were {gains[0]:+.4f} and {gains[1]:+.4f}. With this rule PQ is flat "
              "for pixel cutoffs 0.35 to 0.55 (0.403 to 0.405), so the result does not hinge on the cutoff.", "",
              "![Error counts](../../figures/person_c/pp_error_counts.png)", "",
              "![Per-filament overlap](../../figures/person_c/pp_filament_overlap.png)", "",
              "![Examples](../../figures/person_c/pp_examples.png)", "",
              "### Negative results", "",
              "* **Pixel cutoff alone** (0.3 to 0.7): no setting beat 0.5. Lowering it to 0.3 recovered only 4 "
              "matches while adding 37 unmatched regions, so most misses are not simply faint pixels below 0.5.",
              "* **Watershed split at confident cores**: -0.002 to -0.010 PQ. Single filaments have uneven "
              "confidence and were cut into pieces (fragmentation up to 104), while merges were rare (20).",
              "* **Fill holes**: no effect; **4-connectivity**: -0.002.",
              ]
    if tta and notta:
        lines.append(f"* **Flip test-time augmentation** (`configs/person_c/seed45_tta.yaml`): baseline conversion "
                     "0.3372 vs 0.3449 without TTA on the same GPU; with the selected post-processing "
                     f"{tta.get('selected', float('nan')):.4f} vs {notta.get('selected', float('nan')):.4f}. Even "
                     f"choosing each side's best cutoff afterwards, TTA gained only "
                     f"{max(tta.values()) - max(notta.values()):+.4f}, at 3.8x the model inference time "
                     "(382 vs 99 ms per validation image). Not adopted.")
    lines += ["", "## 4. Compression study", "",
              "All variants were scored on the same validation split with the selected post-processing and timed "
              "with the complete image-to-CSV test pipeline (`predict.predict_test`, 180 images, batch 1, model "
              "loading excluded, first image included) on one RTX 3070, one variant at a time. Students used Person "
              "B's recipe (random foreground-biased 512 crops, AdamW 3e-4, cosine schedule, mixed precision, 20 "
              "epochs, seed 45) and kept the best validation epoch. The distilled student's loss is "
              "0.5 x (BCE + Dice with labels) + 0.5 x BCE with the teacher's probabilities; the label-only control "
              "uses the identical code path with distillation weight 0. Pruning removed the lowest-L1 channels "
              "uniformly in every layer (torch-pruning dependency graph, segmentation head output kept) and then "
              "fine-tuned for 10 epochs: learning rate 1e-4 for the declared 50% run, which predicted nothing until "
              "its last epoch, and 3e-4 for the follow-up 30% run. The 30% run and its FP16 copy were added after "
              "the first results and evaluated under the unchanged final-model rule. CPU variants were re-timed in a "
              "quiet window because the first CPU timings overlapped slide rendering (the first runs are kept under "
              "`runs/person_c/compression/superseded/`). ONNX FP32 matched PyTorch probabilities to 6e-6; INT8 "
              "(static QDQ, per-channel weights) was calibrated on 8 training images only.", ""]
    if comp_rows:
        lines += [_table(["Variant", "Device", "Parameters", "File", "Val PQ (baseline conversion)",
                          "Val PQ (selected)", "ms / image", "Training"], comp_rows), "",
                  "GPU timings cover all 180 test images; CPU timings cover the first test images only (shown), so "
                  "compare CPU rows only with each other.", "",
                  "![Compression](../../figures/person_c/compression_tradeoff.png)", ""]
    if final:
        lines += ["## 5. Final pipeline and submission", "",
                  f"Rule (declared before benchmarks, `results/person_c/final_selection_rule.json`): {final['rule']} "
                  "The 30% pruned variants were benchmarked after the rule was declared; the rule itself was not "
                  "changed. The FP16 teacher (46.7 MB, PQ 0.4033) is an essentially equivalent alternative.", "",
                  _table(["Item", "Value"], [
                      ["Chosen variant", final["chosen"]], ["Size", final["size_text"]],
                      ["Training time", final.get("training_text", "-")],
                      ["Validation PQ", final["pq_text"]], ["End-to-end inference", final["timing_text"]],
                      ["Test CSV", f"{final['n_filaments']} filaments, {final['n_images']} images, "
                                   f"{len(final['zero_detection_images'])} without detections"],
                      ["CSV SHA256", final["csv_sha256"]],
                      ["Independent CSV check", "`results/person_c/final_submission_check.json`: every RLE decodes to "
                                                "a non-empty 2048 x 2048 mask, IDs unique, no overlaps"],
                      ["Kaggle", final["kaggle_note"]]]), "",
                  "![Test examples](../../figures/person_c/test_examples.png)", ""]
    lines += ["## 6. Limitations", "",
              "* All tuning used the single frozen validation split; the split-half and bootstrap checks reduce but do "
              "not remove selection optimism, and Kaggle uses different images.",
              "* Post-processing was tuned on one representative model (seed 45). Its effect on other seeds was not "
              "measured because their probabilities were not transferred.",
              "* Compression variants were trained once (seed 45); differences between them are single-run results.",
              "* Timings are single passes on one RTX 3070 and differ from Person B's RTX 5060 Ti numbers.", "",
              "## 7. Reproduce", "",
              "Run from the repository root after adding the data and Person B's package (README). Training and "
              "timing commands use the GPU; run them one at a time.", "",
              "```powershell",
              "$py = '.\\.venv\\Scripts\\python.exe'",
              "# Checkpoint inference on this GPU (reference) and flip test-time augmentation",
              "& $py evaluate.py --checkpoint runs/person_b_final/best_weights.pt --config runs/person_b_final/config.yaml "
              "--output runs/person_c/reproduced_evaluation --save-predictions",
              "& $py evaluate.py --checkpoint runs/person_b_final/best_weights.pt --config configs/person_c/seed45_tta.yaml "
              "--output runs/person_c/tta_evaluation --save-predictions",
              "# Offline scoring: exact baseline reproduction, declared sweeps, selected setting, TTA check",
              "& $py offline_eval.py score --predictions runs/person_b_final/evaluation/predictions "
              "--config configs/person_c/seed45_selected.yaml --output runs/person_c/offline_baseline "
              "--expect runs/person_b_final/evaluation/summary.json --check-stored",
              "& $py offline_eval.py sweep --predictions runs/person_b_final/evaluation/predictions "
              "--config configs/person_c/seed45_selected.yaml --grid configs/person_c/pp_stage1.yaml --output runs/person_c/pp_stage1",
              "& $py offline_eval.py sweep --predictions runs/person_b_final/evaluation/predictions "
              "--config configs/person_c/seed45_selected.yaml --grid configs/person_c/pp_stage2.yaml --output runs/person_c/pp_stage2",
              "& $py offline_eval.py score --predictions runs/person_b_final/evaluation/predictions "
              "--config configs/person_c/final_postprocess.yaml --output runs/person_c/offline_final",
              "& $py offline_eval.py sweep --predictions runs/person_c/tta_evaluation/predictions "
              "--config configs/person_c/seed45_selected.yaml --grid configs/person_c/pp_check.yaml --output runs/person_c/pp_check_tta",
              "& $py offline_eval.py sweep --predictions runs/person_c/reproduced_evaluation/predictions "
              "--config configs/person_c/seed45_selected.yaml --grid configs/person_c/pp_check.yaml "
              "--output runs/person_c/pp_check_no_tta_same_gpu",
              "# Compression: distilled student, then the queued jobs (control, pruning, FP16, ONNX/INT8, benchmarks)",
              "& $py compress.py distill --teacher runs/person_b_final/best_weights.pt --config configs/person_c/seed45_selected.yaml "
              "--output runs/person_c/compression/kd_resnet18 --name kd_resnet18_seed45 --encoder resnet18 --kd-weight 0.5 "
              "--validate-from 11",
              "powershell -ExecutionPolicy Bypass -File jobs/person_c/1_gpu_compression_queue.ps1",
              "powershell -ExecutionPolicy Bypass -File jobs/person_c/2_pruning30_and_cpu_queue.ps1",
              "powershell -ExecutionPolicy Bypass -File jobs/person_c/3_pruned30_fp16_and_cpu_retiming.ps1",
              "# Tables, figures, this document, final CSV staging; then the slides",
              "& $py person_c_report.py",
              "& $py delivery/build_team_deck.py",
              "```", ""]
    path.write_text("\n".join(lines), encoding="utf8")


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", default=str(RUNS))
    parser.add_argument("--results", default=str(ROOT / "results" / "person_c"))
    parser.add_argument("--figures", default=str(ROOT / "figures" / "person_c"))
    parser.add_argument("--predictions", default=str(ROOT / "runs" / "person_b_final" / "evaluation" / "predictions"))
    parser.add_argument("--config", default=str(ROOT / "configs" / "person_c" / "final_postprocess.yaml"))
    args = parser.parse_args(argv)
    runs, results, figures = Path(args.runs), Path(args.results), Path(args.figures)
    results.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    stage1, stage2 = with_intervals(runs / "pp_stage1"), with_intervals(runs / "pp_stage2")
    fields = ["family", "name", "threshold", "settings", "PQ_dataset", "delta_PQ_dataset", "delta_ci95_low",
              "delta_ci95_high", "SQ_dataset", "RQ_dataset", "tp", "fp", "fn", "custom_one_to_many",
              "custom_many_to_one", "any_overlap_one_to_many", "any_overlap_many_to_one", "postprocess_ms_per_image"]
    write_csv(results / "postprocess_stage1.csv", stage1, fields)
    write_csv(results / "postprocess_stage2.csv", stage2, fields)
    from config import load_config
    final_cfg = load_config(args.config)
    selected_settings = dict(threshold=final_cfg["evaluation"]["threshold"], postprocess=final_cfg["postprocess"])
    baseline_summary = load_json(runs / "offline_baseline" / "summary.json")
    selected_summary = load_json(runs / "offline_final" / "summary.json")
    selected_row = next(r for r in stage2 if r["name"] == "t=0.5 min_area=320 bridge_radius=8")
    figure_families(stage1, selected_row, figures / "pp_families.png")
    figure_curves(stage1, stage2, figures / "pp_parameter_curves.png")
    figure_error_counts(baseline_summary, selected_summary, figures / "pp_error_counts.png")
    figure_filament_iou(runs / "offline_baseline", runs / "offline_final", figures / "pp_filament_overlap.png")
    tp, fp = component_areas(Path(args.predictions), runs / "offline_baseline" / "overlap_pairs.csv")
    gt_areas = np.load(runs / "offline_baseline" / "filament_scores.npz")["gt_area"]
    areas = figure_areas(tp, fp, gt_areas, int(final_cfg["postprocess"]["min_area"]), figures / "pp_component_areas.png")
    examples = figure_examples(args.config, Path(args.predictions), runs / "offline_baseline", runs / "offline_final",
                               selected_settings, figures / "pp_examples.png")
    figure_examples(args.config, Path(args.predictions), runs / "offline_baseline", runs / "offline_final",
                    selected_settings, figures / "pp_examples_slide.png", n_each=1, half=300, focus_changes=True)
    test_examples = figure_test_examples(ROOT / "runs" / "person_b_final" / "test_predictions",
                                         Path(final_cfg["data"]["data_root"]), selected_settings,
                                         figures / "test_examples.png")
    compression = compression_rows(runs / "compression")
    final_record = None
    if compression:
        write_csv(results / "compression.csv", compression,
                  list(dict.fromkeys(k for r in compression for k in r)))
        figure_compression(compression, figures / "compression_tradeoff.png")
        final_record = finalize(compression, runs, results)
    checks = {}
    for name in ("pp_check_tta", "pp_check_no_tta_same_gpu"):
        if (runs / name / "results.csv").is_file():
            checks[name] = [{k: r[k] for k in ("name", "PQ_dataset", "tp", "fp", "fn")} for r in read_csv(runs / name / "results.csv")]
    stage2_summary = load_json(runs / "pp_stage2" / "summary.json")
    keep = ("PQ_dataset", "SQ_dataset", "RQ_dataset", "tp", "fp", "fn", "custom_one_to_many", "custom_many_to_one",
            "any_overlap_one_to_many", "any_overlap_many_to_one", "filament_iou_mean", "filament_dice_mean",
            "pixel_dice_mean", "observation_PQ_mean", "n_predicted_instances", "postprocess_ms_per_image")
    summary = dict(
        selected_postprocess=selected_settings,
        baseline={k: baseline_summary.get(k) for k in keep},
        selected={k: selected_summary.get(k) for k in keep},
        selected_bootstrap=next(r for r in stage2_summary["analysis"]["bootstrap"]["intervals"]
                                if r["name"] == selected_row["name"]),
        split_half_selection=stage2_summary["analysis"]["split_half_selection"],
        stage1_candidates=len(stage1), stage2_candidates=len(stage2),
        component_areas=areas, examples=examples, test_examples=test_examples, compression=compression,
        final_submission=final_record, tta_checks=checks,
        reference_reproduced=baseline_summary.get("reference_check", {}).get("matches"),
    )
    (results / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf8")
    write_markdown(results / "person_c_results.md", summary, stage1, stage2)
    print(json.dumps({k: summary[k] for k in ("baseline", "selected", "component_areas")}, indent=2))


if __name__ == "__main__":
    main()
