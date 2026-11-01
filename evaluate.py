"""Evaluate checkpoints against every annotation set of each validation image.

The selection metric is ``PQ_dataset``: pooled matching counts across all
annotation sets, following the host's Self Evaluation Notebook v6. Equal-weight
observation and annotation-set means remain useful diagnostics. Rare overlapping
GT annotations are decoded separately from the original training JSON so the
integer training cache's overlap resolution does not alter evaluation.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch

from inference import predict_probability, probability_to_instances
from metrics import MetricAccumulator, evaluate_pair, evaluate_mask_layers


def file_sha256(path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_device(device) -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu") if str(device) == "auto" else torch.device(device)


def _raw_annotations_for_loader(loader) -> dict:
    """Load only annotations whose cached maps can lose source information.

    Parsed polygon lists are cached on the dataset; full binary layers are
    decoded one set at a time. This reads training labels only.
    """
    dataset = getattr(loader, "dataset", None)
    while dataset is not None and not hasattr(dataset, "cat") and hasattr(dataset, "dataset"):
        dataset = dataset.dataset
    if dataset is None or not hasattr(dataset, "cat"):
        return {}
    if hasattr(dataset, "_evaluation_raw_annotations"):
        return dataset._evaluation_raw_annotations
    cat = dataset.cat
    selected = set(str(s) for _, group in dataset.groups for s in group["set_id"])
    affected = cat[(cat["overlap_px"] > 0) | (cat["n_empty_masks"] > 0)]
    affected_ids = set(affected["set_id"].astype(str)) & selected
    annotations = {sid: [] for sid in affected_ids}
    if affected_ids:
        import data as D
        source = Path(dataset.img_dir).parent / "MAGFiLO_1.0_Annotations_kaggle2026_train.json"
        coco = D.load_coco(source)
        source_ids = {str(im["id"]) for im in coco["images"]}
        if not affected_ids <= source_ids:
            raise ValueError("cached annotation IDs do not match the original training JSON")
        for ann in coco["annotations"]:
            sid = str(ann["image_id"])
            if sid in annotations:
                annotations[sid].append(ann["segmentation"])
    dataset._evaluation_raw_annotations = annotations
    return annotations


def _write_rows(path: Path, rows: list[dict], fieldnames=None) -> None:
    fields = fieldnames or (list(rows[0]) if rows else [])
    if not fields:
        return
    with path.open("w", newline="", encoding="utf8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def evaluate_model(model, val_loader, device, cfg: dict, output_dir=None,
                   save_predictions: bool = False) -> dict:
    """Return scalar scores; optionally export detailed results and predictions.

    Validation batches must have size one. Timings distinguish prediction
    (transfer + model + return to CPU), instance conversion, and scoring.
    ``validation_seconds`` also includes loading and optional per-image NPZ
    exports, but excludes writing aggregate result tables. It is not the final
    test pipeline's end-to-end inference benchmark.
    """
    device = resolve_device(device)
    ec = cfg.get("evaluation", {})
    threshold, iou_threshold = float(ec.get("threshold", 0.5)), float(ec.get("iou_threshold", 0.5))
    if not 0 < threshold < 1:
        raise ValueError("foreground threshold must lie strictly between 0 and 1")
    if not 0.5 <= iou_threshold <= 1:
        raise ValueError("instance-matching threshold must be in [0.5,1]")
    out = Path(output_dir) if output_dir is not None else None
    if save_predictions and out is None:
        raise ValueError("output_dir is required when saving predictions")
    if out is not None:
        out.mkdir(parents=True, exist_ok=True)
        if save_predictions:
            (out / "predictions").mkdir(exist_ok=True)
    acc = MetricAccumulator()
    filament_rows = []
    overlap_rows = []
    raw_annotations = _raw_annotations_for_loader(val_loader)
    n_raw_sets = 0
    visited = set()
    prediction_seconds = postprocess_seconds = scoring_seconds = 0.0
    start = time.perf_counter()
    for batch in val_loader:
        if batch["image"].shape[0] != 1:
            raise ValueError("validation requires batch_size=1 because annotation counts differ")
        name = batch["name"][0] if isinstance(batch["name"], (list, tuple)) else str(batch["name"])
        if name in visited:
            raise ValueError(f"duplicate validation observation: {name}")
        if Path(name).name != name or any(c in name for c in "/\\:"):
            raise ValueError("invalid observation ID")
        visited.add(name)
        before = time.perf_counter()
        image = batch["image"].to(device, non_blocking=device.type == "cuda")
        probability = predict_probability(model, image, cfg.get("inference", {}))[0, 0].cpu().numpy()
        prediction_seconds += time.perf_counter() - before
        before = time.perf_counter()
        pred = probability_to_instances(probability, threshold)
        postprocess_seconds += time.perf_counter() - before
        before = time.perf_counter()
        sets = batch["gt_sets"][0].numpy()
        if len(sets) == 0:
            raise ValueError(f"no annotation sets for observation {name}")
        set_ids = batch.get("set_ids")
        if raw_annotations and set_ids is None:
            raise ValueError("validation batch needs set_ids for lossless overlapping-annotation scoring")
        for k, gt in enumerate(sets):
            sid = (set_ids[k][0] if isinstance(set_ids[k], (tuple, list)) else str(set_ids[k])) if set_ids else f"{name}_set{k}"
            options = dict(iou_thr=iou_threshold, tau=float(ec.get("relation_tau", 0.1)),
                           own=float(ec.get("relation_own", 0.5)))
            if sid in raw_annotations:
                import data as D
                layers = (D.decode_annotation(seg, *pred.shape)[0] for seg in raw_annotations[sid])
                result = evaluate_mask_layers(pred, layers, **options)
                n_raw_sets += 1
            else:
                result = evaluate_pair(pred, gt, **options)
            acc.add(result, name=sid, observation=name)
            if out is not None:
                fil = result["filament"]
                filament_rows.extend(dict(observation=name, annotation_set=sid, gt_id=int(gid),
                                          iou=float(iou), dice=float(dice), gt_area=int(area))
                                     for gid, iou, dice, area in zip(fil["gt_id"], fil["iou"], fil["dice"], fil["gt_area"]))
                pairs = result["overlap_pairs"]
                overlap_rows.extend(dict(observation=name, annotation_set=sid, gt_id=int(gid), pred_id=int(pid),
                                         iou=float(iou), dice=float(dice))
                                    for gid, pid, iou, dice in zip(pairs["gt_id"], pairs["pred_id"], pairs["iou"], pairs["dice"]))
        scoring_seconds += time.perf_counter() - before
        if save_predictions:
            np.savez_compressed(out / "predictions" / f"{name}.npz", probability=probability,
                                instances=pred.astype(np.uint32), observation=name)
        if ec.get("progress", False) and len(visited) % 20 == 0:
            print(f"  validation: {len(visited)} observations", flush=True)
    if not visited:
        raise ValueError("validation loader is empty")
    summary = acc.summary()
    summary.update(
        evaluation_policy="pooled_counts_over_annotation_sets; official_self_evaluation_v6",
        primary_metric=ec.get("primary_metric", "PQ_dataset"),
        n_raw_annotation_sets=n_raw_sets,
        ground_truth_policy="cached disjoint maps; original binary layers for overlapping/empty annotations",
        foreground_threshold=threshold, iou_matching_threshold=iou_threshold,
        prediction_seconds=prediction_seconds, postprocess_seconds=postprocess_seconds,
        scoring_seconds=scoring_seconds, validation_seconds=time.perf_counter() - start,
        prediction_ms_per_image=1000 * prediction_seconds / len(visited),
        baseline_prediction_and_components_ms_per_image=1000 * (prediction_seconds + postprocess_seconds) / len(visited),
    )
    if out is not None:
        _write_rows(out / "annotation_sets.csv", acc.per_image)
        _write_rows(out / "observations.csv", acc.observation_rows())
        _write_rows(out / "filaments.csv", filament_rows,
                    ["observation", "annotation_set", "gt_id", "iou", "dice", "gt_area"])
        _write_rows(out / "overlap_pairs.csv", overlap_rows,
                    ["observation", "annotation_set", "gt_id", "pred_id", "iou", "dice"])
        np.savez_compressed(out / "filament_scores.npz", **acc.filament_arrays())
        np.savez_compressed(out / "overlap_scores.npz", **acc.overlap_arrays())
        with (out / "summary.json").open("w", encoding="utf8") as f:
            json.dump(summary, f, indent=2, allow_nan=False)
    return summary


def main(argv=None) -> None:
    from config import load_config, save_config
    from models import load_checkpoint_model
    import data as D

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--config", help="optional complete config override; checkpoint settings are the default")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--data-root")
    p.add_argument("--save-predictions", action="store_true")
    args = p.parse_args(argv)
    device = resolve_device(args.device)
    model, cfg, _ = load_checkpoint_model(args.checkpoint, device)
    if args.config:
        override = load_config(args.config)
        if override["model"] != cfg["model"]:
            raise ValueError("config override must describe the checkpoint's model architecture/settings")
        cfg = override
    if args.data_root:
        cfg["data"]["data_root"] = args.data_root
    _, val_loader = D.make_loaders(cfg["data"])
    summary = evaluate_model(model, val_loader, device, cfg, args.output, args.save_predictions)
    summary.update(checkpoint=str(Path(args.checkpoint).resolve()), checkpoint_sha256=file_sha256(args.checkpoint))
    save_config(cfg, Path(args.output) / "config.yaml")
    with (Path(args.output) / "summary.json").open("w", encoding="utf8") as f:
        json.dump(summary, f, indent=2, allow_nan=False)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
