"""Verify and copy a completed Person B run into a portable handoff directory.

This command reads saved artifacts only: it never loads a model or starts a GPU
job. The source run, dataset, and frozen split are left untouched. An existing
package is accepted only if every file still matches the requested package.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
import tempfile
import time

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parent
IMAGE_SHAPE = (2048, 2048)
OBSERVATION = re.compile(r"\d{14}[A-Za-z]{2}\Z")
FILAMENT = re.compile(r"(\d{14}[A-Za-z]{2})_([1-9]\d*)\Z")
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
SOURCE_FILES = ("config.py", "data.py", "models.py", "losses.py", "train.py",
                "inference.py", "metrics.py", "evaluate.py", "predict.py",
                "experiments.py", "report.py", "study_report.py", "local_job.py",
                "package_person_b.py", "requirements.txt")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json(path):
    return json.loads(Path(path).read_text(encoding="utf8"))


def _json_bytes(value):
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf8")


def _publish_directory(stage: Path, output: Path) -> None:
    """Allow up to five seconds for a Windows access/sharing lock to clear."""
    for attempt in range(6):
        if output.exists() or output.is_symlink():
            raise FileExistsError("Destination was created while this package was being prepared")
        try:
            stage.rename(output)
            return
        except OSError as exc:
            # Access denied can be a transient scanner lock. Retry only these
            # Windows errors; missing files, full disks, and other errors fail.
            if getattr(exc, "winerror", None) not in {5, 32, 33} or attempt == 5:
                raise
            print(f"Windows publication error {exc.winerror}; retrying directory rename "
                  f"({attempt + 2}/6)", flush=True)
            time.sleep(1.0)


def _settings(cfg):
    """Ignore only machine/run paths when comparing saved configurations."""
    result = deepcopy(cfg)
    result.pop("run", None)
    for key in ("data_root", "cache_dir", "split"):
        result["data"].pop(key, None)
    return result


def portable_config(cfg):
    """Keep every numerical/model choice; rewrite four filesystem locations."""
    portable = deepcopy(cfg)
    portable["data"].update(data_root="data", cache_dir="cache", split="splits/split_v1.json")
    portable["run"]["output_dir"] = "runs/reproduction"
    return portable


def _csv_rows(path, expected_observations):
    rows, seen = {}, set()
    with Path(path).open(newline="", encoding="utf8") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != ["filament_id", "segmentation_rle"]:
            raise ValueError("Submission CSV must contain filament_id,segmentation_rle")
        for row in reader:
            identifier = row["filament_id"]
            match = FILAMENT.fullmatch(identifier)
            if (not match or match[1] not in expected_observations or identifier in seen
                    or not row["segmentation_rle"] or None in row):
                raise ValueError(f"Invalid, duplicate, or unexpected CSV filament ID: {identifier}")
            seen.add(identifier)
            rows.setdefault(match[1], {})[int(match[2])] = row["segmentation_rle"]
    return rows, len(seen)


def verify_predictions(directory, observations, shape, threshold, csv_rows=None):
    """Inspect one NPZ at a time; optionally decode every corresponding CSV mask."""
    directory = Path(directory)
    files = {p.stem: p for p in directory.glob("*.npz")}
    if set(files) != set(observations):
        raise ValueError(f"Prediction coverage differs from expected observations: {directory}")
    if csv_rows is not None:
        from pycocotools import mask as coco_mask
    counts, empty = {}, []
    low, high = 1.0, 0.0
    for observation, path in sorted(files.items()):
        with np.load(path, allow_pickle=False) as archive:
            if not {"probability", "instances", "observation"} <= set(archive.files):
                raise ValueError(f"Missing prediction arrays: {path}")
            name = archive["observation"]
            if name.shape != () or name.item() != observation or not OBSERVATION.fullmatch(observation):
                raise ValueError(f"Prediction observation identity is incorrect: {path}")
            probability, instances = archive["probability"], archive["instances"]
            if tuple(probability.shape) != tuple(shape) or tuple(instances.shape) != tuple(shape):
                raise ValueError(f"Prediction shape differs from {shape}: {path}")
            if (not np.issubdtype(probability.dtype, np.floating) or not np.isfinite(probability).all()
                    or probability.min() < 0 or probability.max() > 1):
                raise ValueError(f"Prediction probabilities must be finite and within [0,1]: {path}")
            if not np.issubdtype(instances.dtype, np.integer) or np.any(instances < 0):
                raise ValueError(f"Prediction instances must be nonnegative integers: {path}")
            if not np.array_equal(instances > 0, probability > threshold):
                raise ValueError(f"Instances do not match the configured probability threshold: {path}")
            ids = np.unique(instances[instances > 0])
            counts[observation] = len(ids)
            if not len(ids):
                empty.append(observation)
            low, high = min(low, float(probability.min())), max(high, float(probability.max()))
            if csv_rows is not None:
                rows = csv_rows.get(observation, {})
                if set(rows) != set(range(1, len(ids) + 1)):
                    raise ValueError(f"CSV instance coverage differs from saved map: {observation}")
                for index, instance_id in enumerate(ids, 1):
                    decoded = coco_mask.decode({"size": list(shape), "counts": rows[index].encode("ascii")})
                    if not np.array_equal(decoded, instances == instance_id):
                        raise ValueError(f"CSV RLE mask differs from saved instance: {observation}_{index}")
    return dict(n_observations=len(files), n_instances=sum(counts.values()),
                instances_per_observation=counts, zero_detection_images=empty,
                probability_min=low, probability_max=high,
                csv_masks_exactly_verified=csv_rows is not None)


def verify_run(run, *, expected_shape=IMAGE_SHAPE, expected_val_count=122, expected_test_count=180):
    """Verify provenance and coverage before creating any destination files."""
    run = Path(run).resolve()
    cfg = yaml.safe_load((run / "config.yaml").read_text(encoding="utf8"))
    summary, metadata = _json(run / "summary.json"), _json(run / "run_metadata.json")
    if (summary.get("overfit_images", 0) or summary.get("stop_reason") != "completed"
            or summary.get("epochs_completed") != cfg["train"]["epochs"]):
        raise ValueError("Package requires a completed full-budget run, not a diagnostic or partial run")
    split_path = Path(cfg["data"]["split"])
    split_hash = file_sha256(split_path)
    if summary.get("split_sha256") != split_hash or metadata.get("split_sha256") != split_hash:
        raise ValueError("Frozen split hash does not match the training run")
    split = _json(split_path)
    validation = split["val"]
    if (len(validation) != expected_val_count or len(set(validation)) != len(validation)
            or set(validation) & set(split["train"])):
        raise ValueError("Frozen split has incorrect validation coverage or overlapping train IDs")
    weights_hash = file_sha256(run / "best_weights.pt")
    evaluation, submission = _json(run / "evaluation/summary.json"), _json(run / "submission.summary.json")
    for label, artifact in (("evaluation", evaluation), ("submission", submission)):
        if artifact.get("checkpoint_sha256") != weights_hash:
            raise ValueError(f"{label} checkpoint hash differs from best_weights.pt; regenerate its artifacts")
    eval_cfg = yaml.safe_load((run / "evaluation/config.yaml").read_text(encoding="utf8"))
    if _settings(eval_cfg) != _settings(cfg) or _settings(submission["config"]) != _settings(cfg):
        raise ValueError("Evaluation/submission numerical settings differ from the training configuration")
    threshold = float(cfg["evaluation"]["threshold"])
    if not 0 < threshold < 1 or any(a.get("foreground_threshold") != threshold for a in (evaluation, submission)):
        raise ValueError("Evaluation/submission foreground thresholds do not match the configuration")
    if evaluation.get("n_observations") != expected_val_count:
        raise ValueError("Evaluation summary has incorrect validation observation count")
    primary = cfg["evaluation"]["primary_metric"]
    score = evaluation.get(primary)
    trained_score = summary.get("best_validation", {}).get(primary)
    if (score is None or trained_score is None or not math.isfinite(float(score))
            or not math.isfinite(float(trained_score)) or abs(float(score) - float(trained_score)) > 1e-9):
        raise ValueError("Saved evaluation primary score does not match the selected training checkpoint")
    test_dir = Path(cfg["data"]["data_root"]) / "test/test_images"
    test_files = [p for p in test_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES]
    test_names = [p.stem for p in test_files]
    if (len(test_names) != expected_test_count or len(set(test_names)) != len(test_names)
            or any(not OBSERVATION.fullmatch(name) for name in test_names)):
        raise ValueError("Raw test image names/count do not match expected coverage")
    sidecar_images = submission["images"]
    sidecar_names = [row["observation"] for row in sidecar_images]
    if (len(sidecar_names) != len(set(sidecar_names)) or set(sidecar_names) != set(test_names)
            or submission.get("n_images") != len(test_names)):
        raise ValueError("Submission sidecar image coverage differs from test images")
    rows, n_rows = _csv_rows(run / "submission.csv", set(test_names))
    val_check = verify_predictions(run / "evaluation/predictions", validation, expected_shape, threshold)
    test_check = verify_predictions(run / "test_predictions", test_names, expected_shape, threshold, rows)
    if n_rows != test_check["n_instances"] or n_rows != submission.get("n_filaments"):
        raise ValueError("Submission filament count differs from CSV or saved maps")
    empty = submission.get("zero_detection_images", [])
    if len(empty) != len(set(empty)) or set(empty) != set(test_check["zero_detection_images"]):
        raise ValueError("Submission zero-detection image list is incorrect")
    if any(row["n_instances"] != test_check["instances_per_observation"][row["observation"]]
           for row in sidecar_images):
        raise ValueError("Submission per-image instance counts differ from saved maps")
    return cfg, dict(checkpoint_sha256=weights_hash, split_sha256=split_hash,
                     validation=val_check, test=test_check,
                     kaggle_acceptance_verified=bool(submission.get("kaggle_acceptance_verified", False)))


def package_run(run, output, study=None, *, expected_shape=IMAGE_SHAPE,
                expected_val_count=122, expected_test_count=180):
    """Copy only verified artifacts; reject incompatible existing destinations."""
    run, output = Path(run).resolve(), Path(output).resolve()
    if run == output or run in output.parents or output in run.parents:
        raise ValueError("Package output must not overlap the source run")
    cfg, checks = verify_run(run, expected_shape=expected_shape,
                             expected_val_count=expected_val_count, expected_test_count=expected_test_count)
    copies = {"best_weights.pt": run / "best_weights.pt", "config.original.yaml": run / "config.yaml",
              "splits/split_v1.json": Path(cfg["data"]["split"])}
    for name in ("summary.json", "run_metadata.json", "history.csv", "history.json",
                 "submission.csv", "submission.summary.json"):
        copies[name] = run / name
    for directory in ("evaluation", "test_predictions"):
        for path in sorted((run / directory).rglob("*")):
            if path.is_file():
                if path.is_symlink() or path.suffix.lower() not in {".json", ".yaml", ".csv", ".npz"}:
                    raise ValueError(f"Unexpected file in export directory: {path}")
                copies[path.relative_to(run).as_posix()] = path
    for name in SOURCE_FILES:
        copies[name] = ROOT / name
    metadata = _json(run / "run_metadata.json")
    for name, expected in metadata.get("source_sha256", {}).items():
        if name not in SOURCE_FILES or file_sha256(ROOT / name) != expected:
            raise ValueError(f"Current source snapshot differs from the training provenance: {name}")
    if study:
        study = Path(study).resolve()
        if not study.is_dir() or study == output or study in output.parents or output in study.parents:
            raise ValueError("Study directory is missing or overlaps package output")
        for path in sorted(study.rglob("*")):
            if path.relative_to(study).as_posix() == "progress.json":
                continue  # Live status is not part of the immutable final study.
            if path.is_file() and path.suffix.lower() in {".json", ".csv", ".png", ".svg", ".pdf", ".md"}:
                if path.is_symlink():
                    raise ValueError(f"Study snapshot must not follow symlinks: {path}")
                copies["study/" + path.relative_to(study).as_posix()] = path
    handoff = dict(format_version=1, source_run=str(run), checkpoint="best_weights.pt",
                   original_config="config.original.yaml", portable_config="config.yaml",
                   frozen_split="splits/split_v1.json", validation_predictions="evaluation/predictions",
                   test_predictions="test_predictions", submission="submission.csv", checks=checks,
                   model_loading="Use models.load_checkpoint_model with the included best_weights.pt",
                   data_setup="Set config.yaml data.data_root to the MAGFiLO dataset root; rebuild cache locally",
                   evaluation_command="python evaluate.py --checkpoint best_weights.pt --config config.yaml --output reproduced_evaluation --save-predictions",
                   prediction_command="python predict.py --checkpoint best_weights.pt --config config.yaml --output reproduced_submission.csv",
                   exclusions=["raw dataset", "optimizer/resume checkpoints", "Kaggle upload verification"],
                   provenance_note="Original artifacts retain original absolute paths; config.yaml changes only filesystem locations")
    generated = {"config.yaml": yaml.safe_dump(portable_config(cfg), sort_keys=False).encode("utf8"),
                 "handoff.json": _json_bytes(handoff)}
    records = {name: dict(bytes=path.stat().st_size, sha256=file_sha256(path)) for name, path in copies.items()}
    if (records["best_weights.pt"]["sha256"] != checks["checkpoint_sha256"]
            or records["splits/split_v1.json"]["sha256"] != checks["split_sha256"]):
        raise ValueError("Checkpoint or split changed while the package was being verified")
    records.update({name: dict(bytes=len(value), sha256=hashlib.sha256(value).hexdigest())
                    for name, value in generated.items()})
    manifest = dict(format_version=1, checkpoint_sha256=checks["checkpoint_sha256"], files=dict(sorted(records.items())))
    manifest_bytes = _json_bytes(manifest)
    if output.exists():
        expected_names = set(records) | {"manifest.json"}
        existing_names = {p.relative_to(output).as_posix() for p in output.rglob("*") if p.is_file()}
        if existing_names != expected_names or (output / "manifest.json").read_bytes() != manifest_bytes:
            raise FileExistsError("Destination differs from this package; choose a new output directory")
        for name, expected in records.items():
            if (output / name).is_symlink() or file_sha256(output / name) != expected["sha256"]:
                raise FileExistsError(f"Destination content changed: {name}")
        return handoff
    output.parent.mkdir(parents=True, exist_ok=True)
    # Stage next to the destination so the final directory rename stays on one
    # filesystem. TemporaryDirectory owns only its freshly created staging path.
    with tempfile.TemporaryDirectory(prefix=f".{output.name}-", dir=output.parent) as temporary:
        stage = Path(temporary)
        for name, source in copies.items():
            target = stage / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
        for name, value in generated.items():
            (stage / name).write_bytes(value)
        for name, expected in records.items():
            if file_sha256(stage / name) != expected["sha256"]:
                raise ValueError(f"Source changed or copy verification failed: {name}")
        (stage / "manifest.json").write_bytes(manifest_bytes)
        _publish_directory(stage, output)
    return handoff


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--study", type=Path, help="Optional completed study figures/tables/JSON to snapshot")
    args = parser.parse_args(argv)
    handoff = package_run(args.run, args.output, args.study)
    print(json.dumps(dict(output=str(args.output.resolve()), checks=handoff["checks"]), indent=2))


if __name__ == "__main__":
    main()
