"""Run a small, sequential experiment suite and summarize real results.

Each training run is a fresh subprocess, releasing its GPU memory on exit. The
base config supplies ONLY local data/cache/split paths; each named experiment
retains its own model, loss, resolution, and augmentation choices.

Example::

    python experiments.py --base configs/local_baseline.yaml --configs baseline tversky \
        --output runs/screen --epochs 6 --samples-per-epoch 960 --seeds 42

Use a new output directory for a different training budget. ``--resume`` may
continue interrupted compatible runs or skip already completed ones; it never
silently changes a saved experiment. No Kaggle score is inferred or invented.
Suite runs disable early stopping so every successful run gets the same budget.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
import subprocess
import sys
import time

from config import ROOT, load_config, save_config


DEFAULT_EXPERIMENTS = ["baseline", "lr_low", "lr_high", "resolution_1024",
                       "disc_mask", "clahe", "local_normalization", "no_augmentation",
                       "consensus", "tversky", "unetplusplus"]
METRICS = ["PQ_dataset", "SQ_dataset", "RQ_dataset", "observation_PQ_mean",
           "filament_dice_mean", "filament_iou_mean", "custom_one_to_many",
           "custom_many_to_one", "any_overlap_one_to_many", "any_overlap_many_to_one",
           "prediction_ms_per_image", "baseline_prediction_and_components_ms_per_image"]
NUMERIC_RESULTS = METRICS + ["training_seconds", "total_seconds", "parameters",
                            "weights_file_bytes", "optimizer_steps"]


def _json_hash(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                      allow_nan=False).encode()).hexdigest()


def _file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(value, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf8")
    temporary.replace(path)


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def resolve_experiment(name: str) -> Path:
    supplied = Path(name)
    if supplied.is_file():
        return supplied.resolve()
    path = ROOT / "configs" / (name if name.endswith((".yaml", ".yml")) else name + ".yaml")
    if not path.is_file():
        raise FileNotFoundError(f"Experiment config not found: {name}")
    return path.resolve()


def make_run_config(experiment: str | Path, base_cfg: dict, output: str | Path,
                    seed: int, epochs: int, samples_per_epoch: int) -> dict:
    """Resolve one run while isolating experiment choices from machine paths."""
    path = resolve_experiment(str(experiment))
    cfg = load_config(path)
    for key in ("data_root", "cache_dir", "split"):
        cfg["data"][key] = base_cfg["data"][key]
    cfg["seed"] = cfg["data"]["seed"] = int(seed)
    cfg["train"]["epochs"] = int(epochs)
    # A fixed-budget comparison must not give different methods fewer updates.
    # Direct train.py runs can still use their configured early stopping.
    cfg["train"]["patience"] = 0
    cfg["data"]["epoch_length"] = int(samples_per_epoch)
    name = re.sub(r"[^A-Za-z0-9_.-]", "_", path.stem) + f"_seed{seed}"
    cfg["run"] = dict(name=name, output_dir=str((Path(output) / name).resolve()))
    cfg = load_config(cfg)
    physical = cfg["data"]["batch_size"]
    effective = physical * cfg["train"].get("accumulation_steps", 1)
    if samples_per_epoch % effective:
        raise ValueError(f"samples_per_epoch={samples_per_epoch} must be divisible by effective batch "
                         f"size {effective} for a fair complete-update budget ({path.stem})")
    if cfg.get("diagnostic", {}).get("overfit_images"):
        raise ValueError("Diagnostic overfit configs cannot be used as research experiments")
    return cfg


def normalization_key(cfg: dict, max_images=200) -> tuple[str, dict]:
    """Identify training-only statistics, excluding the values they will replace."""
    d = cfg["data"]
    preprocess = {key: value for key, value in d["preprocess"].items() if key not in ("mean", "std")}
    catalog = Path(d["cache_dir"]) / "catalog.csv"
    spec = dict(version=1, data_root=str(Path(d["data_root"]).resolve()),
                split_sha256=_file_hash(Path(d["split"])), catalog_sha256=_file_hash(catalog),
                preprocess=preprocess, max_images=int(max_images))
    return _json_hash(spec), spec


def apply_training_normalization(cfg: dict, max_images=200) -> tuple[dict, dict]:
    """Compute or reuse the same bounded training-image sample for each variant.

    Statistics live in cache/normalization/<key>/, so a CLAHE trial cannot
    overwrite the plain-image statistics used by the baseline.
    """
    import data as D

    cfg = deepcopy(cfg)
    key, spec = normalization_key(cfg, max_images)
    location = Path(cfg["data"]["cache_dir"]) / "normalization" / key
    stats_path, metadata_path = location / "norm_stats.json", location / "source.json"
    if stats_path.exists() and metadata_path.exists():
        if json.loads(metadata_path.read_text(encoding="utf8")) != spec:
            raise ValueError(f"Normalization metadata does not match its cache key: {location}")
        stats = json.loads(stats_path.read_text(encoding="utf8"))
    else:
        print(f"Computing training normalization: {cfg['run']['name']} (at most {max_images} images)", flush=True)
        catalog = D.load_catalog(Path(cfg["data"]["cache_dir"]))
        split = D.load_split(Path(cfg["data"]["split"]))
        D.assert_no_leak(split, catalog)
        stats = D.compute_norm_stats(catalog, split["train"],
                                     Path(cfg["data"]["data_root"]) / "train" / "train_images",
                                     max_images=max_images, cache_dir=location,
                                     preprocess=spec["preprocess"])
        _write_json(spec, metadata_path)
    if (not math.isfinite(float(stats["mean"])) or not math.isfinite(float(stats["std"]))
            or stats["std"] <= 0 or not 0 < stats["n_images"] <= max_images):
        raise ValueError(f"Invalid normalization statistics: {stats_path}")
    cfg["data"]["preprocess"].update(mean=float(stats["mean"]), std=float(stats["std"]))
    return cfg, dict(key=key, path=str(stats_path), n_images=int(stats["n_images"]))


def comparison_key(cfg: dict) -> str:
    """Same experiment across seeds, excluding paths identifying one run."""
    comparable = deepcopy(cfg)
    comparable.pop("run", None)
    comparable.pop("seed", None)
    comparable["data"].pop("seed", None)
    return _json_hash(comparable)


def _record_template(cfg: dict, source: Path, config_path: Path, normalization: dict) -> dict:
    d, t = cfg["data"], cfg["train"]
    return dict(experiment=source.stem, run_name=cfg["run"]["name"], seed=cfg["seed"],
                comparison_key=comparison_key(cfg), config_sha256=_json_hash(cfg),
                config_path=str(config_path), run_dir=cfg["run"]["output_dir"],
                architecture=cfg["model"]["architecture"], encoder=cfg["model"]["encoder_name"],
                loss=cfg["loss"]["name"], learning_rate=t["learning_rate"],
                physical_batch_size=d["batch_size"], accumulation_steps=t.get("accumulation_steps", 1),
                effective_batch_size=d["batch_size"] * t.get("accumulation_steps", 1),
                target=d["target"], crop=d["crop"], train_resize=d["train_resize"],
                samples_per_epoch=d["epoch_length"], planned_epochs=t["epochs"],
                early_stopping_patience=t["patience"],
                planned_updates_per_epoch=d["epoch_length"] // (d["batch_size"] * t.get("accumulation_steps", 1)),
                primary_metric=cfg["evaluation"]["primary_metric"],
                normalization_key=normalization["key"], normalization_images=normalization["n_images"],
                split_sha256=_file_hash(Path(d["split"])),
                status="pending", started_utc=None, finished_utc=None, error=None, Kaggle_PQ=None)


def _apply_summary(record: dict, summary: dict, cfg: dict) -> dict:
    record = deepcopy(record)
    if summary.get("overfit_images", 0):
        record["status"] = "diagnostic"
        record["error"] = "Diagnostic subset scores excluded from research summaries"
        return record
    complete = summary.get("epochs_completed", 0) >= cfg["train"]["epochs"]
    record["status"] = "completed" if complete else "partial"
    if summary.get("stop_reason") == "early_stopping" and not complete:
        record["error"] = "Stopped before the fixed experiment budget was completed"
    for key in ("best_epoch", "epochs_completed", "stop_reason", "training_seconds", "total_seconds",
                "optimizer_steps", "parameters", "weights_file_bytes", "training_checkpoint_bytes",
                "best_checkpoint", "weights_checkpoint", "best_score"):
        record[key] = summary.get(key)
    for key in METRICS:
        record[key] = summary.get("best_validation", {}).get(key)
    primary_score = summary.get("best_validation", {}).get(cfg["evaluation"]["primary_metric"])
    if primary_score is None or not math.isfinite(float(primary_score)):
        record.update(status="failed", error="Training summary has no finite primary validation score")
    # Preserve the provenance of the code that actually trained this model,
    # rather than assigning the source files present when tables are regenerated.
    metadata_path = Path(cfg["run"]["output_dir"]) / "run_metadata.json"
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf8"))
        sources = metadata.get("source_sha256")
        record["source_fingerprint"] = _json_hash(sources) if sources else None
        record["gpu"] = metadata.get("environment", {}).get("gpu")
    return record


def aggregate_seeds(records: list[dict]) -> list[dict]:
    """Across-independent-run means and SAMPLE std; singleton std is undefined.

    Per-image or per-annotation standard deviations are never substituted for
    run-to-run variability. Failed, partial, and diagnostic runs are excluded.
    """
    groups = {}
    for record in records:
        if record.get("status") != "completed":
            continue
        key = (record["experiment"], record["comparison_key"], record.get("source_fingerprint") or "unknown")
        groups.setdefault(key, []).append(record)
    summaries = []
    for (experiment, key, source_fingerprint), runs in sorted(groups.items()):
        seeds = [int(run["seed"]) for run in runs]
        if len(set(seeds)) != len(seeds):
            raise ValueError(f"Duplicate seed for one experiment comparison: {experiment}")
        row = dict(experiment=experiment, comparison_key=key, source_fingerprint=source_fingerprint,
                   n_runs=len(runs),
                   seeds=",".join(map(str, sorted(seeds))))
        for metric in NUMERIC_RESULTS:
            values = [float(run[metric]) for run in runs
                      if run.get(metric) is not None and math.isfinite(float(run[metric]))]
            row[f"{metric}_n"] = len(values)
            row[f"{metric}_mean"] = statistics.mean(values) if values else None
            row[f"{metric}_std_across_runs"] = statistics.stdev(values) if len(values) > 1 else None
        summaries.append(row)
    return summaries


def _write_csv(rows: list[dict], path: Path) -> None:
    fields = list(dict.fromkeys(key for row in rows for key in row))
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf8") as f:
        if fields:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
    temporary.replace(path)


def refresh_tables(output: str | Path) -> list[dict]:
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    records = [json.loads(p.read_text(encoding="utf8")) for p in sorted((output / "records").glob("*.json"))]
    _write_csv(records, output / "experiments.csv")
    _write_csv(aggregate_seeds(records), output / "seed_summary.csv")
    return records


def _run_subprocess(command: list[str], log_path: Path) -> int:
    """Capture complete output while showing readable training progress."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf8") as log:
        log.write(f"\n--- invocation {_timestamp()} ---\n")
        log.flush()
        process = subprocess.Popen(command, cwd=ROOT, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True, encoding="utf8",
                                   errors="replace", bufsize=1)
        try:
            assert process.stdout is not None
            for line in process.stdout:
                log.write(line)
                log.flush()
                clean = line.strip()
                if clean.startswith(("Run ", "Epoch ", "validation:", "DIAGNOSTIC", "Traceback")):
                    print(clean, flush=True)
            return process.wait()
        except BaseException:
            process.terminate()
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            raise


def run_suite(base, experiments, output, epochs=6, samples_per_epoch=960,
              seeds=(42,), resume=False, max_norm_images=200, prepare_only=False) -> list[dict]:
    """Execute selected experiments sequentially; failures are recorded honestly."""
    base_cfg = load_config(base)
    output = Path(output).resolve()
    if len(set(seeds)) != len(seeds):
        raise ValueError("Seed list contains duplicates")
    sources = [resolve_experiment(str(name)) for name in experiments]
    if len({p.stem for p in sources}) != len(sources):
        raise ValueError("Experiment filenames must have unique stems within a suite")
    for directory in (output, output / "configs", output / "records", output / "logs"):
        directory.mkdir(parents=True, exist_ok=True)
    for source in sources:
        for seed in seeds:
            cfg = make_run_config(source, base_cfg, output, seed, epochs, samples_per_epoch)
            cfg, normalization = apply_training_normalization(cfg, max_norm_images)
            name, run_dir = cfg["run"]["name"], Path(cfg["run"]["output_dir"])
            config_path = output / "configs" / f"{name}.yaml"
            record_path = output / "records" / f"{name}.json"
            record = _record_template(cfg, source, config_path, normalization)
            if config_path.exists():
                existing_cfg = load_config(config_path)
                if _json_hash(existing_cfg) != _json_hash(cfg):
                    raise ValueError(f"Settings changed for {name}; use a new output directory")
            if record_path.exists():
                previous = json.loads(record_path.read_text(encoding="utf8"))
                if previous.get("config_sha256") != record["config_sha256"]:
                    raise ValueError(f"Recorded settings changed for {name}; use a new output directory")
                record["Kaggle_PQ"] = previous.get("Kaggle_PQ")
                for key in ("started_utc", "finished_utc", "last_started_utc", "invocation_seconds"):
                    if key in previous:
                        record[key] = previous[key]
            save_config(cfg, config_path)
            resume_path = None
            if run_dir.exists() and any(run_dir.iterdir()):
                if not resume:
                    raise FileExistsError(f"Run already exists: {run_dir}; use --resume or a new output")
                saved_path = run_dir / "config.yaml"
                if not saved_path.exists() or _json_hash(load_config(saved_path)) != _json_hash(cfg):
                    raise ValueError(f"Saved run config differs for {name}; refusing to resume")
                metadata_path = run_dir / "run_metadata.json"
                if metadata_path.exists():
                    metadata = json.loads(metadata_path.read_text(encoding="utf8"))
                    if metadata.get("split_sha256") != record["split_sha256"]:
                        raise ValueError(f"Split changed since {name} was started")
                summary_path = run_dir / "summary.json"
                if summary_path.exists():
                    record = _apply_summary(record, json.loads(summary_path.read_text(encoding="utf8")), cfg)
                    if record["status"] == "completed":
                        _write_json(record, record_path)
                        refresh_tables(output)
                        print(f"Skipping completed compatible run: {name}", flush=True)
                        continue
                resume_path = run_dir / "last.pt"
                if not resume_path.exists():
                    raise FileExistsError(f"Incomplete run has no checkpoint: {run_dir}. Inspect its log; "
                                          "use a new output directory after resolving the failure.")
            if prepare_only:
                _write_json(record, record_path)
                refresh_tables(output)
                print(f"Prepared {name}: {config_path}", flush=True)
                continue
            invocation_started = _timestamp()
            record.update(status="running", started_utc=record["started_utc"] or invocation_started,
                          last_started_utc=invocation_started, finished_utc=None, error=None)
            _write_json(record, record_path)
            refresh_tables(output)
            command = [sys.executable, "-u", str(ROOT / "train.py"), "--config", str(config_path)]
            if resume_path:
                command += ["--resume", str(resume_path)]
            print(f"Starting {name} | {epochs} epochs | {samples_per_epoch} samples/epoch", flush=True)
            start = time.perf_counter()
            try:
                code = _run_subprocess(command, output / "logs" / f"{name}.log")
                summary_path = run_dir / "summary.json"
                if code == 0 and summary_path.exists():
                    record = _apply_summary(record, json.loads(summary_path.read_text(encoding="utf8")), cfg)
                else:
                    record.update(status="failed", error=f"train.py exit code {code}; inspect logs/{name}.log")
            except KeyboardInterrupt:
                record.update(status="interrupted", error="Interrupted; last completed epoch can be resumed")
                record["finished_utc"] = _timestamp()
                _write_json(record, record_path)
                refresh_tables(output)
                raise
            except Exception as exc:
                record.update(status="failed", error=f"{type(exc).__name__}: {exc}")
            record.update(finished_utc=_timestamp(), invocation_seconds=time.perf_counter() - start)
            _write_json(record, record_path)
            refresh_tables(output)
            print(f"Finished {name}: {record['status']} | PQ={record.get('PQ_dataset')}", flush=True)
    return refresh_tables(output)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default="configs/local_baseline.yaml", help="Local data/cache/split paths")
    parser.add_argument("--configs", nargs="+", default=DEFAULT_EXPERIMENTS,
                        help="Names under configs/ or paths to experiment YAML files")
    parser.add_argument("--output", type=Path, default=Path("runs/screen"))
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--samples-per-epoch", type=int, default=960)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42])
    parser.add_argument("--max-norm-images", type=int, default=200)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--prepare-only", action="store_true", help="Resolve configs and stats without starting training")
    parser.add_argument("--aggregate-only", action="store_true", help="Rebuild CSV tables from existing run records")
    args = parser.parse_args(argv)
    if args.aggregate_only:
        records = refresh_tables(args.output)
    else:
        records = run_suite(args.base, args.configs, args.output, args.epochs,
                            args.samples_per_epoch, args.seeds, args.resume,
                            args.max_norm_images, args.prepare_only)
    print(f"Recorded {len(records)} runs. Tables: {args.output / 'experiments.csv'} and "
          f"{args.output / 'seed_summary.csv'}", flush=True)
    if any(row.get("status") in {"failed", "interrupted", "partial"} for row in records):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
