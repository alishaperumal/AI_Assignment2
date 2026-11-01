"""Config-driven, resumable training for solar-filament segmentation.

Run ``python train.py --config configs/baseline.yaml``. Checkpoints resume at
completed epoch boundaries. ``--max-epochs`` stops this invocation early while
preserving the configured cosine-schedule horizon, useful for a resume test.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import csv
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import random
import time

import numpy as np
import torch
from torch import nn

from losses import build_loss
from models import build_model, model_size


def choose_device(requested="auto") -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type not in {"cuda", "cpu"}:
        raise ValueError("Choose auto, cpu, or cuda for this training implementation")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but this PyTorch environment cannot use it")
    return device


def seed_training(seed: int, deterministic=True) -> None:
    # Set before the first CUDA tensor operation, as required by deterministic GEMM.
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = not deterministic
    torch.backends.cudnn.deterministic = bool(deterministic)
    torch.use_deterministic_algorithms(bool(deterministic))


def _random_state() -> dict:
    return dict(python=random.getstate(), numpy=np.random.get_state(),
                torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])


def _restore_random_state(state: dict) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and state.get("cuda"):
        if len(state["cuda"]) != torch.cuda.device_count():
            raise ValueError("Resume CUDA device count differs; use the original environment")
        torch.cuda.set_rng_state_all(state["cuda"])


def _atomic_torch_save(value, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def _write_json(value, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def _write_history(history: list[dict], output_dir: Path) -> None:
    _write_json(history, output_dir / "history.json")
    fields = list(dict.fromkeys(k for row in history for k in row))
    temporary = output_dir / "history.csv.tmp"
    with temporary.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(history)
    temporary.replace(output_dir / "history.csv")


def _environment(device: torch.device) -> dict:
    packages = {}
    for name in ("torch", "torchvision", "segmentation-models-pytorch", "albumentations",
                 "numpy", "opencv-python", "opencv-python-headless", "scipy", "pycocotools"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return dict(python=platform.python_version(), platform=platform.platform(),
                device=str(device), gpu=torch.cuda.get_device_name(device) if device.type == "cuda" else None,
                torch_cuda=torch.version.cuda, packages=packages)


def _source_hashes() -> dict:
    root = Path(__file__).resolve().parent
    return {name: hashlib.sha256((root / name).read_bytes()).hexdigest()
            for name in ("config.py", "data.py", "models.py", "losses.py", "train.py",
                         "inference.py", "evaluate.py", "metrics.py") if (root / name).is_file()}


def _split_hash(cfg: dict) -> str:
    path = Path(cfg["data"].get("split", "splits/split_v1.json"))
    if not path.is_file():
        raise FileNotFoundError(f"Saved split is missing: {path}")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _resume_signature(cfg: dict) -> dict:
    """Settings that must remain identical for an epoch-boundary continuation."""
    signature = {key: deepcopy(cfg[key]) for key in
                 ("seed", "data", "model", "loss", "train", "inference", "evaluation")}
    # Worker count and device can change for portability; mathematical identity
    # across devices is not promised. Dataset randomness does not depend on workers.
    for key in ("num_workers", "val_num_workers", "pin_memory"):
        signature["data"].pop(key, None)
    signature["train"].pop("device", None)
    signature["evaluation"].pop("save_predictions", None)
    return signature


def _make_loaders(cfg: dict):
    from data import make_loaders
    return make_loaders(cfg)


def _evaluate_model(model, val_loader, device, cfg, output_dir=None, save_predictions=False):
    from evaluate import evaluate_model
    return evaluate_model(model, val_loader, device, cfg, output_dir=output_dir,
                          save_predictions=save_predictions)


def _prepare_overfit(train_loader, val_loader, count: int) -> None:
    """Restrict a diagnostic run to a few distinct training observations.

    Validation also uses only a small subset of its usual observations. Its
    diagnostic scores must never be reported as full-validation benchmarks.
    """
    ds = train_loader.dataset
    if not hasattr(ds, "items"):
        raise ValueError("--overfit requires the project's FilamentDataset")
    seen, selected = set(), []
    for item in ds.items:
        if item["obs"] not in seen:
            selected.append(item)
            seen.add(item["obs"])
        if len(selected) == count:
            break
    if not selected:
        raise ValueError("No training observations are available for the overfit check")
    ds.items, ds.epoch_length = selected, len(selected)
    ds.tf = None
    ds.set_epoch(0)
    if hasattr(val_loader.dataset, "groups"):
        val_loader.dataset.groups = val_loader.dataset.groups[:count]


def train_one_epoch(model: nn.Module, loader, loss_fn, optimizer, device,
                    scaler, amp=False, gradient_clip=1.0, accumulation_steps=1) -> dict:
    """Train one epoch and reject invalid losses/gradients immediately."""
    if len(loader) == 0:
        raise ValueError("Training loader is empty; reduce batch_size or set drop_last=false")
    if not isinstance(accumulation_steps, int) or accumulation_steps < 1:
        raise ValueError("accumulation_steps must be a positive integer")
    model.train()
    started, loss_sum, examples, steps, skipped_steps = time.perf_counter(), 0.0, 0, 0, 0
    window_examples = 0
    normalizer = accumulation_steps * (loader.batch_size or 1)
    optimizer.zero_grad(set_to_none=True)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        started = time.perf_counter()
    for batch_index, batch in enumerate(loader):
        image = batch["image"].to(device, non_blocking=device.type == "cuda")
        target = batch["mask"].to(device, non_blocking=device.type == "cuda")
        with torch.autocast(device_type=device.type, enabled=amp, dtype=torch.float16):
            logits = model(image)
            loss = loss_fn(logits, target)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite training loss at optimizer step {steps}")
        batch_size = image.shape[0]
        # Weight by examples, then correct the final (possibly partial) window.
        # This handles a short final batch as well as fewer accumulation batches.
        scaler.scale(loss * batch_size / normalizer).backward()
        window_examples += batch_size
        if (batch_index + 1) % accumulation_steps == 0 or batch_index + 1 == len(loader):
            scaler.unscale_(optimizer)
            correction = normalizer / window_examples
            for parameter in model.parameters():
                if parameter.grad is not None:
                    parameter.grad.mul_(correction)
            norm_limit = float(gradient_clip) if gradient_clip else math.inf
            nn.utils.clip_grad_norm_(model.parameters(), norm_limit,
                                    error_if_nonfinite=not scaler.is_enabled())
            previous_scale = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            # AMP can legitimately skip an overflowing step while reducing scale.
            if scaler.get_scale() < previous_scale:
                skipped_steps += 1
            else:
                steps += 1
            optimizer.zero_grad(set_to_none=True)
            window_examples = 0
        loss_sum += float(loss.detach()) * batch_size
        examples += batch_size
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return dict(train_loss=loss_sum / examples, train_examples=examples,
                optimizer_steps=steps, skipped_optimizer_steps=skipped_steps,
                train_seconds=time.perf_counter() - started)


def run_training(cfg: dict, *, resume=None, max_epochs=None, overfit=0) -> dict:
    """Train an experiment; ``max_epochs`` limits this invocation's final epoch.

    For example, train.epochs=20 plus max_epochs=2 saves an epoch-2 checkpoint
    with the original 20-epoch cosine schedule. Resuming it continues at epoch 3.
    A run directory is never reused without an explicit matching checkpoint.
    """
    from config import save_config

    cfg = deepcopy(cfg)
    resume_state = None
    if resume:
        resume = Path(resume).resolve()
        resume_state = torch.load(resume, map_location="cpu", weights_only=False)
        required = {"model_state_dict", "optimizer_state_dict", "scheduler_state_dict",
                    "epoch", "config", "rng_state", "split_sha256"}
        if not required <= resume_state.keys():
            raise ValueError("Resume requires a full last.pt or best.pt training checkpoint")
        saved_overfit = int(resume_state.get("overfit_images", 0))
        if overfit and overfit != saved_overfit:
            raise ValueError("The resumed run's overfit setting cannot change")
        overfit = saved_overfit
    if overfit < 0:
        raise ValueError("overfit must be zero or a positive number of images")
    seed = int(cfg.get("seed", 42))
    cfg["data"]["seed"] = seed
    if overfit:
        cfg["data"].update(augment=False, drop_last=False,
                            batch_size=min(int(cfg["data"].get("batch_size", 4)), overfit))
        cfg.setdefault("diagnostic", {})["overfit_images"] = overfit
    tc = cfg["train"]
    total_epochs = int(tc["epochs"])
    stop_at = total_epochs if max_epochs is None else min(total_epochs, int(max_epochs))
    if total_epochs < 1 or stop_at < 1:
        raise ValueError("Epoch counts must be positive")
    validate_every = int(tc.get("validate_every", 1))
    if validate_every < 1:
        raise ValueError("validate_every must be at least one")
    primary = cfg["evaluation"].get("primary_metric", "PQ_dataset")
    output_dir = Path(cfg["run"]["output_dir"]).resolve()
    split_hash = _split_hash(cfg)
    if resume_state:
        if output_dir != resume.parent:
            raise ValueError("Resume must use the checkpoint's existing run.output_dir")
        if split_hash != resume_state["split_sha256"]:
            raise ValueError("The split contents changed since this checkpoint was saved")
        if _resume_signature(cfg) != _resume_signature(resume_state["config"]):
            raise ValueError("Resume settings differ from the checkpoint; keep the same model, loss, "
                             "data, seed, schedule horizon and evaluation settings")
    elif output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Run directory is not empty: {output_dir}. Choose a new name or --resume.")

    device = choose_device(tc.get("device", "auto"))
    seed_training(seed, tc.get("deterministic", True))
    train_loader, val_loader = _make_loaders(cfg["data"])
    if overfit:
        _prepare_overfit(train_loader, val_loader, overfit)
    if len(train_loader) == 0 or len(val_loader) == 0:
        raise ValueError("Training and validation loaders must both contain at least one batch")
    # Loading weights on resume must not redownload the initialization checkpoint.
    model_cfg = deepcopy(cfg["model"])
    if resume_state:
        model_cfg["encoder_weights"] = None
    model = build_model(model_cfg).to(device)
    loss_fn = build_loss(cfg["loss"]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(tc["learning_rate"]),
                                  weight_decay=float(tc.get("weight_decay", 0.0001)))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_epochs)
    amp = bool(tc.get("amp", True)) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    history, best_score, best_epoch = [], -math.inf, None
    start_epoch, bad_validations, total_seconds = 0, 0, 0.0
    if resume_state:
        model.load_state_dict(resume_state["model_state_dict"])
        optimizer.load_state_dict(resume_state["optimizer_state_dict"])
        scheduler.load_state_dict(resume_state["scheduler_state_dict"])
        if resume_state.get("scaler_state_dict"):
            scaler.load_state_dict(resume_state["scaler_state_dict"])
        start_epoch = int(resume_state["epoch"])
        history = list(resume_state.get("history", []))
        best_score = float(resume_state.get("best_score", -math.inf))
        best_epoch = resume_state.get("best_epoch")
        bad_validations = int(resume_state.get("bad_validations", 0))
        total_seconds = float(resume_state.get("total_seconds", 0.0))
        _restore_random_state(resume_state["rng_state"])
    if stop_at <= start_epoch:
        raise ValueError(f"Checkpoint already completed epoch {start_epoch}; stop limit is {stop_at}")

    output_dir.mkdir(parents=True, exist_ok=True)
    save_config(cfg, output_dir / "config.yaml")
    environment = _environment(device)
    source_hashes = _source_hashes()
    size = model_size(model)
    _write_json(dict(environment=environment, model=size, split_sha256=split_hash,
                     source_sha256=source_hashes,
                     overfit_images=overfit, primary_metric=primary,
                     evaluation_status="diagnostic_subset" if overfit else "full_validation",
                     resume_from=str(resume) if resume else None,
                     invocation_stop_epoch=stop_at), output_dir / "run_metadata.json")
    print(f"Run {cfg['run'].get('name', output_dir.name)} | device={device} | "
          f"parameters={size['parameters']:,} | primary={primary}", flush=True)
    if overfit:
        print("DIAGNOSTIC RUN: small training/validation subsets; not benchmark results.", flush=True)

    stop_reason = "invocation_limit" if stop_at < total_epochs else "completed"
    patience = int(tc.get("patience", 0))
    for epoch in range(start_epoch, stop_at):
        epoch_started = time.perf_counter()
        if hasattr(train_loader.dataset, "set_epoch"):
            train_loader.dataset.set_epoch(0 if overfit else epoch)
        if train_loader.generator is not None:
            train_loader.generator.manual_seed(seed + epoch)
        learning_rate = float(optimizer.param_groups[0]["lr"])
        row = dict(epoch=epoch + 1, learning_rate=learning_rate)
        row.update(train_one_epoch(model, train_loader, loss_fn, optimizer, device,
                                    scaler, amp, tc.get("gradient_clip", 1.0),
                                    tc.get("accumulation_steps", 1)))
        should_validate = epoch == 0 or (epoch + 1) % validate_every == 0 or epoch + 1 == total_epochs
        improved = False
        if should_validate:
            validation_started = time.perf_counter()
            model.eval()
            with torch.no_grad():
                validation = _evaluate_model(model, val_loader, device, cfg)
            if primary not in validation or not math.isfinite(float(validation[primary])):
                raise ValueError(f"Evaluation did not return finite primary metric {primary!r}")
            for key, value in validation.items():
                if isinstance(value, (float, int, np.number)):
                    numeric = float(value)
                    if not math.isfinite(numeric):
                        raise FloatingPointError(f"Validation metric {key} is not finite")
                    row[f"val_{key}"] = numeric
            row["validation_seconds"] = time.perf_counter() - validation_started
            score = float(validation[primary])
            improved = score > best_score
            if improved:
                best_score, best_epoch, bad_validations = score, epoch + 1, 0
            else:
                bad_validations += 1
        scheduler.step()
        row["epoch_seconds"] = time.perf_counter() - epoch_started
        total_seconds += row["epoch_seconds"]
        row["total_seconds"] = total_seconds
        history.append(row)
        checkpoint = dict(format_version=1, model_state_dict=model.state_dict(),
                          optimizer_state_dict=optimizer.state_dict(), scheduler_state_dict=scheduler.state_dict(),
                          scaler_state_dict=scaler.state_dict(), epoch=epoch + 1, best_score=best_score,
                          best_epoch=best_epoch, bad_validations=bad_validations, config=cfg,
                          rng_state=_random_state(), history=history, total_seconds=total_seconds,
                          split_sha256=split_hash, environment=environment,
                          source_sha256=source_hashes, overfit_images=overfit)
        _atomic_torch_save(checkpoint, output_dir / "last.pt")
        if improved:
            _atomic_torch_save(checkpoint, output_dir / "best.pt")
        _write_history(history, output_dir)
        score_text = f" {primary}={row['val_' + primary]:.4f}" if should_validate else ""
        print(f"Epoch {epoch + 1}/{total_epochs}: loss={row['train_loss']:.6f}{score_text} "
              f"train={row['train_seconds']:.1f}s total={row['epoch_seconds']:.1f}s", flush=True)
        if should_validate and patience > 0 and bad_validations >= patience:
            stop_reason = "early_stopping"
            break

    # One compact inference checkpoint per invocation; training checkpoints also
    # contain AdamW moments and are much larger than the actual model weights.
    best_path = output_dir / "best.pt"
    if not best_path.exists():
        raise RuntimeError("No best checkpoint exists; validation must run before completing a run")
    best = torch.load(best_path, map_location="cpu", weights_only=False)
    weights_path = output_dir / "best_weights.pt"
    _atomic_torch_save(dict(format_version=1, model_state_dict=best["model_state_dict"],
                            config=best["config"], epoch=best["epoch"], best_score=best["best_score"],
                            split_sha256=split_hash, overfit_images=overfit), weights_path)
    summary = dict(run_name=cfg["run"].get("name", output_dir.name), seed=seed,
                   output_dir=str(output_dir), primary_metric=primary, best_score=best_score,
                   best_epoch=best_epoch, epochs_completed=history[-1]["epoch"], stop_reason=stop_reason,
                   total_seconds=total_seconds, training_seconds=sum(r["train_seconds"] for r in history),
                   total_seconds_scope="epoch training plus validation; excludes setup and checkpoint serialization",
                   optimizer_steps=sum(r["optimizer_steps"] for r in history),
                   best_checkpoint=str(best_path), weights_checkpoint=str(weights_path),
                   weights_file_bytes=weights_path.stat().st_size,
                   training_checkpoint_bytes=(output_dir / "last.pt").stat().st_size,
                   split_sha256=split_hash, overfit_images=overfit, **size)
    best_row = next(row for row in history if row["epoch"] == best_epoch)
    summary["best_validation"] = {key[4:]: value for key, value in best_row.items()
                                   if key.startswith("val_")}
    _write_json(summary, output_dir / "summary.json")
    return summary


def main(argv=None) -> None:
    from config import load_config
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--resume", type=Path, help="Full last.pt or best.pt from the same run")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--max-epochs", type=int, help="Stop this invocation without changing the schedule horizon")
    parser.add_argument("--overfit", type=int, default=0, metavar="N", help="Diagnostic run on N training observations")
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    if args.device:
        cfg["train"]["device"] = args.device
    print(json.dumps(run_training(cfg, resume=args.resume, max_epochs=args.max_epochs,
                                  overfit=args.overfit), indent=2), flush=True)


if __name__ == "__main__":
    main()
