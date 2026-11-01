"""Small, explicit YAML configuration shared by training and prediction.

Paths are relative to the repository, not the shell's current directory. An
experiment may ``extend`` another YAML file (relative to that YAML's directory).
Unknown keys are rejected so a typo cannot silently change an experiment.
"""
from __future__ import annotations

import copy
import os
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent

DEFAULTS = {
    "run": {"name": "baseline", "output_dir": "runs/baseline_seed42"},
    "seed": 42,
    "data": {
        "data_root": "data", "cache_dir": "cache", "split": "splits/split_v1.json",
        "crop": 512, "train_resize": None, "pos_frac": 0.7,
        "batch_size": 4, "num_workers": 0, "val_num_workers": 0,
        "target": "per_set", "epoch_length": None, "augment": True,
        "seed": 42, "pin_memory": True, "drop_last": True,
        "preprocess": {"mask_disc": False, "clahe": False, "clahe_clip": 2.0,
                       "clahe_tile": 8, "local_norm": False, "local_sigma": 60.0,
                       "mean": 0.5, "std": 0.25},
        "aug": {"hflip": 0.5, "vflip": 0.5, "rot90": 0.5, "affine": 0.5,
                "rotate_limit": 30, "scale_limit": 0.2,
                "brightness_contrast": 0.5, "brightness_limit": 0.2,
                "contrast_limit": 0.2, "noise": 0.3, "blur": 0.2},
    },
    "model": {"architecture": "unet", "encoder_name": "resnet34",
              "encoder_weights": "imagenet", "in_channels": 1, "classes": 1},
    "loss": {"name": "bce_dice", "bce_weight": 1.0, "dice_weight": 1.0,
             "alpha": 0.3, "beta": 0.7, "smooth": 1.0, "pos_weight": None},
    "train": {"epochs": 20, "learning_rate": 0.0003, "weight_decay": 0.0001,
              "amp": True, "patience": 8, "validate_every": 1,
              "gradient_clip": 1.0, "deterministic": True, "device": "auto",
              "accumulation_steps": 1},
    "inference": {"tile_size": None, "overlap": 128, "resize": None, "amp": True},
    "evaluation": {"threshold": 0.5, "iou_threshold": 0.5,
                   "primary_metric": "PQ_dataset", "save_predictions": False,
                   "relation_tau": 0.1, "relation_own": 0.5, "progress": True},
    "diagnostic": {},
}


def merge_config(base: dict, override: dict, prefix: str = "") -> dict:
    """Recursively override known settings, without modifying either input."""
    result = copy.deepcopy(base)
    if not isinstance(override, dict):
        raise ValueError("Configuration must be a mapping")
    for key, value in override.items():
        path = f"{prefix}.{key}" if prefix else key
        if key not in result:
            raise ValueError(f"Unknown configuration key: {path}")
        if isinstance(result[key], dict):
            if not isinstance(value, dict):
                raise ValueError(f"{path} must be a mapping")
            result[key] = (copy.deepcopy(value) if path == "diagnostic"
                           else merge_config(result[key], value, path))
        else:
            result[key] = copy.deepcopy(value)
    return result


def _read_yaml(path: Path, seen: set[Path]) -> dict:
    path = path.resolve()
    if path in seen:
        raise ValueError(f"Circular config inheritance: {path}")
    content = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(content, dict):
        raise ValueError(f"Configuration must be a mapping: {path}")
    parent = content.pop("extends", None)
    if parent is None:
        return content
    inherited = _read_yaml(path.parent / parent, seen | {path})
    def overlay(base, additions):
        result = copy.deepcopy(base)
        for key, value in additions.items():
            if isinstance(value, dict) and isinstance(result.get(key), dict):
                result[key] = overlay(result[key], value)
            else:
                result[key] = copy.deepcopy(value)
        return result
    # Strict validation occurs against DEFAULTS after inheritance. Keeping this
    # sparse preserves environment defaults for paths not explicitly supplied.
    return overlay(inherited, content)


def validate_config(cfg: dict) -> None:
    """Check a compact set of constraints that otherwise fail late in a run."""
    d, t, m, e, inf = (cfg[k] for k in ("data", "train", "model", "evaluation", "inference"))
    for key in ("batch_size",):
        if not isinstance(d[key], int) or d[key] < 1:
            raise ValueError(f"data.{key} must be a positive integer")
    for key in ("num_workers", "val_num_workers", "crop"):
        if not isinstance(d[key], int) or d[key] < 0:
            raise ValueError(f"data.{key} must be a nonnegative integer")
    if d["crop"] and d["crop"] % 32:
        raise ValueError("data.crop must be divisible by 32 (or zero for whole images)")
    if d["target"] not in ("per_set", "consensus"):
        raise ValueError("data.target must be per_set or consensus")
    if not 0 <= d["pos_frac"] <= 1 or d["preprocess"]["std"] <= 0:
        raise ValueError("pos_frac must be in [0,1] and normalization std must be positive")
    if d["epoch_length"] is not None and d["epoch_length"] < 1:
        raise ValueError("data.epoch_length must be positive or null")
    if t["epochs"] < 1 or t["validate_every"] < 1 or t["learning_rate"] <= 0:
        raise ValueError("epochs, validate_every and learning_rate must be positive")
    if not isinstance(t["accumulation_steps"], int) or t["accumulation_steps"] < 1:
        raise ValueError("train.accumulation_steps must be a positive integer")
    if t["patience"] < 0 or t["weight_decay"] < 0 or t["gradient_clip"] < 0:
        raise ValueError("patience, weight_decay and gradient_clip cannot be negative")
    if m["architecture"] not in ("unet", "unetplusplus"):
        raise ValueError("model.architecture must be unet or unetplusplus")
    if m["in_channels"] != 1 or m["classes"] != 1:
        raise ValueError("This binary grayscale pipeline requires one input and one output channel")
    if cfg["loss"]["name"] not in ("bce_dice", "bce_tversky"):
        raise ValueError("loss.name must be bce_dice or bce_tversky")
    if not 0 < e["threshold"] < 1 or not 0.5 <= e["iou_threshold"] < 1:
        raise ValueError("Foreground threshold must be (0,1); PQ IoU threshold must be [0.5,1)")
    for name, value in (("data.train_resize", d["train_resize"]),
                        ("inference.resize", inf["resize"]),
                        ("inference.tile_size", inf["tile_size"])):
        if value is not None and (not isinstance(value, int) or value < 32 or value % 32):
            raise ValueError(f"{name} must be null or a positive multiple of 32")
    if inf["overlap"] < 0 or (inf["tile_size"] and inf["overlap"] >= inf["tile_size"]):
        raise ValueError("inference.overlap must be nonnegative and smaller than tile_size")
    if d["train_resize"] and d["crop"] > d["train_resize"]:
        raise ValueError("data.crop cannot exceed data.train_resize")


def load_config(path: str | Path | dict | None = None) -> dict:
    """Load a resolved run config. FILAMENT_DATA/CACHE override default paths."""
    defaults = copy.deepcopy(DEFAULTS)
    defaults["data"]["data_root"] = os.environ.get("FILAMENT_DATA", "data")
    defaults["data"]["cache_dir"] = os.environ.get("FILAMENT_CACHE", "cache")
    content = path if isinstance(path, dict) else (_read_yaml(Path(path), set()) if path else {})
    cfg = merge_config(defaults, content)
    cfg["data"]["seed"] = cfg["seed"]
    for section, key in (("data", "data_root"), ("data", "cache_dir"),
                         ("data", "split"), ("run", "output_dir")):
        p = Path(os.path.expandvars(str(cfg[section][key]))).expanduser()
        cfg[section][key] = str((p if p.is_absolute() else ROOT / p).resolve())
    validate_config(cfg)
    return cfg


def save_config(cfg: dict, path: str | Path) -> None:
    """Save the complete settings next to a run, with real newlines."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
