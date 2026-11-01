"""Person C compression study: smaller or cheaper versions of the selected model.

Each variant is compared with the same evidence: pooled validation PQ on the
unchanged split (``evaluate.evaluate_model``), parameter count, deployable file
size, and the complete image-to-CSV time of ``predict.predict_test`` on the
180 test images (same GPU, batch size one, model loading excluded).

Subcommands (run from the repository root):

``distill``   train a smaller U-Net student from the labels plus the teacher's
              per-pixel probabilities (knowledge distillation). ``--kd-weight 0``
              trains the identical label-only control with the same code path.
``prune``     L1-magnitude structured channel pruning of the teacher
              (torch-pruning), followed by fine-tuning on the training split.
``fp16``      store the teacher's weights in half precision.
``benchmark`` validation scores (optionally probability archives) and end-to-end
              test timing for one variant at a chosen numerical precision.

Training reuses the project's data loaders, loss, ``train.train_one_epoch`` and
validation code, so only the deliberate change differs between variants.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time
from copy import deepcopy
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from evaluate import file_sha256, resolve_device

VALIDATION_KEYS = ("PQ_dataset", "SQ_dataset", "RQ_dataset", "tp", "fp", "fn",
                   "custom_one_to_many", "custom_many_to_one", "pixel_dice_mean", "filament_dice_mean")


# --------------------------------------------------------------------------- #
# Loading and saving variants
# --------------------------------------------------------------------------- #
def parameter_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def load_variant(path, device="cpu"):
    """Load a project checkpoint, a student, an FP16 copy or a pruned module.

    Returns ``(model_in_eval_mode, saved_config, checkpoint_dictionary)``. Only
    load files produced by this project: pruned modules are pickled objects.
    """
    from models import load_checkpoint_model

    if Path(path).suffix.lower() == ".onnx":
        from compress_onnx import load_onnx_variant
        return load_onnx_variant(path, device)
    state = torch.load(Path(path), map_location="cpu", weights_only=False)
    if "pruned_module" in state:
        model = state["pruned_module"]
    else:
        model, _, _ = load_checkpoint_model(path, "cpu")
    if state.get("precision") == "fp16":
        model = model.half()
    return model.to(device).eval(), deepcopy(state["config"]), state


def _save(value, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def _dump(value, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf8")


# --------------------------------------------------------------------------- #
# Knowledge distillation
# --------------------------------------------------------------------------- #
class DistillationPair(nn.Module):
    """Student wrapper that also records the frozen teacher's logits.

    The teacher is kept outside the registered submodules, so ``parameters()``,
    gradient clipping and ``train()`` only affect the student and the teacher
    always stays in evaluation mode (fixed batch-norm statistics).
    """

    def __init__(self, student: nn.Module, teacher: nn.Module):
        super().__init__()
        self.student = student
        self._teacher = [teacher.eval()]
        self.teacher_logits = None

    def train(self, mode: bool = True):
        self.training = mode
        self.student.train(mode)
        return self

    def forward(self, image):
        with torch.no_grad():
            self.teacher_logits = self._teacher[0](image).detach()
        return self.student(image)


class DistillationLoss(nn.Module):
    """(1 - w) * label loss + w * T^2 * BCE(student / T, sigmoid(teacher / T))."""

    def __init__(self, label_loss: nn.Module, pair: DistillationPair, weight: float, temperature: float):
        super().__init__()
        if not 0 <= weight <= 1 or temperature <= 0:
            raise ValueError("kd weight must be in [0,1] and temperature positive")
        self.label_loss, self.pair = label_loss, [pair]
        self.weight, self.temperature = float(weight), float(temperature)

    def forward(self, logits, target):
        teacher = self.pair[0].teacher_logits
        if teacher is None or teacher.shape != logits.shape:
            raise ValueError("teacher logits are missing or mismatched")
        t = self.temperature
        soft = torch.sigmoid(teacher.float() / t)
        kd = F.binary_cross_entropy_with_logits(logits.float() / t, soft) * (t * t)
        return (1 - self.weight) * self.label_loss(logits, target) + self.weight * kd


# --------------------------------------------------------------------------- #
# Shared training loop for students and pruned models
# --------------------------------------------------------------------------- #
def train_variant(model: nn.Module, cfg: dict, output_dir, *, make_checkpoint, teacher=None,
                  kd_weight: float = 0.0, temperature: float = 1.0, validate_from: int = 1,
                  record: dict | None = None) -> dict:
    """Train/fine-tune ``model`` with the project's loop; keep the best-PQ epoch.

    ``make_checkpoint(model, cfg, extra)`` returns the dictionary to save as
    ``best_weights.pt`` (no optimizer state). Validation uses the unchanged
    split and the config's baseline instance conversion, as in Person B's study.
    """
    from config import save_config
    from data import make_loaders
    from evaluate import evaluate_model
    from losses import build_loss
    from train import _environment, _split_hash, _write_history, choose_device, seed_training, train_one_epoch

    out = Path(output_dir)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"{out} is not empty; choose a new output directory")
    out.mkdir(parents=True, exist_ok=True)
    cfg = deepcopy(cfg)
    cfg["run"]["output_dir"] = str(out.resolve())
    tc, seed = cfg["train"], int(cfg["seed"])
    epochs = int(tc["epochs"])
    device = choose_device(tc.get("device", "auto"))
    seed_training(seed, tc.get("deterministic", True))
    train_loader, val_loader = make_loaders(cfg["data"])
    model = model.to(device)
    label_loss = build_loss(cfg["loss"]).to(device)
    if teacher is not None and kd_weight > 0:
        network = DistillationPair(model, teacher.to(device))
        loss_fn = DistillationLoss(label_loss, network, kd_weight, temperature)
    else:
        network, loss_fn = model, label_loss
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(tc["learning_rate"]),
                                  weight_decay=float(tc.get("weight_decay", 0.0001)))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    amp = bool(tc.get("amp", True)) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    save_config(cfg, out / "config.yaml")
    metadata = dict(environment=_environment(device), split_sha256=_split_hash(cfg),
                    parameters=parameter_count(model), kd_weight=kd_weight if teacher is not None else 0.0,
                    temperature=temperature, validate_from_epoch=validate_from, **(record or {}))
    _dump(metadata, out / "run_metadata.json")
    print(f"{cfg['run']['name']}: {metadata['parameters']:,} parameters on {device}; "
          f"kd_weight={metadata['kd_weight']}", flush=True)
    history, best_score, best_epoch, training_seconds = [], -math.inf, None, 0.0
    started = time.perf_counter()
    for epoch in range(epochs):
        train_loader.dataset.set_epoch(epoch)
        train_loader.generator.manual_seed(seed + epoch)
        row = dict(epoch=epoch + 1, learning_rate=float(optimizer.param_groups[0]["lr"]))
        row.update(train_one_epoch(network, train_loader, loss_fn, optimizer, device, scaler, amp,
                                   tc.get("gradient_clip", 1.0), tc.get("accumulation_steps", 1)))
        training_seconds += row["train_seconds"]
        scheduler.step()
        if epoch + 1 >= validate_from or epoch + 1 == epochs:
            before = time.perf_counter()
            summary = evaluate_model(model, val_loader, device, cfg)
            row.update({f"val_{k}": summary[k] for k in VALIDATION_KEYS}, validation_seconds=time.perf_counter() - before)
            if summary["PQ_dataset"] > best_score:
                best_score, best_epoch = float(summary["PQ_dataset"]), epoch + 1
                _save(make_checkpoint(model, cfg, dict(epoch=epoch + 1, best_score=best_score)),
                      out / "best_weights.pt")
        history.append(row)
        _write_history(history, out)
        print(f"epoch {epoch + 1}/{epochs}: loss {row['train_loss']:.4f}"
              + (f", val PQ {row['val_PQ_dataset']:.4f}" if "val_PQ_dataset" in row else ""), flush=True)
    result = dict(metadata)
    result.update(run_name=cfg["run"]["name"], best_score=best_score, best_epoch=best_epoch,
                  epochs_completed=epochs, training_seconds=training_seconds,
                  total_seconds=time.perf_counter() - started,
                  total_seconds_scope="training plus validation epochs; excludes setup",
                  parameters=parameter_count(model), weights_file_bytes=(out / "best_weights.pt").stat().st_size,
                  weights_sha256=file_sha256(out / "best_weights.pt"))
    _dump(result, out / "summary.json")
    return result


def _teacher(path, cfg_path):
    from config import load_config
    from models import load_checkpoint_model

    teacher, teacher_cfg, _ = load_checkpoint_model(path, "cpu")
    cfg = load_config(cfg_path)
    if cfg["model"] != teacher_cfg["model"]:
        raise ValueError("--config must describe the teacher checkpoint's model")
    return teacher.eval(), cfg


def distill_command(args) -> dict:
    from models import build_model
    from train import seed_training

    teacher, cfg = _teacher(args.teacher, args.config)
    student_cfg = deepcopy(cfg)
    student_cfg["model"].update(encoder_name=args.encoder, encoder_weights=args.encoder_weights)
    student_cfg["run"]["name"] = args.name
    if args.epochs:
        student_cfg["train"]["epochs"] = int(args.epochs)
    # Seed before construction so the distilled and control students start identically.
    seed_training(int(student_cfg["seed"]), student_cfg["train"].get("deterministic", True))
    student = build_model(student_cfg["model"])

    def checkpoint(model, cfg_, extra):
        return dict(config=cfg_, model_state_dict={k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                    teacher_sha256=file_sha256(args.teacher), kd_weight=args.kd_weight,
                    temperature=args.temperature, **extra)

    return train_variant(student, student_cfg, args.output, make_checkpoint=checkpoint,
                         teacher=teacher if args.kd_weight > 0 else None, kd_weight=args.kd_weight,
                         temperature=args.temperature, validate_from=args.validate_from,
                         record=dict(variant="distillation" if args.kd_weight > 0 else "label_only_control",
                                     teacher=str(Path(args.teacher).resolve()),
                                     teacher_sha256=file_sha256(args.teacher),
                                     student_encoder=args.encoder, teacher_parameters=parameter_count(teacher)))


def prune_command(args) -> dict:
    import torch_pruning as tp

    teacher, cfg = _teacher(args.teacher, args.config)
    model = deepcopy(teacher).float().eval()
    before = parameter_count(model)
    example = torch.randn(1, 1, 512, 512)
    pruner = tp.pruner.MagnitudePruner(model, example, importance=tp.importance.MagnitudeImportance(p=1),
                                       pruning_ratio=args.ratio, ignored_layers=[model.segmentation_head])
    pruner.step()
    after = parameter_count(model)
    cfg["run"]["name"] = args.name
    cfg["train"].update(epochs=int(args.epochs), learning_rate=float(args.learning_rate))

    def checkpoint(module, cfg_, extra):
        return dict(config=cfg_, pruned_module=deepcopy(module).cpu().eval(),
                    pruning=dict(method="L1 magnitude structured channel pruning (torch-pruning)",
                                 ratio=args.ratio, parameters_before=before, parameters_after=after),
                    source_checkpoint_sha256=file_sha256(args.teacher), **extra)

    print(f"pruned {args.ratio:.0%} of channels per layer: {before:,} -> {after:,} parameters", flush=True)
    return train_variant(model, cfg, args.output, make_checkpoint=checkpoint, validate_from=args.validate_from,
                         record=dict(variant="structured_pruning", pruning_ratio=args.ratio,
                                     parameters_before=before, parameters_after=after,
                                     source_checkpoint=str(Path(args.teacher).resolve()),
                                     source_checkpoint_sha256=file_sha256(args.teacher)))


def fp16_command(args) -> dict:
    state = torch.load(Path(args.teacher), map_location="cpu", weights_only=False)
    output = Path(args.output)
    if "pruned_module" in state:
        value = dict(state, pruned_module=deepcopy(state["pruned_module"]).half(), precision="fp16",
                     source_checkpoint_sha256=file_sha256(args.teacher))
    else:
        halved = {k: (v.half() if v.is_floating_point() else v) for k, v in state["model_state_dict"].items()}
        value = dict(config=state["config"], model_state_dict=halved, precision="fp16",
                     source_checkpoint_sha256=file_sha256(args.teacher))
    _save(value, output)
    result = dict(source=str(Path(args.teacher).resolve()), output=str(output.resolve()),
                  source_bytes=Path(args.teacher).stat().st_size, output_bytes=output.stat().st_size,
                  output_sha256=file_sha256(output))
    print(json.dumps(result, indent=2))
    return result


# --------------------------------------------------------------------------- #
# Benchmark: validation quality, deployable size and end-to-end test timing
# --------------------------------------------------------------------------- #
def benchmark_command(args) -> dict:
    import numpy as np
    from config import load_config, save_config
    from data import make_loaders
    from evaluate import evaluate_model
    from predict import predict_test

    device = resolve_device(args.device)
    model, saved_cfg, state = load_variant(args.variant, device)
    cfg = load_config(args.config)
    onnx = Path(args.variant).suffix.lower() == ".onnx"
    if onnx or args.precision == "fp32":
        model = model.float()
        cfg["inference"]["amp"] = False
    elif args.precision == "amp16":
        model = model.float()
        cfg["inference"]["amp"] = True
    else:
        model = model.half()
        cfg["inference"]["amp"] = True
    out = Path(args.output)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"{out} is not empty; choose a new output directory")
    out.mkdir(parents=True)
    save_config(cfg, out / "config.yaml")
    result = dict(name=args.name, variant=str(Path(args.variant).resolve()), variant_sha256=file_sha256(args.variant),
                  precision=state.get("precision", args.precision) if onnx else args.precision, device=str(device),
                  gpu=torch.cuda.get_device_name(device) if device.type == "cuda" else None,
                  parameters=state["parameters"] if onnx else parameter_count(model),
                  deployable_file_bytes=Path(args.variant).stat().st_size,
                  inference=cfg["inference"], foreground_threshold=cfg["evaluation"]["threshold"],
                  postprocess=cfg["postprocess"], model_architecture=saved_cfg["model"])
    if not args.skip_validation:
        _, val_loader = make_loaders(cfg["data"])
        summary = evaluate_model(model, val_loader, device, cfg, out / "validation", args.save_predictions)
        result["validation"] = {k: summary[k] for k in VALIDATION_KEYS}
        result["validation_prediction_ms_per_image"] = summary["prediction_ms_per_image"]
    if not args.skip_test:
        import shutil

        data_root = Path(cfg["data"]["data_root"])
        files = sorted((data_root / "test" / "test_images").iterdir())
        test_cfg, csv_name = cfg, "submission.csv"
        if args.test_subset:
            # Slow CPU variants: time the first N test images only; never a submission file.
            subset_root = out / "test_subset"
            (subset_root / "test" / "test_images").mkdir(parents=True)
            files = files[: args.test_subset]
            for path in files:
                shutil.copy2(path, subset_root / "test" / "test_images" / path.name)
            test_cfg = deepcopy(cfg)
            test_cfg["data"]["data_root"] = str(subset_root)
            csv_name = "timing_subset_not_a_submission.csv"
        for path in files:  # warm the file-system cache, as in Person B's timing protocol
            path.read_bytes()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        report = predict_test(model, test_cfg, device, out / csv_name)
        times = np.array([r["seconds"] for r in report["images"]])
        result["test"] = dict(n_images=report["n_images"], n_filaments=report["n_filaments"],
                              subset_of_test_images=args.test_subset or None,
                              zero_detection_images=report["zero_detection_images"],
                              end_to_end_seconds=report["end_to_end_seconds"],
                              end_to_end_ms_per_image=report["end_to_end_ms_per_image"],
                              ms_per_image_excluding_first=float(1000 * times[1:].mean()),
                              median_ms_per_image=float(1000 * np.median(times)),
                              timing_scope=report["timing_scope"] + "; model loading excluded",
                              csv_sha256=file_sha256(out / csv_name))
    _dump(result, out / "variant.json")
    print(json.dumps({k: v for k, v in result.items() if k in ("name", "precision", "parameters",
                                                             "deployable_file_bytes", "validation", "test")}, indent=2))
    return result


def main(argv=None) -> None:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("distill", "prune"):
        p = sub.add_parser(name)
        p.add_argument("--teacher", required=True, help="teacher/source best_weights.pt")
        p.add_argument("--config", required=True, help="portable config matching the teacher")
        p.add_argument("--output", required=True)
        p.add_argument("--name", required=True)
        p.add_argument("--validate-from", type=int, default=1, help="first epoch with full validation")
        if name == "distill":
            p.add_argument("--encoder", default="resnet18")
            p.add_argument("--encoder-weights", default="imagenet")
            p.add_argument("--kd-weight", type=float, default=0.5)
            p.add_argument("--temperature", type=float, default=1.0)
            p.add_argument("--epochs", type=int, help="default: the config's train.epochs")
        else:
            p.add_argument("--ratio", type=float, default=0.5, help="fraction of channels removed per layer")
            p.add_argument("--epochs", type=int, default=10)
            p.add_argument("--learning-rate", type=float, default=1e-4)
    p = sub.add_parser("fp16")
    p.add_argument("--teacher", required=True)
    p.add_argument("--output", required=True)
    p = sub.add_parser("benchmark")
    p.add_argument("--variant", required=True)
    p.add_argument("--name", required=True)
    p.add_argument("--config", required=True, help="data paths, threshold, postprocess and inference settings")
    p.add_argument("--output", required=True)
    p.add_argument("--precision", choices=("fp32", "amp16", "fp16"), default="amp16")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--save-predictions", action="store_true")
    p.add_argument("--skip-validation", action="store_true")
    p.add_argument("--skip-test", action="store_true")
    p.add_argument("--test-subset", type=int, help="time only the first N test images (slow CPU variants)")
    args = parser.parse_args(argv)
    commands = dict(distill=distill_command, prune=prune_command, fp16=fp16_command, benchmark=benchmark_command)
    commands[args.command](args)


if __name__ == "__main__":
    main()
