"""Compression helpers: distillation wiring, FP16 export and variant loading."""
import pytest
import torch

from compress import DistillationLoss, DistillationPair, fp16_command, load_variant, parameter_count
from losses import build_loss


class Tiny(torch.nn.Module):
    def __init__(self, scale=1.0):
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 1, 3, padding=1)
        self.norm = torch.nn.BatchNorm2d(1)
        torch.nn.init.constant_(self.conv.weight, scale)

    def forward(self, x):
        return self.norm(self.conv(x))


def test_pair_trains_only_student_and_keeps_teacher_in_eval_mode():
    student, teacher = Tiny(), Tiny(2.0)
    pair = DistillationPair(student, teacher)
    pair.train()
    assert student.training and not teacher.training
    assert parameter_count(pair) == parameter_count(student)
    logits = pair(torch.randn(2, 1, 8, 8))
    assert logits.shape == pair.teacher_logits.shape and not pair.teacher_logits.requires_grad


def test_distillation_loss_interpolates_label_and_teacher_terms():
    pair = DistillationPair(Tiny(), Tiny(2.0))
    image, target = torch.randn(2, 1, 8, 8), (torch.rand(2, 1, 8, 8) > 0.5).float()
    logits = pair(image)
    label = build_loss()
    teacher_term = torch.nn.functional.binary_cross_entropy_with_logits(
        logits, torch.sigmoid(pair.teacher_logits))
    assert float(DistillationLoss(label, pair, 0.0, 1.0)(logits, target)) == pytest.approx(float(label(logits, target)))
    assert float(DistillationLoss(label, pair, 1.0, 1.0)(logits, target)) == pytest.approx(float(teacher_term))
    with pytest.raises(ValueError):
        DistillationLoss(label, pair, 1.5, 1.0)


def test_fp16_export_halves_weights_and_reloads(tmp_path):
    from argparse import Namespace
    from models import build_model

    cfg = {"model": {"architecture": "unet", "encoder_name": "resnet18", "encoder_weights": None,
                     "in_channels": 1, "classes": 1}}
    model = build_model(cfg["model"]).eval()
    source = tmp_path / "weights.pt"
    torch.save({"config": cfg, "model_state_dict": model.state_dict()}, source)
    result = fp16_command(Namespace(teacher=str(source), output=str(tmp_path / "half.pt")))
    assert result["output_bytes"] < 0.6 * result["source_bytes"]
    half, saved_cfg, state = load_variant(tmp_path / "half.pt")
    assert state["precision"] == "fp16" and saved_cfg == cfg
    assert next(half.parameters()).dtype == torch.float16
    assert parameter_count(half) == parameter_count(model)

def test_train_variant_distills_keeps_best_epoch_and_writes_summary(tmp_path, monkeypatch):
    import json
    import data
    import evaluate
    from compress import train_variant
    from config import load_config
    from test_training import toy_loaders, toy_model

    def toy_validation(model, loader, device, cfg, *args, **kwargs):
        model.eval()
        with torch.no_grad():
            loss = sum(float(torch.nn.functional.binary_cross_entropy_with_logits(model(b["image"]), b["mask"]))
                       for b in loader)
        score = 1.0 / (1.0 + loss)
        return dict(PQ_dataset=score, SQ_dataset=score, RQ_dataset=score, tp=1, fp=0, fn=0,
                    custom_one_to_many=0, custom_many_to_one=0, pixel_dice_mean=score, filament_dice_mean=score)

    monkeypatch.setattr(data, "make_loaders", toy_loaders)
    monkeypatch.setattr(evaluate, "evaluate_model", toy_validation)
    split = tmp_path / "split.json"
    split.write_text('{"train":["a"],"val":["b"]}', encoding="utf8")
    cfg = load_config({"run": {"name": "toy_student", "output_dir": str(tmp_path / "unused")}, "seed": 3,
                       "data": {"split": str(split)}, "model": {"encoder_weights": None},
                       "train": {"device": "cpu", "epochs": 3, "amp": False, "deterministic": False}})
    torch.manual_seed(0)
    student, teacher = toy_model(cfg), toy_model(cfg)
    result = train_variant(student, cfg, tmp_path / "run",
                           make_checkpoint=lambda m, c, extra: dict(config=c, model_state_dict=m.state_dict(), **extra),
                           teacher=teacher, kd_weight=0.5, validate_from=2)
    summary = json.loads((tmp_path / "run" / "summary.json").read_text())
    history = json.loads((tmp_path / "run" / "history.json").read_text())
    saved = torch.load(tmp_path / "run" / "best_weights.pt", weights_only=False)
    assert summary == result and summary["parameters"] == parameter_count(student)
    assert summary["kd_weight"] == 0.5 and len(history) == 3 and "val_PQ_dataset" not in history[0]
    best = max(history[1:], key=lambda r: r["val_PQ_dataset"])
    assert saved["epoch"] == summary["best_epoch"] == best["epoch"]
    with pytest.raises(FileExistsError):
        train_variant(student, cfg, tmp_path / "run", make_checkpoint=lambda *a: {})

def test_onnx_runtime_wrapper_matches_pytorch_through_shared_inference(tmp_path):
    import json
    import warnings
    from compress_onnx import OnnxRuntimeModel, load_onnx_variant, sidecar
    from inference import predict_probability

    model = Tiny(0.3).eval()
    path = tmp_path / "tiny.onnx"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        torch.onnx.export(model, (torch.randn(1, 1, 32, 32),), str(path), input_names=["image"],
                          output_names=["logits"], opset_version=17, dynamo=False,
                          dynamic_axes={"image": {2: "h", 3: "w"}, "logits": {2: "h", 3: "w"}})
    image = torch.randn(1, 1, 64, 96)
    expected = predict_probability(model, image, {"amp": False})
    actual = predict_probability(OnnxRuntimeModel(path), image, {"amp": False})
    assert torch.allclose(actual, expected, atol=1e-5)
    sidecar(path).write_text(json.dumps({"config": {"model": {}}, "parameters": 11}), encoding="utf8")
    wrapped, cfg, info = load_onnx_variant(path)
    assert info["parameters"] == 11 and cfg == {"model": {}}
    with pytest.raises(ValueError, match="CPU"):
        load_onnx_variant(path, "cuda")
