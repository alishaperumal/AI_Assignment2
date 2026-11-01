"""Person C: ONNX export and INT8 quantisation for CPU deployment.

``export`` writes the teacher as an FP32 ONNX graph with dynamic image size and
checks that ONNX Runtime reproduces PyTorch's probabilities. ``int8`` applies
ONNX Runtime static quantisation (QDQ format, per-channel INT8 weights, UINT8
activations) calibrated on crops of *training* images only, so no validation or
test pixels influence the quantised model. A JSON sidecar next to each ONNX file
keeps the source config and provenance; ``compress.py benchmark`` loads these
files through ``load_onnx_variant`` and runs them on the CPU.

    python compress_onnx.py export --teacher runs/person_b_final/best_weights.pt \
        --config configs/person_c/seed45_selected.yaml --output runs/person_c/compression/teacher_fp32.onnx
    python compress_onnx.py int8 --onnx runs/person_c/compression/teacher_fp32.onnx \
        --output runs/person_c/compression/teacher_int8.onnx
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from evaluate import file_sha256


class OnnxRuntimeModel(nn.Module):
    """Expose an ONNX Runtime CPU session as a logits-producing module."""

    def __init__(self, path, threads: int | None = None):
        super().__init__()
        import onnxruntime as ort

        options = ort.SessionOptions()
        if threads:
            options.intra_op_num_threads = int(threads)
        self.session = ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name

    def forward(self, image):
        logits = self.session.run(None, {self.input_name: image.detach().float().cpu().numpy()})[0]
        return torch.from_numpy(logits).to(image.device)


def sidecar(path) -> Path:
    return Path(str(path) + ".json")


def load_onnx_variant(path, device="cpu"):
    if torch.device(device).type != "cpu":
        raise ValueError("ONNX variants run on the CPU execution provider; use --device cpu")
    info = json.loads(sidecar(path).read_text(encoding="utf8"))
    return OnnxRuntimeModel(path).eval(), info["config"], info


def _normalized_images(cfg, observations, crop=None):
    """Normalised float32 [1,1,H,W] arrays, optionally centre-cropped."""
    import data as D

    catalog = D.load_catalog(Path(cfg["data"]["cache_dir"]))
    pre = cfg["data"]["preprocess"]
    for obs in observations:
        name = catalog.loc[catalog.obs == obs, "file_name"].iloc[0]
        x = D.normalise(D.preprocess_image(D.load_image(Path(cfg["data"]["data_root"]) / "train" / "train_images" / name), pre), pre)
        if crop:
            top = (x.shape[0] - crop) // 2
            x = x[top:top + crop, top:top + crop]
        yield np.ascontiguousarray(x[None, None], dtype=np.float32)


def export_command(args) -> dict:
    import warnings
    import data as D
    from config import load_config
    from models import load_checkpoint_model

    model, saved_cfg, _ = load_checkpoint_model(args.teacher, "cpu")
    cfg = load_config(args.config)
    if cfg["model"] != saved_cfg["model"]:
        raise ValueError("--config must describe the checkpoint's model")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        torch.onnx.export(model.float().eval(), (torch.randn(1, 1, 512, 512),), str(output),
                          input_names=["image"], output_names=["logits"], opset_version=17,
                          dynamic_axes={"image": {0: "batch", 2: "height", 3: "width"},
                                        "logits": {0: "batch", 2: "height", 3: "width"}}, dynamo=False)
    split = D.load_split(Path(cfg["data"]["split"]))
    image = next(_normalized_images(cfg, sorted(split["train"])[:1]))
    with torch.no_grad():
        expected = torch.sigmoid(model(torch.from_numpy(image))).numpy()
    actual = 1 / (1 + np.exp(-OnnxRuntimeModel(output)(torch.from_numpy(image)).numpy()))
    info = dict(config=cfg, precision="onnx-fp32", parameters=sum(p.numel() for p in model.parameters()),
                source_checkpoint=str(Path(args.teacher).resolve()), source_checkpoint_sha256=file_sha256(args.teacher),
                onnx_sha256=file_sha256(output), onnx_bytes=output.stat().st_size,
                check_image="first training observation (full 2048 x 2048)",
                max_abs_probability_difference_vs_pytorch=float(np.abs(actual - expected).max()))
    sidecar(output).write_text(json.dumps(info, indent=2) + "\n", encoding="utf8")
    print(json.dumps({k: v for k, v in info.items() if k != "config"}, indent=2))
    return info


def int8_command(args) -> dict:
    import data as D
    from onnxruntime.quantization import (CalibrationDataReader, CalibrationMethod, QuantFormat, QuantType,
                                          quantize_static)

    source = json.loads(sidecar(args.onnx).read_text(encoding="utf8"))
    cfg = source["config"]
    split = D.load_split(Path(cfg["data"]["split"]))
    rng = np.random.default_rng(args.seed)
    observations = sorted(rng.choice(sorted(split["train"]), size=args.calibration_images, replace=False))

    class Reader(CalibrationDataReader):
        def __init__(self):
            self.images = _normalized_images(cfg, observations, crop=args.crop)

        def get_next(self):
            image = next(self.images, None)
            return None if image is None else {"image": image}

    output = Path(args.output)
    started = time.perf_counter()
    quantize_static(str(args.onnx), str(output), Reader(), quant_format=QuantFormat.QDQ, per_channel=True,
                    activation_type=QuantType.QUInt8, weight_type=QuantType.QInt8,
                    calibrate_method=CalibrationMethod.MinMax)
    image = next(_normalized_images(cfg, sorted(split["train"])[:1]))
    reference = OnnxRuntimeModel(args.onnx)(torch.from_numpy(image)).numpy()
    quantized = OnnxRuntimeModel(output)(torch.from_numpy(image)).numpy()
    sigmoid = lambda z: 1 / (1 + np.exp(-z))
    info = dict(config=cfg, precision="onnx-int8", parameters=source["parameters"],
                source_onnx=str(Path(args.onnx).resolve()), source_onnx_sha256=file_sha256(args.onnx),
                onnx_sha256=file_sha256(output), onnx_bytes=output.stat().st_size,
                quantization=dict(method="ONNX Runtime static QDQ", weights="INT8 per-channel", activations="UINT8",
                                  calibration="MinMax", calibration_images=len(observations),
                                  calibration_source="training split only", crop=args.crop, seed=args.seed,
                                  seconds=time.perf_counter() - started),
                calibration_observations=[str(o) for o in observations],
                max_abs_probability_difference_vs_fp32=float(np.abs(sigmoid(quantized) - sigmoid(reference)).max()),
                foreground_agreement_vs_fp32=float(((sigmoid(quantized) > 0.5) == (sigmoid(reference) > 0.5)).mean()))
    sidecar(output).write_text(json.dumps(info, indent=2) + "\n", encoding="utf8")
    print(json.dumps({k: v for k, v in info.items() if k not in ("config", "calibration_observations")}, indent=2))
    return info


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("export")
    p.add_argument("--teacher", required=True)
    p.add_argument("--config", required=True)
    p.add_argument("--output", required=True)
    p = sub.add_parser("int8")
    p.add_argument("--onnx", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--calibration-images", type=int, default=8)
    p.add_argument("--crop", type=int, default=1024, help="centre crop used for calibration (multiple of 32)")
    p.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    (export_command if args.command == "export" else int8_command)(args)


if __name__ == "__main__":
    main()
