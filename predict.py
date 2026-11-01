"""Export the baseline's test predictions as a Kaggle instance-mask CSV.

Official format: filament_id,segmentation_rle; one row per predicted filament,
COCO compressed RLE counts only, and 2048x2048 masks. Source:
https://www.kaggle.com/competitions/filament-segmentation-2026/overview/leaderboard-ranking

Zero-detection images produce no fake objects and are listed in the JSON
sidecar. The host's self-evaluation notebook counts their GT objects as false
negatives; actual upload acceptance still needs the competition account. This command
creates files locally and never uploads them or claims leaderboard acceptance.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import time
from pathlib import Path

import numpy as np
import torch

from inference import predict_probability
from evaluate import file_sha256, resolve_device
from postprocess import probability_to_instances_pp, resolve_postprocess

IMAGE_SHAPE = (2048, 2048)
_OBSERVATION_ID = re.compile(r"\d{14}[A-Za-z]{2}\Z")


def encode_binary_mask(mask: np.ndarray, expected_shape=IMAGE_SHAPE) -> str:
    """Encode a non-empty binary mask and verify an exact COCO round trip."""
    from pycocotools import mask as mu

    a = np.asarray(mask)
    if a.ndim != 2 or tuple(a.shape) != tuple(expected_shape):
        raise ValueError(f"mask shape must be {tuple(expected_shape)}, received {a.shape}")
    if not np.isfinite(a).all() or not np.isin(a, [0, 1]).all() or not a.any():
        raise ValueError("each exported mask must be binary and non-empty")
    binary = np.asfortranarray(a, dtype=np.uint8)
    encoded = mu.encode(binary)
    counts = encoded["counts"].decode("ascii")
    decoded = mu.decode({"size": list(expected_shape), "counts": counts.encode("ascii")})
    if not np.array_equal(binary, decoded):
        raise ValueError("RLE round-trip changed the mask")
    return counts


def instance_rows(observation: str, instances: np.ndarray, expected_shape=IMAGE_SHAPE):
    """Yield unique filament IDs and RLE strings for one complete image."""
    if not _OBSERVATION_ID.fullmatch(observation):
        raise ValueError(f"invalid observation ID: {observation!r}")
    inst = np.asarray(instances)
    if inst.ndim != 2 or tuple(inst.shape) != tuple(expected_shape):
        raise ValueError(f"instance map shape must be {tuple(expected_shape)}")
    if not np.issubdtype(inst.dtype, np.integer) or np.any(inst < 0):
        raise ValueError("instance maps must contain non-negative integer IDs")
    for index, instance_id in enumerate(np.unique(inst[inst > 0]), start=1):
        yield {"filament_id": f"{observation}_{index}",
               "segmentation_rle": encode_binary_mask(inst == instance_id, expected_shape)}


def write_submission(predictions, output_path, expected_shape=IMAGE_SHAPE) -> dict:
    """Write (observation, instance_map) pairs atomically; return coverage facts.

    An empty map contributes no rows and is explicitly recorded in the report.
    It must not be encoded as a fabricated filament or quietly hidden.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    seen, empty, total = set(), [], 0
    try:
        with temporary.open("w", encoding="utf8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["filament_id", "segmentation_rle"])
            writer.writeheader()
            for observation, inst in predictions:
                if observation in seen:
                    raise ValueError(f"duplicate observation in export: {observation}")
                seen.add(observation)
                count = 0
                for row in instance_rows(observation, inst, expected_shape):
                    writer.writerow(row)
                    count += 1
                total += count
                if count == 0:
                    empty.append(observation)
        if not seen:
            raise ValueError("no test images were supplied")
        temporary.replace(path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return dict(n_images=len(seen), n_filaments=total, zero_detection_images=empty,
                rle_round_trip_verified=True, kaggle_acceptance_verified=False,
                empty_image_policy="omit rows; official reference counts GT as false negatives; upload acceptance unverified")


def predict_test(model, cfg: dict, device, output_path, save_probabilities=None) -> dict:
    """Read test images, predict, convert, encode, and report actual coverage."""
    import data as D

    device = resolve_device(device)
    data_root = Path(cfg["data"].get("data_root") or D.DATA_ROOT)
    test_dir = data_root / "test" / "test_images"
    if not test_dir.is_dir():
        raise FileNotFoundError(f"missing test image directory: {test_dir}")
    files = sorted(p for p in test_dir.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".tif", ".tiff"))
    if not files:
        raise ValueError(f"no test images in {test_dir}")
    save_dir = Path(save_probabilities) if save_probabilities else None
    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)
    preprocessing = cfg["data"].get("preprocess", {})
    threshold = float(cfg.get("evaluation", {}).get("threshold", 0.5))
    postprocess = resolve_postprocess(cfg.get("postprocess"), threshold)
    image_timings = []

    def predictions():
        for i, path in enumerate(files):
            started = time.perf_counter()
            observation = D.obs_id_from_name(path.name)
            raw = D.load_image(path)
            if raw.shape != IMAGE_SHAPE:
                raise ValueError(f"test image {path.name} is {raw.shape}, expected {IMAGE_SHAPE}")
            x = D.normalise(D.preprocess_image(raw, preprocessing), preprocessing)
            image = torch.from_numpy(np.ascontiguousarray(x[None, None])).to(device)
            probability = predict_probability(model, image, cfg.get("inference", {}))[0, 0].cpu().numpy()
            instances = probability_to_instances_pp(probability, threshold, postprocess)
            if save_dir:
                np.savez_compressed(save_dir / f"{observation}.npz", probability=probability,
                                    instances=instances.astype(np.uint32), observation=observation)
            yield observation, instances  # encoding and CSV writing happen before resuming
            image_timings.append(dict(observation=observation, seconds=time.perf_counter() - started,
                                      n_instances=int(instances.max())))
            if (i + 1) % 10 == 0 or i + 1 == len(files):
                print(f"test prediction: {i + 1}/{len(files)} images", flush=True)

    started = time.perf_counter()
    report = write_submission(predictions(), output_path)
    report.update(
        data_root=str(data_root), foreground_threshold=threshold, inference=cfg.get("inference", {}),
        postprocess=postprocess, config=cfg,
        end_to_end_seconds=time.perf_counter() - started,
        end_to_end_ms_per_image=1000 * sum(r["seconds"] for r in image_timings) / len(image_timings),
        timing_scope="read + preprocess + transfer + model + components + RLE round-trip + CSV; optional probability export included; no warm-up excluded",
        images=image_timings,
    )
    sidecar = Path(output_path).with_suffix(".summary.json")
    with sidecar.open("w", encoding="utf8") as f:
        json.dump(report, f, indent=2, allow_nan=False)
    return report


def main(argv=None) -> None:
    from config import load_config
    from models import load_checkpoint_model

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output", required=True, help="submission CSV path")
    p.add_argument("--config", help="optional complete config override")
    p.add_argument("--data-root")
    p.add_argument("--save-probabilities", metavar="DIR")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
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
    report = predict_test(model, cfg, device, args.output, args.save_probabilities)
    report["checkpoint"] = str(Path(args.checkpoint).resolve())
    report["checkpoint_sha256"] = file_sha256(args.checkpoint)
    with Path(args.output).with_suffix(".summary.json").open("w", encoding="utf8") as f:
        json.dump(report, f, indent=2, allow_nan=False)
    print(json.dumps({k: v for k, v in report.items() if k != "images"}, indent=2))


if __name__ == "__main__":
    main()
