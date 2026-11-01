"""Score-improvement experiments after the first Kaggle result (feat/improvements).

Subcommands, run from the repository root:

``train``   train one configuration after computing its training-only
            normalisation statistics (needed when preprocessing changes).
``cache``   run one trained model over the validation and test images and store
            its probabilities (uint16-quantised) for fast offline combination.
``score``   average cached probabilities of one or more models (an ensemble)
            and score post-processing settings with the project's validation
            scorer (pooled PQ_dataset, IoU > 0.5), including paired bootstrap
            intervals against a reference candidate.
``export``  the deployable pipeline: read each test image, run every model,
            average, post-process, encode and write a checked Kaggle CSV with
            end-to-end timing.

The organiser's public self-evaluation notebook (version 6) computes the same
pooled PQ over annotator-image entries as ``evaluate.score_observation``.
"""
from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
QUANT = 65535.0


def _dump(value, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf8")


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #
def train_command(args) -> dict:
    from config import load_config
    from experiments import apply_training_normalization
    from train import run_training

    cfg = load_config(args.config)
    cfg, normalization = apply_training_normalization(cfg)
    print(f"normalisation: mean={cfg['data']['preprocess']['mean']:.6f} "
          f"std={cfg['data']['preprocess']['std']:.6f} ({normalization['n_images']} training images)", flush=True)
    started = time.perf_counter()
    summary = run_training(cfg)
    summary["wall_seconds"] = time.perf_counter() - started
    summary["normalization"] = normalization
    _dump(summary, Path(cfg["run"]["output_dir"]) / "improve_train_summary.json")
    print(json.dumps({k: summary[k] for k in ("best_score", "best_epoch", "training_seconds", "wall_seconds")}), flush=True)
    return summary


# --------------------------------------------------------------------------- #
# Model loading and inference
# --------------------------------------------------------------------------- #
def load_model(path, device):
    """Any project checkpoint (training, FP16 or pruned). Returns (model, config, precision)."""
    from compress import load_variant

    model, cfg, state = load_variant(path, device)
    precision = state.get("precision", "fp32")
    if precision != "fp16":
        model = model.float()
    return model, cfg, precision


def image_paths(cfg: dict, subset: str) -> list[tuple[str, Path]]:
    import data as D

    root = Path(cfg["data"]["data_root"])
    if subset == "test":
        files = sorted(p for p in (root / "test" / "test_images").iterdir()
                       if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".tif", ".tiff"))
        return [(D.obs_id_from_name(p.name), p) for p in files]
    catalog = D.load_catalog(Path(cfg["data"]["cache_dir"]))
    split = D.load_split(Path(cfg["data"]["split"]))
    first = catalog.groupby("obs")["file_name"].first()
    return [(obs, root / "train" / "train_images" / first[obs]) for obs in sorted(split["val"])]


def model_probability(model, cfg: dict, raw: np.ndarray, device, tta: bool = False) -> np.ndarray:
    """Probability map for one raw image using the model's own preprocessing."""
    import torch
    import data as D
    from inference import predict_probability

    pre = cfg["data"].get("preprocess", {})
    x = D.normalise(D.preprocess_image(raw, pre), pre)
    image = torch.from_numpy(np.ascontiguousarray(x[None, None])).to(device)
    if next(model.parameters()).dtype == torch.float16:
        image = image.half()
    inference = dict(cfg.get("inference", {}), amp=True, tta=bool(tta))
    return predict_probability(model, image, inference)[0, 0].float().cpu().numpy()


def cache_command(args) -> dict:
    import torch
    import data as D
    from config import load_config

    device = torch.device(args.device)
    model, saved_cfg, precision = load_model(args.model, device)
    cfg = deepcopy(saved_cfg)
    paths_cfg = load_config(args.paths_config)
    for key in ("data_root", "cache_dir", "split"):
        cfg["data"][key] = paths_cfg["data"][key]
    out = Path(args.output)
    report = dict(model=str(Path(args.model).resolve()), precision=precision, tta=args.tta,
                  preprocess=cfg["data"].get("preprocess"), inference=cfg.get("inference"), subsets={})
    for subset in args.subsets:
        target = out / subset
        target.mkdir(parents=True, exist_ok=True)
        seconds = []
        for i, (obs, path) in enumerate(image_paths(cfg, subset)):
            raw = D.load_image(path)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            started = time.perf_counter()
            probability = model_probability(model, cfg, raw, device, args.tta)
            seconds.append(time.perf_counter() - started)
            quantised = np.rint(np.clip(probability, 0, 1) * QUANT).astype(np.uint16)
            np.savez_compressed(target / f"{obs}.npz", q=quantised)
            if (i + 1) % 40 == 0:
                print(f"{subset}: {i + 1} images", flush=True)
        report["subsets"][subset] = dict(n_images=len(seconds),
                                         model_ms_per_image=1000 * float(np.mean(seconds[1:] or seconds)))
    _dump(report, out / "cache.json")
    print(json.dumps(report["subsets"]), flush=True)
    return report


def _convert_one(task):
    source, target = task
    with np.load(source, allow_pickle=False) as archive:
        probability = np.asarray(archive["probability"], np.float32)
    np.savez_compressed(target, q=np.rint(np.clip(probability, 0, 1) * QUANT).astype(np.uint16))
    return target


def convert_command(args) -> dict:
    """Reuse validation probabilities already saved by evaluate.py / compress.py benchmarks."""
    source = Path(args.predictions)
    target = Path(args.output) / "val"
    target.mkdir(parents=True, exist_ok=True)
    tasks = [(p, target / p.name) for p in sorted(source.glob("*.npz"))]
    with ProcessPoolExecutor(args.workers) as pool:
        list(pool.map(_convert_one, tasks))
    report = dict(source=str(source.resolve()), n_images=len(tasks), converted_from="saved validation archives")
    _dump(report, Path(args.output) / "cache.json")
    print(f"converted {len(tasks)} archives -> {target}", flush=True)
    return report


def load_cached(directory: Path, obs: str) -> np.ndarray:
    with np.load(directory / f"{obs}.npz", allow_pickle=False) as archive:
        return archive["q"].astype(np.float32) / QUANT


# --------------------------------------------------------------------------- #
# Offline scoring of ensembles and post-processing
# --------------------------------------------------------------------------- #
def expand(spec: dict) -> list[dict]:
    """Candidates = every ensemble x every post-processing grid point."""
    import itertools
    from postprocess import resolve_postprocess

    candidates = []
    for ensemble in spec["ensembles"]:
        members = ensemble["members"]
        weights = ensemble.get("weights") or [1.0] * len(members)
        for grid in spec["postprocess"]:
            keys = sorted(grid)
            values = [grid[k] if isinstance(grid[k], list) else [grid[k]] for k in keys]
            for combo in itertools.product(*values):
                settings = dict(zip(keys, combo))
                threshold = float(settings.pop("threshold", 0.5))
                pp = resolve_postprocess(settings, threshold)
                described = " ".join(f"{k}={settings[k]}" for k in sorted(settings))
                candidates.append(dict(name=f"{ensemble['name']} | t={threshold} {described}".strip(),
                                       ensemble=ensemble["name"], members=members, weights=weights,
                                       threshold=threshold, postprocess=pp))
    return candidates


_STATE: dict = {}


def _init(cache_root, cache_dir, raw, candidates, options):
    import cv2
    cv2.setNumThreads(1)
    _STATE.update(cache_root=Path(cache_root), cache_dir=Path(cache_dir), raw=raw,
                  candidates=candidates, options=options, memo={})


def _task(task):
    import data as D
    from evaluate import score_observation
    from postprocess import probability_to_instances_pp

    obs, set_ids = task
    s = _STATE
    gt = np.stack([D.load_instance_map(sid, s["cache_dir"]) for sid in set_ids])
    members = {m for c in s["candidates"] for m in c["members"]}
    probs = {m: load_cached(s["cache_root"] / m / "val", obs) for m in members}
    averaged, out = {}, []
    for c in s["candidates"]:
        key = (tuple(c["members"]), tuple(c["weights"]))
        if key not in averaged:
            w = np.asarray(c["weights"], np.float32)
            averaged[key] = sum(wi * probs[m] for wi, m in zip(w / w.sum(), c["members"]))
        pred = probability_to_instances_pp(np.clip(averaged[key], 0, 1), c["threshold"], c["postprocess"])
        scored = score_observation(pred, gt, set_ids, obs, s["raw"], s["options"])
        out.append([(sid, result) for sid, result, _ in scored])
    return obs, out


def score_command(args) -> dict:
    import yaml
    from config import load_config
    from evaluate import scoring_options
    from metrics import MetricAccumulator
    from offline_eval import observation_blocks, robustness, validation_groups

    cfg = load_config(args.config)
    spec = yaml.safe_load(Path(args.spec).read_text(encoding="utf8"))
    candidates = expand(spec)
    groups, raw, catalog, split = validation_groups(cfg)
    options = scoring_options(cfg.get("evaluation", {}))
    accs = [MetricAccumulator() for _ in candidates]
    set_counts = []
    started = time.perf_counter()
    with ProcessPoolExecutor(args.workers, initializer=_init,
                             initargs=(args.cache_root, cfg["data"]["cache_dir"], raw, candidates, options)) as pool:
        for done, (obs, results) in enumerate(pool.map(_task, groups), start=1):
            for c, sets in enumerate(results):
                for sid, scored in sets:
                    accs[c].add(scored, name=sid, observation=obs)
                    set_counts.append(dict(candidate=c, observation=obs, annotation_set=sid,
                                           tp=int(scored["tp"]), fp=int(scored["fp"]), fn=int(scored["fn"]),
                                           sum_iou=float(scored["sum_iou"])))
            if done % 30 == 0:
                print(f"  scored {done}/{len(groups)} observations ({time.perf_counter() - started:.0f}s)", flush=True)
    rows = []
    for c, acc in zip(candidates, accs):
        summary = acc.summary()
        rows.append(dict(name=c["name"], ensemble=c["ensemble"], threshold=c["threshold"],
                         **{k: summary[k] for k in ("PQ_dataset", "SQ_dataset", "RQ_dataset", "tp", "fp", "fn",
                                                    "custom_one_to_many", "custom_many_to_one")}))
    blocks = observation_blocks(catalog, split)
    robust = robustness(set_counts, candidates, blocks, n_boot=args.bootstrap)
    for row, interval in zip(rows, robust["bootstrap"]["intervals"]):
        row.update({k: interval[k] for k in ("delta_PQ", "delta_ci95_low", "delta_ci95_high",
                                             "bootstrap_probability_delta_positive")})
    for row in rows:
        row["PQ_half_A"] = robust["pq_by_half"]["A"][row["name"]]
        row["PQ_half_B"] = robust["pq_by_half"]["B"][row["name"]]
    order = sorted(range(len(rows)), key=lambda i: -rows[i]["PQ_dataset"])
    result = dict(spec=spec, reference=candidates[0]["name"], rows=rows, ranking=[rows[i]["name"] for i in order],
                  split_half=robust["split_half_selection"], elapsed_seconds=time.perf_counter() - started)
    _dump(result, Path(args.output))
    print(f"reference: {rows[0]['name']}  PQ={rows[0]['PQ_dataset']:.4f}")
    for i in order[: args.show]:
        r = rows[i]
        print(f"{r['PQ_dataset']:.4f} d={r['delta_PQ']:+.4f} [{r['delta_ci95_low']:+.4f},{r['delta_ci95_high']:+.4f}] "
              f"A={r['PQ_half_A']:.4f} B={r['PQ_half_B']:.4f} tp={r['tp']} fp={r['fp']} fn={r['fn']}  {r['name']}")
    for key, value in result["split_half"].items():
        print(f"{key}: {value['selected']} held-out gain {value['held_out_gain']:+.4f}")
    return result


# --------------------------------------------------------------------------- #
# Deployable end-to-end export
# --------------------------------------------------------------------------- #
def export_command(args) -> dict:
    import hashlib
    import torch
    import data as D
    from config import load_config
    from postprocess import probability_to_instances_pp, resolve_postprocess
    from predict import IMAGE_SHAPE, write_submission

    device = torch.device(args.device)
    paths_cfg = load_config(args.paths_config)
    models = []
    for path in args.models:
        model, cfg, precision = load_model(path, device)
        cfg = deepcopy(cfg)
        cfg["data"]["data_root"] = paths_cfg["data"]["data_root"]
        models.append((model, cfg, path))
    weights = np.asarray(args.weights or [1.0] * len(models), np.float32)
    weights = weights / weights.sum()
    postprocess = resolve_postprocess(json.loads(args.postprocess), args.threshold)
    timings = []

    def predictions():
        for i, (obs, path) in enumerate(image_paths(models[0][1], "test")):
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            started = time.perf_counter()
            raw = D.load_image(path)
            if raw.shape != IMAGE_SHAPE:
                raise ValueError(f"{path.name} is {raw.shape}, expected {IMAGE_SHAPE}")
            probability = sum(w * model_probability(m, c, raw, device, args.tta)
                              for w, (m, c, _) in zip(weights, models))
            instances = probability_to_instances_pp(np.clip(probability, 0, 1), args.threshold, postprocess)
            yield obs, instances
            timings.append(time.perf_counter() - started)
            if (i + 1) % 30 == 0:
                print(f"test: {i + 1} images", flush=True)

    started = time.perf_counter()
    report = write_submission(predictions(), args.output)
    output = Path(args.output)
    report.update(models=[str(Path(p).resolve()) for p in args.models], weights=weights.tolist(), tta=args.tta,
                  threshold=args.threshold, postprocess=postprocess,
                  end_to_end_seconds=time.perf_counter() - started,
                  end_to_end_ms_per_image=1000 * float(np.mean(timings)),
                  ms_per_image_excluding_first=1000 * float(np.mean(timings[1:])),
                  timing_scope="read + per-model preprocessing + inference + averaging + post-processing + RLE "
                               "round-trip + CSV; model loading excluded; batch size 1",
                  device=torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
                  csv_sha256=hashlib.sha256(output.read_bytes()).hexdigest(),
                  parameters=sum(sum(p.numel() for p in m.parameters()) for m, _, _ in models),
                  model_file_bytes=sum(Path(p).stat().st_size for p in args.models))
    _dump(report, output.with_suffix(".summary.json"))
    print(json.dumps({k: report[k] for k in ("n_images", "n_filaments", "zero_detection_images",
                                             "end_to_end_ms_per_image", "csv_sha256")}), flush=True)
    return report


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    t = sub.add_parser("train")
    t.add_argument("--config", required=True)
    c = sub.add_parser("cache")
    c.add_argument("--model", required=True)
    c.add_argument("--output", required=True)
    c.add_argument("--paths-config", default="configs/person_c/seed45_selected.yaml")
    c.add_argument("--subsets", nargs="+", default=["val", "test"], choices=["val", "test"])
    c.add_argument("--tta", action="store_true")
    c.add_argument("--device", default="cuda")
    v = sub.add_parser("convert")
    v.add_argument("--predictions", required=True, help="directory of saved validation probability archives")
    v.add_argument("--output", required=True)
    v.add_argument("--workers", type=int, default=6)
    s = sub.add_parser("score")
    s.add_argument("--spec", required=True, help="YAML with 'ensembles' and 'postprocess' grids")
    s.add_argument("--cache-root", default="runs/improvements/probs")
    s.add_argument("--config", default="configs/person_c/seed45_selected.yaml")
    s.add_argument("--output", required=True)
    s.add_argument("--workers", type=int, default=12)
    s.add_argument("--bootstrap", type=int, default=1000)
    s.add_argument("--show", type=int, default=25)
    e = sub.add_parser("export")
    e.add_argument("--models", nargs="+", required=True)
    e.add_argument("--weights", nargs="+", type=float)
    e.add_argument("--threshold", type=float, default=0.5)
    e.add_argument("--postprocess", default='{"bridge_radius": 8, "min_area": 320}')
    e.add_argument("--tta", action="store_true")
    e.add_argument("--paths-config", default="configs/person_c/seed45_selected.yaml")
    e.add_argument("--output", required=True)
    e.add_argument("--device", default="cuda")
    args = p.parse_args(argv)
    {"train": train_command, "cache": cache_command, "convert": convert_command, "score": score_command,
     "export": export_command}[args.command](args)


if __name__ == "__main__":
    main()
