"""Score saved validation probabilities with post-processing settings (Person C).

No network inference happens here. Each ``{observation}.npz`` written by
``evaluate.py --save-predictions`` supplies the model's full-resolution
probabilities. They are converted by ``postprocess.probability_to_instances_pp``
and scored by ``evaluate.score_observation``: the same policy as checkpoint
evaluation (pooled ``PQ_dataset``, strict IoU > 0.5 matching, and original
binary layers for overlapping/empty annotation sets).

Commands, run from the repository root:

``score``  one setting (the config's ``evaluation.threshold`` and ``postprocess``)
           with the same detailed tables as ``evaluate.py``; ``--expect`` checks
           the result against a recorded summary.
``sweep``  a declared list of settings (YAML); one result row per setting,
           per-annotation-set counts, a time-block split-half selection check
           and paired observation-bootstrap intervals against the reference.

Timings here cover post-processing only. They exclude image reading and network
inference and must not be reported as end-to-end latency.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from evaluate import (_raw_annotations_for_loader, _write_rows, detail_rows, file_sha256,
                      score_observation, scoring_options, write_details)
from metrics import MetricAccumulator
from postprocess import DEFAULT_POSTPROCESS, describe, probability_to_instances_pp, resolve_postprocess

ROOT = Path(__file__).resolve().parent
SOURCE_FILES = ("postprocess.py", "offline_eval.py", "evaluate.py", "metrics.py", "data.py", "config.py")
CHECK_KEYS = ("PQ_dataset", "SQ_dataset", "RQ_dataset", "tp", "fp", "fn", "sum_iou",
              "one_to_many", "many_to_one", "any_overlap_one_to_many", "any_overlap_many_to_one",
              "n_annotation_sets", "n_observations", "observation_PQ_mean", "pixel_dice_mean",
              "filament_dice_mean", "filament_iou_mean")
RESULT_KEYS = ("PQ_dataset", "SQ_dataset", "RQ_dataset", "tp", "fp", "fn", "sum_iou",
               "custom_one_to_many", "custom_many_to_one", "any_overlap_one_to_many",
               "any_overlap_many_to_one", "observation_PQ_mean", "pixel_dice_mean",
               "filament_dice_mean", "filament_iou_mean")


# --------------------------------------------------------------------------- #
# Candidates
# --------------------------------------------------------------------------- #
def make_candidate(threshold=0.5, postprocess=None, name=None, family="custom") -> dict:
    pp = resolve_postprocess(postprocess, threshold)
    return dict(name=name or describe(threshold, pp), family=family,
                threshold=float(threshold), postprocess=pp)


def _key(candidate: dict) -> str:
    return json.dumps([candidate["threshold"], candidate["postprocess"]], sort_keys=True)


def expand_candidates(entries: list) -> list[dict]:
    """Expand list-valued settings into named candidates; the baseline comes first.

    Each entry may contain ``family``, ``name``, ``threshold`` and postprocess
    keys. A list value means "try each value"; several lists form a product.
    Identical settings are evaluated once, under their first name.
    """
    candidates = [make_candidate(name="baseline", family="reference")]
    seen = {_key(candidates[0])}
    for entry in entries:
        entry = dict(entry)
        family, name = entry.pop("family", "custom"), entry.pop("name", None)
        keys = sorted(entry)
        grids = [entry[k] if isinstance(entry[k], list) else [entry[k]] for k in keys]
        combos = list(itertools.product(*grids))
        for combo in combos:
            settings = dict(zip(keys, combo))
            threshold = settings.pop("threshold", 0.5)
            candidate = make_candidate(threshold, settings, family=family)
            if name:
                candidate["name"] = name if len(combos) == 1 else f"{name}: {candidate['name']}"
            if _key(candidate) not in seen:
                seen.add(_key(candidate))
                candidates.append(candidate)
    names = [c["name"] for c in candidates]
    if len(set(names)) != len(names):
        raise ValueError("candidate names must be unique")
    return candidates


# --------------------------------------------------------------------------- #
# Validation data and workers
# --------------------------------------------------------------------------- #
def validation_groups(cfg: dict):
    """Frozen-split validation observations, their set IDs and lossless raw layers."""
    import data as D

    cache_dir, data_root = Path(cfg["data"]["cache_dir"]), Path(cfg["data"]["data_root"])
    catalog = D.load_catalog(cache_dir)
    split = D.load_split(Path(cfg["data"]["split"]))
    D.validate_data_inputs(catalog, split, data_root / "train" / "train_images", cache_dir)
    dataset = D.ObservationDataset(catalog, split["val"], data_root / "train" / "train_images",
                                   cache_dir=cache_dir)
    raw = _raw_annotations_for_loader(SimpleNamespace(dataset=dataset))
    groups = [(obs, [str(s) for s in g["set_id"]]) for obs, g in dataset.groups]
    return groups, raw, catalog, split


_STATE: dict = {}


def _init_worker(cache_dir, raw, candidates, options, check_stored) -> None:
    import cv2
    cv2.setNumThreads(1)
    _STATE.update(cache_dir=Path(cache_dir), raw=raw, candidates=candidates,
                  options=options, check_stored=check_stored)


def _score_task(task):
    """Load one archive and its annotation sets once, then score every candidate."""
    import data as D

    obs, set_ids, path = task
    s = _STATE
    with np.load(path, allow_pickle=False) as archive:
        if str(archive["observation"].item()) != obs:
            raise ValueError(f"{path} does not contain observation {obs}")
        probability = np.asarray(archive["probability"])
        stored = np.asarray(archive["instances"]) if s["check_stored"] else None
    gt_sets = np.stack([D.load_instance_map(sid, s["cache_dir"]) for sid in set_ids])
    if probability.shape != gt_sets.shape[1:]:
        raise ValueError(f"{obs}: probability {probability.shape} does not match annotations {gt_sets.shape[1:]}")
    results, stored_match = [], None
    for index, candidate in enumerate(s["candidates"]):
        started = time.perf_counter()
        pred = probability_to_instances_pp(probability, candidate["threshold"], candidate["postprocess"])
        seconds = time.perf_counter() - started
        if stored is not None and index == 0:
            stored_match = bool(np.array_equal(pred, stored))
        scored = score_observation(pred, gt_sets, set_ids, obs, s["raw"], s["options"])
        results.append(dict(seconds=seconds, n_pred=int(pred.max()),
                            sets=[(sid, result) for sid, result, _ in scored],
                            n_layers=sum(int(used) for *_, used in scored)))
    return obs, results, stored_match


def run_candidates(cfg: dict, predictions_dir, candidates: list[dict], workers: int = 1,
                   details: bool = False, check_stored: bool = False) -> dict:
    """Score candidates over every validation archive; returns accumulators and rows."""
    groups, raw, catalog, split = validation_groups(cfg)
    archives = {p.stem: p for p in Path(predictions_dir).glob("*.npz")}
    expected = {obs for obs, _ in groups}
    if set(archives) != expected:
        raise ValueError(f"archives differ from validation observations: missing "
                         f"{sorted(expected - set(archives))[:3]}, unexpected {sorted(set(archives) - expected)[:3]}")
    options = scoring_options(cfg.get("evaluation", {}))
    tasks = [(obs, set_ids, str(archives[obs])) for obs, set_ids in groups]
    accs = [MetricAccumulator() for _ in candidates]
    rows = [([], []) for _ in candidates]
    set_counts, seconds, n_pred = [], np.zeros(len(candidates)), np.zeros(len(candidates), np.int64)
    stored_matches, n_layers = [], 0
    initargs = (str(Path(cfg["data"]["cache_dir"])), raw, candidates, options, check_stored)

    def collect(obs, results, stored_match):
        nonlocal n_layers
        if stored_match is not None:
            stored_matches.append(stored_match)
        n_layers += results[0]["n_layers"]
        for c, result in enumerate(results):
            seconds[c] += result["seconds"]
            n_pred[c] += result["n_pred"]
            for sid, scored in result["sets"]:
                accs[c].add(scored, name=sid, observation=obs)
                set_counts.append(dict(candidate=c, observation=obs, annotation_set=sid,
                                       tp=int(scored["tp"]), fp=int(scored["fp"]), fn=int(scored["fn"]),
                                       sum_iou=float(scored["sum_iou"]), n_gt=int(scored["n_gt"]),
                                       n_pred=int(scored["n_pred"])))
                if details:
                    filaments, overlaps = detail_rows(obs, sid, scored)
                    rows[c][0].extend(filaments)
                    rows[c][1].extend(overlaps)

    started = time.perf_counter()
    if workers > 1:
        with ProcessPoolExecutor(workers, initializer=_init_worker, initargs=initargs) as pool:
            for done, output in enumerate(pool.map(_score_task, tasks), start=1):
                collect(*output)
                if done % 20 == 0:
                    print(f"  scored {done}/{len(tasks)} observations", flush=True)
    else:
        _init_worker(*initargs)
        for done, task in enumerate(tasks, start=1):
            collect(*_score_task(task))
            if done % 20 == 0:
                print(f"  scored {done}/{len(tasks)} observations", flush=True)
    return dict(accumulators=accs, detail_rows=rows, set_counts=set_counts,
                postprocess_seconds=seconds, n_pred_instances=n_pred, n_raw_annotation_sets=n_layers,
                stored_instances_match=(all(stored_matches) if stored_matches else None),
                n_stored_checked=len(stored_matches), elapsed_seconds=time.perf_counter() - started,
                groups=groups, catalog=catalog, split=split)


# --------------------------------------------------------------------------- #
# Robustness of a selection made on the same validation split
# --------------------------------------------------------------------------- #
def observation_blocks(catalog, split, block_days: float = 14.0) -> dict:
    """Recompute each validation observation's 14-day block exactly as the split did."""
    times = catalog.groupby("obs")["time"].first()
    t0 = times.min()
    blocks = ((times - t0).dt.total_seconds() // (block_days * 86400)).astype(int)
    val = {o: int(blocks[o]) for o in split["val"]}
    recorded = split.get("val_blocks")
    if recorded is not None and set(val.values()) != set(int(b) for b in recorded):
        raise ValueError("recomputed validation blocks differ from the frozen split")
    return val


def robustness(set_counts: list[dict], candidates: list[dict], blocks: dict,
               n_boot: int = 2000, seed: int = 0) -> dict:
    """Split-half selection check and paired cluster-bootstrap PQ differences.

    Validation blocks are ordered in time and alternately assigned to halves A
    and B. The best setting chosen on one half is scored on the other half.
    The bootstrap resamples whole observations (all of their annotation sets),
    keeping every candidate paired with the reference on the same resample.
    """
    observations = sorted(blocks)
    index = {o: i for i, o in enumerate(observations)}
    counts = np.zeros((len(candidates), len(observations), 4))
    for row in set_counts:
        counts[row["candidate"], index[row["observation"]]] += (row["sum_iou"], row["tp"], row["fp"], row["fn"])

    def pq(c):
        denominator = c[..., 1] + 0.5 * c[..., 2] + 0.5 * c[..., 3]
        return np.where(denominator > 0, c[..., 0] / np.maximum(denominator, 1e-12), 0.0)

    ordered_blocks = sorted(set(blocks.values()))
    half = {b: "A" if i % 2 == 0 else "B" for i, b in enumerate(ordered_blocks)}
    masks = {h: np.array([half[blocks[o]] == h for o in observations]) for h in "AB"}
    by_half = {h: pq(counts[:, masks[h]].sum(1)) for h in "AB"}
    selection = {}
    for chosen_on, held_out in (("A", "B"), ("B", "A")):
        best = int(np.argmax(by_half[chosen_on]))
        selection[f"select_on_{chosen_on}"] = dict(
            selected=candidates[best]["name"],
            selected_pq_on_selection_half=float(by_half[chosen_on][best]),
            held_out_half=held_out,
            selected_pq_on_held_out_half=float(by_half[held_out][best]),
            baseline_pq_on_held_out_half=float(by_half[held_out][0]),
            held_out_gain=float(by_half[held_out][best] - by_half[held_out][0]),
            n_held_out_observations=int(masks[held_out].sum()))
    rng = np.random.default_rng(seed)
    weights = rng.multinomial(len(observations), np.full(len(observations), 1 / len(observations)), size=n_boot)
    boot = np.einsum("bo,cok->cbk", weights, counts)
    boot_pq = pq(boot)
    delta = boot_pq - boot_pq[0:1]
    full = pq(counts.sum(1))
    intervals = [dict(name=c["name"], PQ=float(full[i]), delta_PQ=float(full[i] - full[0]),
                      delta_ci95_low=float(np.percentile(delta[i], 2.5)),
                      delta_ci95_high=float(np.percentile(delta[i], 97.5)),
                      bootstrap_probability_delta_positive=float((delta[i] > 0).mean()))
                 for i, c in enumerate(candidates)]
    return dict(halves={h: dict(blocks=[b for b in ordered_blocks if half[b] == h],
                                n_observations=int(masks[h].sum())) for h in "AB"},
                pq_by_half={h: {c["name"]: float(by_half[h][i]) for i, c in enumerate(candidates)} for h in "AB"},
                split_half_selection=selection, bootstrap=dict(n_resamples=n_boot, seed=seed,
                                                               unit="validation observation",
                                                               intervals=intervals))


# --------------------------------------------------------------------------- #
# Provenance and outputs
# --------------------------------------------------------------------------- #
def _sha256_bytes(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def provenance(cfg: dict, predictions_dir, config_path) -> dict:
    predictions_dir = Path(predictions_dir).resolve()
    source = predictions_dir.parent / "summary.json"
    origin = {}
    if source.is_file():
        recorded = json.loads(source.read_text(encoding="utf8"))
        origin = {k: recorded.get(k) for k in ("checkpoint", "checkpoint_sha256", "PQ_dataset", "tp", "fp", "fn")}
    return dict(predictions_dir=str(predictions_dir), n_archives=len(list(predictions_dir.glob("*.npz"))),
                predictions_origin=origin, config=str(Path(config_path).resolve()) if config_path else None,
                split=cfg["data"]["split"], split_sha256=_sha256_bytes(Path(cfg["data"]["split"])),
                source_sha256={name: _sha256_bytes(ROOT / name) for name in SOURCE_FILES if (ROOT / name).is_file()},
                timing_scope="post-processing only (probability -> instance map); excludes image I/O, "
                             "network inference and scoring; not end-to-end latency")


def compare_summary(summary: dict, expected_path, tolerance: float = 1e-12) -> dict:
    expected = json.loads(Path(expected_path).read_text(encoding="utf8"))
    differences = {}
    for key in CHECK_KEYS:
        if key not in expected:
            continue
        a, b = summary.get(key), expected[key]
        if a is None or abs(float(a) - float(b)) > tolerance:
            differences[key] = dict(offline=a, expected=b)
    return dict(expected_summary=str(Path(expected_path).resolve()), tolerance=tolerance,
                checked_keys=[k for k in CHECK_KEYS if k in expected], differences=differences,
                matches=not differences)


def _prepare_output(path, overwrite: bool) -> Path:
    out = Path(path)
    if out.exists() and any(out.iterdir()) and not overwrite:
        raise FileExistsError(f"{out} is not empty; choose a new output directory (or --overwrite)")
    out.mkdir(parents=True, exist_ok=True)
    return out


def _dump(value, path: Path) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf8")


def score_command(args) -> dict:
    from config import load_config, save_config

    cfg = load_config(args.config)
    if args.data_root:
        cfg["data"]["data_root"] = str(Path(args.data_root).resolve())
    threshold = float(cfg["evaluation"]["threshold"])
    candidate = make_candidate(threshold, cfg.get("postprocess"), name=args.name or None)
    out = _prepare_output(args.output, args.overwrite)
    run = run_candidates(cfg, args.predictions, [candidate], args.workers, details=True,
                         check_stored=args.check_stored)
    acc = run["accumulators"][0]
    summary = acc.summary()
    summary.update(evaluation_policy="pooled_counts_over_annotation_sets; official_self_evaluation_v6",
                   primary_metric="PQ_dataset", n_raw_annotation_sets=run["n_raw_annotation_sets"],
                   ground_truth_policy="cached disjoint maps; original binary layers for overlapping/empty annotations",
                   foreground_threshold=threshold, iou_matching_threshold=scoring_options(cfg["evaluation"])["iou_thr"],
                   postprocess=candidate["postprocess"], setting=candidate["name"],
                   n_predicted_instances=int(run["n_pred_instances"][0]),
                   postprocess_ms_per_image=1000 * float(run["postprocess_seconds"][0]) / summary["n_observations"],
                   offline_elapsed_seconds=run["elapsed_seconds"], workers=args.workers,
                   stored_instances_match=run["stored_instances_match"],
                   n_stored_instances_checked=run["n_stored_checked"],
                   provenance=provenance(cfg, args.predictions, args.config))
    if args.expect:
        summary["reference_check"] = compare_summary(summary, args.expect)
    write_details(out, acc, *run["detail_rows"][0])
    save_config(cfg, out / "config.yaml")
    _dump(summary, out / "summary.json")
    print(json.dumps({k: summary[k] for k in ("setting", "PQ_dataset", "SQ_dataset", "RQ_dataset", "tp", "fp", "fn",
                                               "custom_one_to_many", "custom_many_to_one")}, indent=2))
    if args.expect:
        print("reference check:", "MATCH" if summary["reference_check"]["matches"]
              else summary["reference_check"]["differences"])
    if args.check_stored:
        print("stored baseline instance maps identical:", run["stored_instances_match"])
    return summary


def sweep_command(args) -> dict:
    import yaml
    from config import load_config, save_config

    cfg = load_config(args.config)
    if args.data_root:
        cfg["data"]["data_root"] = str(Path(args.data_root).resolve())
    spec = yaml.safe_load(Path(args.grid).read_text(encoding="utf8"))
    candidates = expand_candidates(spec["candidates"])
    out = _prepare_output(args.output, args.overwrite)
    _dump(dict(grid=str(Path(args.grid).resolve()), description=spec.get("description"),
               candidates=candidates), out / "candidates.json")
    print(f"scoring {len(candidates)} settings on saved probabilities with {args.workers} workers", flush=True)
    run = run_candidates(cfg, args.predictions, candidates, args.workers, check_stored=True)
    rows = []
    for i, (candidate, acc) in enumerate(zip(candidates, run["accumulators"])):
        s = acc.summary()
        rows.append(dict(name=candidate["name"], family=candidate["family"], threshold=candidate["threshold"],
                         **{k: s[k] for k in RESULT_KEYS},
                         n_predicted_instances=int(run["n_pred_instances"][i]),
                         postprocess_ms_per_image=1000 * float(run["postprocess_seconds"][i]) / s["n_observations"],
                         settings=json.dumps({k: v for k, v in candidate["postprocess"].items()
                                              if v != DEFAULT_POSTPROCESS[k]}, sort_keys=True)))
    for row in rows:
        for key in ("PQ_dataset", "tp", "fp", "fn", "custom_one_to_many", "custom_many_to_one"):
            row[f"delta_{key}"] = row[key] - rows[0][key]
    _write_rows(out / "results.csv", rows)
    for row in run["set_counts"]:
        row["name"] = candidates[row["candidate"]]["name"]
    _write_rows(out / "set_counts.csv", run["set_counts"])
    blocks = observation_blocks(run["catalog"], run["split"])
    analysis = robustness(run["set_counts"], candidates, blocks, n_boot=args.bootstrap)
    best = max(range(len(rows)), key=lambda i: rows[i]["PQ_dataset"])
    summary = dict(n_candidates=len(candidates), reference=rows[0], best=rows[best],
                   stored_baseline_instances_match=run["stored_instances_match"],
                   n_stored_instances_checked=run["n_stored_checked"],
                   offline_elapsed_seconds=run["elapsed_seconds"], workers=args.workers,
                   analysis=analysis, provenance=provenance(cfg, args.predictions, args.config))
    if args.expect:
        summary["reference_check"] = compare_summary(run["accumulators"][0].summary(), args.expect)
    save_config(cfg, out / "config.yaml")
    _dump(summary, out / "summary.json")
    ranked = sorted(rows, key=lambda r: -r["PQ_dataset"])
    for row in ranked[: args.show]:
        print(f"{row['PQ_dataset']:.5f} ({row['delta_PQ_dataset']:+.5f}) TP {row['tp']} FP {row['fp']} "
              f"FN {row['fn']}  {row['name']}")
    if args.expect:
        print("reference check:", "MATCH" if summary["reference_check"]["matches"]
              else summary["reference_check"]["differences"])
    return summary


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("score", "sweep"):
        p = sub.add_parser(name)
        p.add_argument("--predictions", required=True, help="directory of {observation}.npz validation archives")
        p.add_argument("--config", required=True, help="config supplying data paths and evaluation settings")
        p.add_argument("--output", required=True)
        p.add_argument("--data-root")
        p.add_argument("--workers", type=int, default=max(1, min(8, (os.cpu_count() or 2) - 2)))
        p.add_argument("--expect", help="recorded summary.json that the reference setting must reproduce")
        p.add_argument("--overwrite", action="store_true")
        if name == "score":
            p.add_argument("--name", default="")
            p.add_argument("--check-stored", action="store_true",
                           help="compare recomputed instance maps with the archived baseline maps")
        else:
            p.add_argument("--grid", required=True, help="YAML list of declared candidate settings")
            p.add_argument("--bootstrap", type=int, default=2000)
            p.add_argument("--show", type=int, default=15)
    args = parser.parse_args(argv)
    (score_command if args.command == "score" else sweep_command)(args)


if __name__ == "__main__":
    main()
