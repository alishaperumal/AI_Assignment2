"""Staged-study integration on real evaluator artifacts from tiny synthetic images."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image
import pytest
import torch
import yaml

from evaluate import evaluate_model
import study_report as S


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf8")


@pytest.fixture
def study(tmp_path):
    data_root, cache = tmp_path / "data", tmp_path / "cache"
    for directory in (data_root / "train/train_images", data_root / "test/test_images", cache / "instances"):
        directory.mkdir(parents=True)
    names = ["20150101000000Mh", "20150102000000Mh", "20150103000000Mh"]
    test_names = ["20150104000000Bh", "20150105000000Cy"]
    gt = np.zeros((32, 32), np.int32)
    gt[10:20, 10:20] = 1
    catalog = []
    for name in names:
        sid = f"annotator-{name}"
        catalog.append(dict(obs=name, set_id=sid, set_code="annotator", time="2015-01-01", file_name=f"{name}.png"))
        cv2.imwrite(str(data_root / "train/train_images" / f"{name}.png"), np.full((32, 32), 150, np.uint8))
        cv2.imwrite(str(cache / "instances" / f"{sid}.png"), gt.astype(np.uint16))
    pd.DataFrame(catalog).to_csv(cache / "catalog.csv", index=False)
    for name in test_names:
        cv2.imwrite(str(data_root / "test/test_images" / f"{name}.png"), np.full((32, 32), 150, np.uint8))

    def make_run(directory, variant, seed, width, epochs):
        directory.mkdir(parents=True)
        cfg = dict(run={"name": directory.name}, seed=seed,
                   train={"epochs": epochs}, loss={"name": "bce_dice"},
                   data={"seed": seed, "data_root": str(data_root), "cache_dir": str(cache), "preprocess": {}},
                   inference={"amp": False}, evaluation={"threshold": .5, "iou_threshold": .5})
        (directory / "config.yaml").write_text(yaml.safe_dump(cfg), encoding="utf8")
        prediction = np.zeros_like(gt)
        prediction[10:20, 10:10 + width] = 1
        image = torch.from_numpy(np.where(prediction, 12., -12.).astype(np.float32))[None, None]
        batches = [dict(image=image, gt_sets=torch.from_numpy(gt)[None, None], name=[name]) for name in names]
        metrics = evaluate_model(torch.nn.Identity(), batches, "cpu", cfg, directory / "evaluation", True)
        weights = directory / "best_weights.pt"
        weights.write_bytes(f"synthetic weights {variant} {seed}".encode())
        sha = hashlib.sha256(weights.read_bytes()).hexdigest()
        metrics.update(checkpoint=str(weights), checkpoint_sha256=sha)
        write_json(directory / "evaluation/summary.json", metrics)
        (directory / "evaluation/config.yaml").write_text(yaml.safe_dump(cfg), encoding="utf8")
        summary = dict(run_name=directory.name, seed=seed, epochs_completed=epochs, best_epoch=epochs,
                       stop_reason="completed", best_validation=metrics, best_score=metrics["PQ_dataset"],
                       split_sha256="fixture_split", parameters=10, weights_file_bytes=weights.stat().st_size,
                       weights_checkpoint=str(weights), training_seconds=2, total_seconds=3, optimizer_steps=epochs)
        write_json(directory / "summary.json", summary)
        write_json(directory / "run_metadata.json", {"source_sha256": {"metrics.py": "same_source"}})
        pd.DataFrame([dict(epoch=1, train_loss=1, val_PQ_dataset=metrics["PQ_dataset"])]).to_csv(directory / "history.csv", index=False)
        return dict(variant=variant, seed=seed, run_dir=str(directory))

    records = tmp_path / "screen/records"
    for variant, width in [("baseline", 6), ("tversky", 8)]:
        run = make_run(tmp_path / "screen" / f"{variant}_seed42", variant, 42, width, 2)
        cfg = S._config(Path(run["run_dir"]) / "config.yaml")
        digest = hashlib.sha256(json.dumps(cfg, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        write_json(records / f"{variant}_seed42.json", dict(experiment=variant, status="completed",
            config_sha256=digest, invocation_seconds=4.2, **run))
    confirmation = [make_run(tmp_path / "full" / f"{v}_seed42", v, 42, w, 4)
                    for v, w in [("baseline", 6), ("tversky", 8)]]
    fresh = [make_run(tmp_path / "final" / f"tversky_seed{seed}", "tversky", seed, width, 4)
             for seed, width in [(43, 7), (44, 8), (45, 9)]]
    representative = Path(fresh[0]["run_dir"])
    test_predictions = representative / "test_predictions"
    test_predictions.mkdir()
    for name in test_names:
        np.savez_compressed(test_predictions / f"{name}.npz", probability=gt.astype(np.float32),
                            instances=gt.astype(np.uint32), observation=name)
    write_json(representative / "submission.summary.json", dict(
        checkpoint_sha256=hashlib.sha256((representative / "best_weights.pt").read_bytes()).hexdigest(),
        config=S._config(representative / "config.yaml"), n_images=2, end_to_end_seconds=2.4,
        end_to_end_ms_per_image=1200, timing_scope="synthetic end-to-end export timing",
        images=[dict(observation=n, n_instances=1, seconds=1.2) for n in test_names]))
    protocol = dict(schema_version=1, primary_metric="PQ_dataset", split_sha256="fixture_split",
        screen=dict(records_dir=str(records), epochs=2, seed=42, expected_variants=["baseline", "tversky"]),
        confirmation=dict(epochs=4, seed=42), final=dict(epochs=4, seeds=[43, 44, 45]))
    selection = dict(selected_variant="tversky", selection_seed=42, selection_rule="Synthetic fixture selection",
        confirmation_runs=confirmation, fresh_seed_runs=fresh, baseline_run=confirmation[0]["run_dir"],
        representative_run=fresh[0]["run_dir"])
    pp, sp = tmp_path / "protocol.json", tmp_path / "selection.json"
    write_json(pp, protocol); write_json(sp, selection)
    return dict(protocol=pp, selection=sp, output=tmp_path / "study", declaration=selection)


def test_study_separates_selection_from_fresh_seeds_and_renders_evidence(study):
    result = S.build_study(study["protocol"], study["selection"], study["output"], require_complete=True)
    score = result["fresh_seed_summary"]["PQ_dataset"]
    assert score["mean"] == pytest.approx(.8)
    assert score["std_across_runs"] == pytest.approx(.1)  # ddof=1, not population std
    assert result["fresh_seed_summary"]["tp"]["mean"] == 3
    assert result["fresh_seed_summary"]["fp"]["mean"] == 0
    assert result["fresh_seed_summary"]["fn"]["mean"] == 0
    assert result["fresh_seed_summary"]["sum_iou"]["mean"] == pytest.approx(2.4)
    assert result["fresh_seed_summary"]["sum_iou"]["std_across_runs"] == pytest.approx(.3)
    assert result["fresh_seed_summary"]["seeds"] == [43, 44, 45]
    assert result["kaggle"]["score"] is None
    assert result["completeness"]["all_training_complete"]
    assert result["representative"]["end_to_end_test"]["ms_per_image"] == 1200
    assert all(row["subprocess_wall_seconds"] == 4.2 for row in result["screening"])
    assert all(row["unlabeled"] for row in result["examples"]["test"])
    assert len(result["examples"]["validation"]) == 3
    paths = list(result["figures"].values())
    paths += [row["panel"] for rows in result["examples"].values() for row in rows]
    paths += [row["zoom_panel"] for row in result["examples"]["validation"]]
    assert all(paths)
    for path in paths:
        with Image.open(path) as image:
            assert image.width > 500
            image.verify()
    saved = S.read_json(study["output"] / "summary.json")
    assert saved["fresh_seed_summary"]["PQ_dataset"] == score


def test_partial_seeds_are_excluded_and_selection_seed_cannot_enter_estimate(study):
    selection = study["declaration"]
    rows = [S.load_run(r, 4, "fixture_split") for r in selection["fresh_seed_runs"]]
    rows[-1]["status"] = "partial"
    result = S.fresh_seed_statistics(rows, [43, 44, 45])
    assert result["n_runs"] == 2 and result["PQ_dataset"]["mean"] == pytest.approx(.75)
    singleton = S.fresh_seed_statistics(rows[:1], [43, 44, 45])
    assert singleton["PQ_dataset"]["std_across_runs"] is None
    invalid = deepcopy(rows)
    invalid[0]["seed"] = 42
    with pytest.raises(ValueError, match="selection seeds"):
        S.fresh_seed_statistics(invalid, [43, 44, 45])


def test_report_rejects_changed_inference_settings_and_incomplete_final_study(study):
    run = study["declaration"]["fresh_seed_runs"][0]
    path = Path(run["run_dir"]) / "evaluation/config.yaml"
    original = path.read_text()
    cfg = yaml.safe_load(original)
    cfg["evaluation"]["threshold"] = .6
    path.write_text(yaml.safe_dump(cfg))
    with pytest.raises(ValueError, match="settings differ"):
        S.load_run(run, 4, "fixture_split")
    path.write_text(original)
    last = Path(study["declaration"]["fresh_seed_runs"][-1]["run_dir"]) / "summary.json"
    summary = S.read_json(last)
    summary.update(epochs_completed=2, stop_reason="invocation_limit")
    write_json(last, summary)
    with pytest.raises(ValueError, match="pending/partial"):
        S.build_study(study["protocol"], study["selection"], study["output"], examples=False, require_complete=True)


def test_amp_skips_are_counted_separately_without_marking_training_failed(tmp_path):
    cfg = dict(data={"epoch_length": 960, "batch_size": 4, "drop_last": True},
               train={"epochs": 20, "accumulation_steps": 1})
    pd.DataFrame([dict(optimizer_steps=239 if epoch in (3, 9, 12) else 240,
                       skipped_optimizer_steps=1 if epoch in (3, 9, 12) else 0)
                  for epoch in range(1, 21)]).to_csv(tmp_path / "history.csv", index=False)
    result = S._update_accounting(tmp_path, cfg, 20, {"optimizer_steps": 4797})
    assert result["planned_update_attempts"] == result["recorded_update_attempts"] == 4800
    assert result["optimizer_steps"] == 4797 and result["amp_skipped_steps"] == 3
    assert result["update_accounting"] == "recorded"
    with pytest.raises(ValueError, match="disagrees"):
        S._update_accounting(tmp_path, cfg, 20, {"optimizer_steps": 4800})
    pd.DataFrame([dict(epoch=1)]).to_csv(tmp_path / "history.csv", index=False)
    missing = S._update_accounting(tmp_path, cfg, 20, {})
    assert missing["amp_skipped_steps"] is None  # missing evidence is not zero skips


def test_validation_selection_includes_actual_score_extremes_and_deltas():
    table = pd.DataFrame(dict(observation=list("abcdef"),
                              PQ_baseline=[.9, .1, .2, .9, .5, .5],
                              PQ_final=[.95, .1, .8, .3, .5, .55]))
    selected = S.select_validation_examples(table)
    assert [r["observation"] for r in selected] == list("abcde")
    assert "Highest" in selected[0]["reason"] and "Lowest" in selected[1]["reason"]
    assert selected[2]["delta"] == pytest.approx(.6)
    assert selected[3]["delta"] == pytest.approx(-.6)
    assert selected == S.select_validation_examples(table.iloc[::-1])


@pytest.mark.parametrize("baseline,final", [([.9, .2, .5], [.95, .1, .5]),
                                           ([.5, .5, .5], [.6, .6, .6])])
def test_validation_selection_deduplicates_but_preserves_all_reasons(baseline, final):
    selected = S.select_validation_examples(pd.DataFrame(dict(observation=list("abc"),
                                                              PQ_baseline=baseline, PQ_final=final)))
    assert len({r["observation"] for r in selected}) == len(selected) == 3
    assert sum(len(r["reasons"]) for r in selected) == 5
    assert max(r["final_PQ"] for r in selected) == max(final)
    assert min(r["final_PQ"] for r in selected) == min(final)


def test_experiment_timing_requires_matching_run_seed_and_config(tmp_path):
    directory = tmp_path / "run"
    cfg = {"seed": 43, "train": {"epochs": 20}}
    digest = hashlib.sha256(json.dumps(cfg, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    record = dict(run_dir=str(directory), seed=43, config_sha256=digest,
                  invocation_seconds=123.4, status="completed", started_utc="first invocation",
                  last_started_utc="resumed invocation", finished_utc="finished")
    path = tmp_path / "record.json"
    write_json(path, record)
    timing = S._experiment_timing(path, directory, 43, cfg)
    assert timing["subprocess_wall_seconds"] == 123.4
    assert timing["subprocess_wall_started_utc"] == "resumed invocation"
    assert "earlier invocations" in timing["subprocess_wall_scope"]
    for field, value in [("seed", 42), ("run_dir", str(tmp_path / "other")), ("config_sha256", "stale")]:
        write_json(path, {**record, field: value})
        with pytest.raises(ValueError, match="provenance|configuration hash"):
            S._experiment_timing(path, directory, 43, cfg)


def test_test_timing_uses_complete_operation_instead_of_generator_timer_mean(tmp_path):
    sidecar = dict(end_to_end_seconds=4.0, n_images=2, end_to_end_ms_per_image=1500.0,
                   timing_scope="fixture including CSV close")
    result = S.summarize_test_timing(sidecar, tmp_path / "submission.summary.json")
    assert result["ms_per_image"] == 2000.0
    assert result["exporter_per_image_mean_ms"] == 1500.0
    assert "complete CSV-operation" in result["ms_per_image_derivation"]
    missing = S.summarize_test_timing({**sidecar, "n_images": 0}, tmp_path / "invalid.json")
    assert missing["ms_per_image"] is None  # never substitute the narrower timer average


def test_figure_gallery_uses_portable_relative_links_and_honest_test_labels(tmp_path):
    figures = dict(learning_curves=str(tmp_path / "figures/learning_curves.png"),
                   test_examples=str(tmp_path / "figures/test_examples.png"))
    gallery = "\n".join(S.figure_gallery(figures, tmp_path))
    assert "[Learning curves](<figures/learning_curves.png>)" in gallery
    assert "[Representative test predictions](<figures/test_examples.png>)" in gallery
    assert "Unlabelled visual examples; no best/worst accuracy ranking" in gallery
    assert "Screening comparisons: not generated" in gallery
    assert str(tmp_path) not in gallery


def test_mask_zoom_stays_in_image_and_includes_reference_pixels():
    mask = np.zeros((32, 48), bool)
    mask[0:4, 40:48] = True
    x0, y0, x1, y1 = S.mask_zoom_bounds(mask, minimum_size=12)
    assert 0 <= x0 < x1 <= 48 and 0 <= y0 < y1 <= 32
    assert x1-x0 == y1-y0
    assert mask[y0:y1, x0:x1].sum() == mask.sum()
    assert S.mask_zoom_bounds(np.zeros_like(mask)) is None
