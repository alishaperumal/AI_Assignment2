"""Experiment budgeting, honest aggregation, and safe continuation checks."""
from copy import deepcopy
import json
from pathlib import Path

import pytest

from config import load_config, save_config
import experiments as E


def local_base(tmp_path):
    split = tmp_path / "split.json"
    split.write_text('{"train":["train_obs"],"val":["val_obs"]}', encoding="utf8")
    cache = tmp_path / "cache"
    cache.mkdir(exist_ok=True)
    (cache / "catalog.csv").write_text("obs\ntrain_obs\nval_obs\n", encoding="utf8")
    return load_config({"data": {"data_root": str(tmp_path / "raw"),
                                 "cache_dir": str(cache), "split": str(split)},
                        "model": {"architecture": "unetplusplus"},
                        "loss": {"name": "bce_tversky"}})


def test_run_config_inherits_only_machine_paths_and_fixes_budget(tmp_path):
    base = local_base(tmp_path)
    baseline = E.make_run_config("baseline", base, tmp_path / "screen", 43, 6, 960)
    consensus = E.make_run_config("consensus", base, tmp_path / "screen", 43, 6, 960)
    resolution = E.make_run_config("resolution_1024", base, tmp_path / "screen", 43, 6, 960)
    assert baseline["model"]["architecture"] == "unet"
    assert baseline["loss"]["name"] == "bce_dice"
    assert consensus["data"]["target"] == "consensus"
    assert resolution["data"]["train_resize"] == 1024
    for cfg in (baseline, consensus, resolution):
        assert cfg["data"]["data_root"] == base["data"]["data_root"]
        assert cfg["data"]["split"] == base["data"]["split"]
        assert cfg["seed"] == cfg["data"]["seed"] == 43
        assert cfg["train"]["epochs"] == 6 and cfg["data"]["epoch_length"] == 960
        assert cfg["train"]["patience"] == 0
        effective = cfg["data"]["batch_size"] * cfg["train"]["accumulation_steps"]
        assert 960 // effective == 240
    with pytest.raises(ValueError, match="divisible"):
        E.make_run_config("baseline", base, tmp_path / "screen", 43, 6, 961)


def test_comparison_key_groups_seeds_but_not_different_methods(tmp_path):
    base = local_base(tmp_path)
    a = E.make_run_config("baseline", base, tmp_path / "screen", 42, 6, 960)
    b = E.make_run_config("baseline", base, tmp_path / "screen", 43, 6, 960)
    c = E.make_run_config("tversky", base, tmp_path / "screen", 43, 6, 960)
    assert E.comparison_key(a) == E.comparison_key(b)
    assert E.comparison_key(a) != E.comparison_key(c)


def test_seed_summary_uses_sample_std_and_excludes_failed_or_partial_runs():
    records = [dict(experiment="baseline", comparison_key="same", seed=seed,
                    status="completed", PQ_dataset=pq)
               for seed, pq in [(42, 0.2), (43, 0.3), (44, 0.4)]]
    records += [dict(experiment="baseline", comparison_key="same", seed=45,
                     status="failed", PQ_dataset=0.99),
                dict(experiment="baseline", comparison_key="same", seed=46,
                     status="partial", PQ_dataset=0.99)]
    result = E.aggregate_seeds(records)[0]
    assert result["n_runs"] == result["PQ_dataset_n"] == 3
    assert result["PQ_dataset_mean"] == pytest.approx(0.3)
    assert result["PQ_dataset_std_across_runs"] == pytest.approx(0.1)
    single = E.aggregate_seeds(records[:1])[0]
    assert single["PQ_dataset_std_across_runs"] is None
    assert single["SQ_dataset_mean"] is None
    with pytest.raises(ValueError, match="Duplicate seed"):
        E.aggregate_seeds(records[:1] * 2)


def test_seed_summary_keeps_different_code_versions_separate():
    records = [dict(experiment="baseline", comparison_key="same", seed=seed,
                    source_fingerprint=source, status="completed", PQ_dataset=pq)
               for seed, source, pq in [(42, "old", 0.2), (43, "new", 0.4)]]
    summaries = E.aggregate_seeds(records)
    assert len(summaries) == 2
    assert all(row["n_runs"] == 1 and row["PQ_dataset_std_across_runs"] is None
               for row in summaries)


def test_fixed_budget_summary_excludes_early_stopped_and_missing_score(tmp_path):
    cfg = E.make_run_config("baseline", local_base(tmp_path), tmp_path / "screen", 42, 12, 960)
    summary = dict(epochs_completed=8, stop_reason="early_stopping", best_validation={"PQ_dataset": 0.2})
    record = E._apply_summary({}, summary, cfg)
    assert record["status"] == "partial"
    summary.update(epochs_completed=12, stop_reason="completed", best_validation={})
    record = E._apply_summary({}, summary, cfg)
    assert record["status"] == "failed" and "primary validation score" in record["error"]


def test_normalization_uses_only_train_ids_and_distinct_preprocessing_cache(tmp_path, monkeypatch):
    import data as D

    cfg = local_base(tmp_path)
    calls = []
    monkeypatch.setattr(D, "load_catalog", lambda p: "catalog")
    monkeypatch.setattr(D, "assert_no_leak", lambda *args: None)

    def compute(catalog, observations, img_dir, max_images, cache_dir, preprocess):
        calls.append((list(observations), deepcopy(preprocess), Path(cache_dir)))
        stats = dict(mean=0.4 if preprocess["clahe"] else 0.3, std=0.2,
                     n_images=1, fg_frac=0.01, pos_weight=99)
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        (Path(cache_dir) / "norm_stats.json").write_text(json.dumps(stats), encoding="utf8")
        return stats

    monkeypatch.setattr(D, "compute_norm_stats", compute)
    a, first = E.apply_training_normalization(cfg)
    repeated, second = E.apply_training_normalization(cfg)
    changed = deepcopy(cfg)
    changed["data"]["preprocess"]["clahe"] = True
    b, third = E.apply_training_normalization(changed)
    assert len(calls) == 2 and all(c[0] == ["train_obs"] for c in calls)
    assert first["key"] == second["key"] != third["key"]
    assert a["data"]["preprocess"]["mean"] == repeated["data"]["preprocess"]["mean"] == 0.3
    assert b["data"]["preprocess"]["mean"] == 0.4
    assert not (Path(cfg["data"]["cache_dir"]) / "norm_stats.json").exists()


def test_runner_records_success_skips_compatible_run_and_refuses_changed_budget(tmp_path, monkeypatch):
    base = local_base(tmp_path)
    calls = []
    monkeypatch.setattr(E, "apply_training_normalization",
                        lambda cfg, n: (cfg, {"key": "fixture", "n_images": 1}))

    def fake_train(command, log_path):
        calls.append(command)
        cfg = load_config(command[command.index("--config") + 1])
        run = Path(cfg["run"]["output_dir"])
        run.mkdir(parents=True)
        save_config(cfg, run / "config.yaml")
        (run / "run_metadata.json").write_text(json.dumps(dict(
            split_sha256=E._file_hash(Path(cfg["data"]["split"])),
            source_sha256={"train.py": "fixture"}, environment={"gpu": "test GPU"})), encoding="utf8")
        (run / "summary.json").write_text(json.dumps(dict(
            epochs_completed=cfg["train"]["epochs"], stop_reason="completed",
            best_validation={"PQ_dataset": 0.25}, training_seconds=2,
            parameters=10, optimizer_steps=1440)), encoding="utf8")
        return 0

    monkeypatch.setattr(E, "_run_subprocess", fake_train)
    output = tmp_path / "screen"
    records = E.run_suite(base, ["baseline"], output)
    assert records[0]["status"] == "completed" and records[0]["PQ_dataset"] == 0.25
    assert records[0]["Kaggle_PQ"] is None
    assert records[0]["source_fingerprint"] == E._json_hash({"train.py": "fixture"})
    assert (output / "experiments.csv").exists() and (output / "seed_summary.csv").exists()
    skipped = E.run_suite(base, ["baseline"], output, resume=True)
    assert len(calls) == 1
    assert skipped[0]["started_utc"] == records[0]["started_utc"]
    assert skipped[0]["finished_utc"] == records[0]["finished_utc"]
    with pytest.raises(ValueError, match="Settings changed"):
        E.run_suite(base, ["baseline"], output, epochs=7, resume=True)


def test_runner_failure_has_no_invented_score(tmp_path, monkeypatch):
    base = local_base(tmp_path)
    monkeypatch.setattr(E, "apply_training_normalization",
                        lambda cfg, n: (cfg, {"key": "fixture", "n_images": 1}))
    monkeypatch.setattr(E, "_run_subprocess", lambda *args: 1)
    records = E.run_suite(base, ["baseline"], tmp_path / "failure")
    assert records[0]["status"] == "failed"
    assert records[0].get("PQ_dataset") is None
    assert E.aggregate_seeds(records) == []
