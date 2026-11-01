"""The offline runner must score saved probabilities exactly like evaluate_model."""
import numpy as np
import pytest
import torch

import data as D
from config import load_config
from evaluate import evaluate_model
from offline_eval import expand_candidates, make_candidate, robustness, run_candidates
from synthetic import make_synthetic


class SineLogits(torch.nn.Module):
    """Deterministic stand-in network producing varied, fragmented regions."""

    def forward(self, image):
        return 4 * torch.sin(3 * image)


@pytest.fixture(scope="module")
def synthetic_validation(tmp_path_factory):
    root = make_synthetic(tmp_path_factory.mktemp("offline_raw"), n_obs=12, size=64)
    cache = tmp_path_factory.mktemp("offline_cache")
    catalog = D.build_cache(root / "train" / "MAGFiLO_1.0_Annotations_kaggle2026_train.json",
                            cache, workers=1, verbose=False)
    observations = sorted(catalog.obs.unique())
    split = cache / "split.json"
    D.save_split(dict(train=observations[:7], val=observations[7:], dropped=[]), split)
    return root, cache, split


def config(synthetic_validation, postprocess=None):
    root, cache, split = synthetic_validation
    return load_config({"data": {"data_root": str(root), "cache_dir": str(cache), "split": str(split),
                                 "num_workers": 0, "val_num_workers": 0, "pin_memory": False},
                        "model": {"encoder_weights": None},
                        "inference": {"amp": False},
                        "evaluation": {"progress": False},
                        "postprocess": postprocess or {}})


@pytest.mark.parametrize("postprocess", [{}, {"bridge_radius": 2, "min_area": 3}, {"seed_threshold": 0.9}])
def test_offline_scores_equal_checkpoint_evaluation(synthetic_validation, tmp_path, postprocess):
    reference_cfg = config(synthetic_validation)
    _, val_loader = D.make_loaders(reference_cfg["data"])
    evaluate_model(SineLogits(), val_loader, "cpu", reference_cfg, tmp_path, save_predictions=True)
    cfg = config(synthetic_validation, postprocess)
    expected = evaluate_model(SineLogits(), val_loader, "cpu", cfg)
    candidate = make_candidate(cfg["evaluation"]["threshold"], postprocess)
    run = run_candidates(cfg, tmp_path / "predictions", [candidate], workers=1, check_stored=not postprocess)
    offline = run["accumulators"][0].summary()
    for key in ("PQ_dataset", "SQ_dataset", "RQ_dataset", "tp", "fp", "fn", "sum_iou", "one_to_many",
                "many_to_one", "any_overlap_one_to_many", "any_overlap_many_to_one", "observation_PQ_mean",
                "pixel_dice_mean", "n_annotation_sets", "n_observations"):
        assert offline[key] == pytest.approx(expected[key], abs=1e-12), key
    if not postprocess:
        assert run["stored_instances_match"] is True


def test_missing_archive_is_rejected(synthetic_validation, tmp_path):
    cfg = config(synthetic_validation)
    (tmp_path / "predictions").mkdir()
    with pytest.raises(ValueError, match="archives differ"):
        run_candidates(cfg, tmp_path / "predictions", [make_candidate()], workers=1)


def test_candidate_expansion_puts_baseline_first_and_deduplicates():
    candidates = expand_candidates([{"family": "t", "threshold": [0.4, 0.5]},
                                    {"family": "area", "min_area": [0, 10]}])
    assert [c["name"] for c in candidates] == ["baseline", "t=0.4", "t=0.5 min_area=10"]
    assert candidates[0]["family"] == "reference"


def test_split_half_and_bootstrap_on_known_counts():
    candidates = [make_candidate(name="baseline"), make_candidate(0.5, {"min_area": 5}, name="better")]
    rows = []
    for obs, block in (("a", 0), ("b", 1), ("c", 2), ("d", 3)):
        rows.append(dict(candidate=0, observation=obs, tp=1, fp=2, fn=1, sum_iou=0.8))
        rows.append(dict(candidate=1, observation=obs, tp=1, fp=0, fn=1, sum_iou=0.8))
    result = robustness(rows, candidates, {"a": 0, "b": 1, "c": 2, "d": 3}, n_boot=200)
    assert result["halves"]["A"]["blocks"] == [0, 2]
    for choice in result["split_half_selection"].values():
        assert choice["selected"] == "better"
        assert choice["held_out_gain"] == pytest.approx(0.8 / 1.5 - 0.8 / 2.5)
    intervals = {r["name"]: r for r in result["bootstrap"]["intervals"]}
    assert intervals["baseline"]["delta_PQ"] == 0
    assert intervals["better"]["delta_ci95_low"] > 0
