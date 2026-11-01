"""Person C post-processing: each step does what it claims, and defaults equal the baseline."""
import numpy as np
import pytest

from inference import probability_to_instances
from postprocess import (bridge_fragments, is_baseline, probability_to_instances_pp,
                         relabel_sequential, resolve_postprocess, split_by_cores)


def test_defaults_reproduce_baseline_conversion_exactly():
    rng = np.random.default_rng(0)
    for threshold in (0.3, 0.5, 0.7):
        probability = rng.random((64, 80)).astype(np.float32)
        expected = probability_to_instances(probability, threshold)
        result = probability_to_instances_pp(probability, threshold)
        assert result.dtype == np.int32
        assert np.array_equal(result, expected)
    assert is_baseline({}) and not is_baseline({"min_area": 5})


def test_hysteresis_keeps_weak_extension_and_drops_weak_island():
    p = np.zeros((20, 40), np.float32)
    p[5:8, 2:20] = 0.6     # weak filament body
    p[6, 4] = 0.95          # one confident seed pixel
    p[12:15, 25:35] = 0.6   # weak island without a seed
    labels = probability_to_instances_pp(p, 0.5, {"seed_threshold": 0.9})
    assert labels.max() == 1
    assert labels[5:8, 2:20].all() and not labels[12:15, 25:35].any()


def test_min_area_removes_small_regions_and_relabels():
    p = np.zeros((30, 30), np.float32)
    p[2:4, 2:4] = 0.9       # 4 pixels
    p[10:20, 10:20] = 0.9   # 100 pixels
    labels = probability_to_instances_pp(p, 0.5, {"min_area": 5})
    assert labels.max() == 1 and labels[10:20, 10:20].all() and not labels[2:4, 2:4].any()


def test_bridge_joins_close_fragments_without_adding_pixels():
    p = np.zeros((10, 40), np.float32)
    p[4:6, 2:10] = 0.9
    p[4:6, 14:25] = 0.9     # 4-pixel gap: bridged by radius 2 (gaps up to 2 * radius)
    p[4:6, 30] = 0.9        # 5-pixel gap: stays separate
    baseline = probability_to_instances_pp(p, 0.5)
    bridged = probability_to_instances_pp(p, 0.5, {"bridge_radius": 2})
    assert baseline.max() == 3
    assert np.array_equal(bridged > 0, baseline > 0)
    assert bridged[4, 5] == bridged[4, 20] != bridged[4, 30]
    assert np.array_equal(bridge_fragments(baseline, 0), baseline)


def test_closing_adds_gap_pixels_and_fill_holes_fills_enclosed_background():
    p = np.zeros((14, 20), np.float32)
    p[3:10, 2:9] = 0.9
    p[3:10, 11:18] = 0.9    # 2-pixel gap between two thick regions
    closed = probability_to_instances_pp(p, 0.5, {"close_radius": 2})
    assert probability_to_instances_pp(p, 0.5).max() == 2
    assert closed.max() == 1 and closed[6, 9:11].all()
    ring = np.zeros((9, 9), np.float32)
    ring[1:8, 1:8] = 0.9
    ring[3:6, 3:6] = 0.1
    filled = probability_to_instances_pp(ring, 0.5, {"fill_holes": True})
    assert filled[4, 4] == 1 and probability_to_instances_pp(ring, 0.5)[4, 4] == 0


def test_core_split_separates_two_confident_cores_joined_by_weak_neck():
    p = np.zeros((10, 30), np.float32)
    p[3:7, 2:12] = 0.95
    p[3:7, 18:28] = 0.95
    p[4:6, 12:18] = 0.6     # weak bridge between the two cores
    merged = probability_to_instances_pp(p, 0.5)
    split = probability_to_instances_pp(p, 0.5, {"split_threshold": 0.9, "split_min_core_area": 10})
    assert merged.max() == 1 and split.max() == 2
    assert np.array_equal(split > 0, merged > 0)
    assert split[4, 5] != split[4, 25]
    one_core = split_by_cores(merged, np.where(merged > 0, 0.6, 0).astype(np.float32), 0.9, 10)
    assert np.array_equal(one_core, merged)


def test_connectivity_one_separates_diagonal_neighbours():
    p = np.zeros((4, 4), np.float32)
    p[0, 0] = p[1, 1] = 0.9
    assert probability_to_instances_pp(p, 0.5).max() == 1
    assert probability_to_instances_pp(p, 0.5, {"connectivity": 1}).max() == 2


def test_relabel_sequential_preserves_order():
    labels = np.array([[0, 7, 7], [3, 0, 9]])
    assert np.array_equal(relabel_sequential(labels), [[0, 2, 2], [1, 0, 3]])


@pytest.mark.parametrize("settings, message", [
    ({"min_aera": 3}, "Unknown postprocess"),
    ({"min_area": -1}, "nonnegative"),
    ({"min_area": 2.5}, "nonnegative"),
    ({"seed_threshold": 0.3}, "threshold, 1"),
    ({"bridge_radius": 2, "split_threshold": 0.8}, "cannot be combined"),
    ({"connectivity": 3}, "connectivity"),
])
def test_invalid_settings_rejected(settings, message):
    with pytest.raises(ValueError, match=message):
        resolve_postprocess(settings, 0.5)


def test_config_validates_postprocess_and_tta():
    from config import load_config

    cfg = load_config({"postprocess": {"min_area": 160, "bridge_radius": 8}, "inference": {"tta": True}})
    assert cfg["postprocess"]["min_area"] == 160 and cfg["inference"]["tta"] is True
    with pytest.raises(ValueError, match="Unknown configuration key"):
        load_config({"postprocess": {"min_aera": 3}})
    with pytest.raises(ValueError, match="tta"):
        load_config({"inference": {"tta": "yes"}})
