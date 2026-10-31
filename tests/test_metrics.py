"""Unit tests for metrics.py on tiny hand-made masks (run: pytest -q)."""
import numpy as np
import pytest

import metrics as M


def blank(n=40):
    return np.zeros((n, n), np.int32)


def two_filaments():
    gt = blank()
    gt[5:10, 2:38] = 1      # horizontal bar
    gt[20:25, 2:38] = 2     # second bar
    return gt


def test_perfect_prediction_scores_one():
    gt = two_filaments()
    r = M.panoptic_quality(gt.copy(), gt)
    assert r["PQ"] == pytest.approx(1.0) and r["SQ"] == pytest.approx(1.0) and r["RQ"] == pytest.approx(1.0)
    rel = M.relations(gt.copy(), gt)
    assert rel["one_to_many"] == 0 and rel["many_to_one"] == 0


def test_relabelled_ids_do_not_matter():
    gt = two_filaments()
    pred = np.where(gt == 1, 7, np.where(gt == 2, 3, 0))
    assert M.panoptic_quality(pred, gt)["PQ"] == pytest.approx(1.0)


def test_empty_prediction_scores_zero():
    r = M.panoptic_quality(blank(), two_filaments())
    assert r["PQ"] == 0 and r["fn"] == 2 and r["tp"] == 0


def test_both_empty_is_one_by_convention():
    assert M.panoptic_quality(blank(), blank())["PQ"] == 1.0


def test_all_false_positives_scores_zero():
    r = M.panoptic_quality(two_filaments(), blank())
    assert r["PQ"] == 0 and r["fp"] == 2


def test_merged_filaments_below_one_and_counted():
    gt = two_filaments()
    pred = blank()
    pred[5:25, 2:38] = 1    # one big blob covering both bars
    r = M.panoptic_quality(pred, gt)
    assert r["PQ"] < 1.0
    rel = M.relations(pred, gt)
    assert rel["many_to_one"] == 1 and rel["one_to_many"] == 0


def test_split_filament_below_one_and_counted():
    gt = two_filaments()
    pred = gt.copy()
    pred[5:10, 20:38] = 3   # second half of filament 1 becomes its own instance
    pred[20:25, 2:38] = 2
    r = M.panoptic_quality(pred, gt)
    assert r["PQ"] < 1.0
    rel = M.relations(pred, gt)
    assert rel["one_to_many"] == 1 and rel["many_to_one"] == 0


def test_iou_exactly_half_is_not_a_match():
    gt = blank(); gt[0:10, 0:10] = 1                 # area 100
    pred = blank(); pred[0:10, 0:5] = 1              # area 50 -> IoU = 0.5 exactly
    assert M.panoptic_quality(pred, gt)["tp"] == 0
    pred2 = blank(); pred2[0:10, 0:6] = 1            # IoU 0.6
    assert M.panoptic_quality(pred2, gt)["tp"] == 1


def test_pq_equals_sq_times_rq():
    gt = two_filaments()
    pred = blank(); pred[5:10, 2:30] = 1; pred[30:35, 2:10] = 2
    r = M.panoptic_quality(pred, gt)
    assert r["PQ"] == pytest.approx(r["SQ"] * r["RQ"])
    iou = 28 / 36  # pred 5x28 inside gt 5x36
    assert r["PQ"] == pytest.approx(iou / (1 + 0.5 + 0.5))


def test_small_noise_island_is_fp_not_fragmentation():
    gt = two_filaments()
    pred = gt.copy(); pred[7, 40 - 1] = 0; pred[30:32, 30:32] = 5
    rel = M.relations(pred, gt)
    assert rel["one_to_many"] == 0 and rel["many_to_one"] == 0
    assert M.panoptic_quality(pred, gt)["fp"] == 1


def test_per_filament_scores_and_histogram_arrays():
    gt = two_filaments()
    pred = blank(); pred[5:10, 2:38] = 1             # finds only filament 1
    f = M.per_filament_scores(pred, gt)
    assert f["iou"].tolist() == pytest.approx([1.0, 0.0]) and f["dice"].tolist() == pytest.approx([1.0, 0.0])


def test_accumulator_pooling():
    acc = M.MetricAccumulator()
    gt = two_filaments()
    acc.add_pair(gt.copy(), gt, "a")
    acc.add_pair(blank(), gt, "b")
    s = acc.summary()
    assert s["tp"] == 2 and s["fn"] == 2 and s["fp"] == 0
    assert s["PQ_dataset"] == pytest.approx(2 / (2 + 1))   # sum IoU 2 / (2 + 0.5*2)
    assert s["PQ_mean"] == pytest.approx(0.5)


def test_annotator_agreement_pairs():
    gt = two_filaments()
    rows = M.annotator_agreement({"o1": [gt, gt.copy()], "o2": [gt]})
    assert len(rows) == 2 and all(r["PQ"] == pytest.approx(1.0) for r in rows)


def test_label_components():
    b = np.zeros((10, 10), bool); b[0:2, 0:2] = True; b[5:7, 5:7] = True
    assert M.label_components(b).max() == 2
    d = np.zeros((10, 10), bool); d[0, 0] = d[1, 1] = True
    assert M.label_components(d, 2).max() == 1 and M.label_components(d, 1).max() == 2


def test_shape_mismatch_raises():
    with pytest.raises(ValueError):
        M.panoptic_quality(np.zeros((3, 3)), np.zeros((4, 4)))