"""COCO RLE orientation, instance identity and empty-image export behavior."""
import csv

import numpy as np
import pytest

mu = pytest.importorskip("pycocotools.mask")
from predict import encode_binary_mask, instance_rows, write_submission


def test_non_square_mask_exact_round_trip():
    mask = np.zeros((9, 13), np.uint8)
    mask[1:4, 7:12] = 1
    mask[7, 2] = 1
    counts = encode_binary_mask(mask, expected_shape=mask.shape)
    decoded = mu.decode({"size": list(mask.shape), "counts": counts.encode("ascii")})
    assert np.array_equal(mask, decoded)
    assert not counts.startswith("b'")


def test_csv_preserves_instances_and_reports_zero_detection(tmp_path):
    inst = np.zeros((9, 13), np.int32)
    inst[1:4, 7:12] = 4
    inst[7, 2] = 9
    path = tmp_path / "submission.csv"
    report = write_submission([
        ("20150125172714Mh", inst),
        ("20150126172714Mh", np.zeros_like(inst)),
    ], path, expected_shape=inst.shape)
    with path.open(newline="", encoding="utf8") as f:
        rows = list(csv.DictReader(f))
    assert [r["filament_id"] for r in rows] == ["20150125172714Mh_1", "20150125172714Mh_2"]
    for row, instance_id in zip(rows, [4, 9]):
        decoded = mu.decode({"size": list(inst.shape), "counts": row["segmentation_rle"].encode("ascii")})
        assert np.array_equal(decoded, inst == instance_id)
    assert report["zero_detection_images"] == ["20150126172714Mh"]
    assert report["n_images"] == 2 and report["n_filaments"] == 2
    assert report["kaggle_acceptance_verified"] is False


def test_export_rejects_invalid_masks_and_ids(tmp_path):
    with pytest.raises(ValueError, match="non-empty"):
        encode_binary_mask(np.zeros((3, 4), np.uint8), (3, 4))
    with pytest.raises(ValueError, match="shape"):
        encode_binary_mask(np.ones((3, 4), np.uint8))
    with pytest.raises(ValueError, match="observation"):
        list(instance_rows("bad-id", np.ones((3, 4), np.int32), (3, 4)))
    with pytest.raises(ValueError, match="duplicate"):
        write_submission([("20150125172714Mh", np.ones((3, 4), np.int32))] * 2,
                         tmp_path / "submission.csv", (3, 4))
    assert not (tmp_path / "submission.csv").exists()
