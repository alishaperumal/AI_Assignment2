"""Checks for the shared prediction/evaluation path, without pretrained weights."""
import numpy as np
import pytest
import torch

from evaluate import evaluate_model
from inference import predict_probability, probability_to_instances


class IdentityLogits(torch.nn.Module):
    def forward(self, image):
        assert image.shape[-1] % 32 == 0 and image.shape[-2] % 32 == 0
        return image


def test_tiles_restore_shape_and_do_not_split_crossing_object():
    model = IdentityLogits().train()
    image = torch.full((1, 1, 67, 91), -12.0)
    image[..., 30:35, 3:88] = 12
    full = predict_probability(model, image, {"amp": False})
    tiled = predict_probability(model, image, {"tile_size": 32, "overlap": 12, "amp": False})
    assert model.training
    assert tiled.shape == image.shape
    assert torch.allclose(tiled, full, atol=2e-6)
    assert probability_to_instances(tiled[0, 0].numpy()).max() == 1


def test_resized_prediction_returns_original_dimensions():
    image = torch.full((2, 1, 43, 57), 2.0)
    probability = predict_probability(IdentityLogits(), image, {"resize": 32, "amp": False})
    assert probability.shape == image.shape
    assert torch.allclose(probability, torch.sigmoid(image), atol=1e-6)


def test_bad_tiling_or_probability_rejected():
    with pytest.raises(ValueError, match="overlap"):
        predict_probability(IdentityLogits(), torch.zeros(1, 1, 40, 40), {"tile_size": 32, "overlap": 32})
    with pytest.raises(ValueError, match="finite"):
        probability_to_instances(np.array([[np.nan]]))


def test_evaluator_uses_all_sets_but_equal_observation_weights(tmp_path):
    gt = torch.zeros((16, 16), dtype=torch.int32)
    gt[2:8, 3:10] = 1
    good = torch.where(gt > 0, 12.0, -12.0)[None, None]
    batches = [
        {"image": good, "gt_sets": gt[None, None], "name": ["observation_a"]},
        {"image": torch.full_like(good, -12), "gt_sets": gt[None, None].repeat(1, 3, 1, 1), "name": ["observation_b"]},
    ]
    summary = evaluate_model(IdentityLogits(), batches, "cpu", {"inference": {"amp": False}}, tmp_path, True)
    assert summary["observation_PQ_mean"] == pytest.approx(0.5)
    assert summary["annotation_set_PQ_mean"] == pytest.approx(0.25)
    assert summary["n_observations"] == 2
    assert summary["n_annotation_sets"] == 4
    assert summary["PQ_dataset"] == pytest.approx(0.4)
    assert summary["primary_metric"] == "PQ_dataset"
    assert (tmp_path / "annotation_sets.csv").exists()
    result = np.load(tmp_path / "predictions" / "observation_a.npz")
    assert result["probability"].shape == (16, 16)
    assert np.array_equal(result["instances"], gt.numpy())


def test_evaluator_uses_original_masks_for_overlapping_annotation(tmp_path):
    import json
    from types import SimpleNamespace
    import pandas as pd
    from pycocotools import mask as mu

    a = np.zeros((16, 16), np.uint8)
    b = a.copy()
    a[2:6, 2:6] = 1
    b[2:6, 4:8] = 1
    annotations = []
    for mask in [a, b]:
        rle = mu.encode(np.asfortranarray(mask))
        rle["counts"] = rle["counts"].decode("ascii")
        annotations.append({"image_id": "annotator-observation", "segmentation": rle})
    source = {"images": [{"id": "annotator-observation"}], "annotations": annotations}
    (tmp_path / "MAGFiLO_1.0_Annotations_kaggle2026_train.json").write_text(json.dumps(source))
    cat = pd.DataFrame([{"set_id": "annotator-observation", "overlap_px": 8, "n_empty_masks": 0}])
    dataset = SimpleNamespace(cat=cat, groups=[("observation", cat)], img_dir=tmp_path / "train_images")
    overwritten = a.astype(np.int32)
    overwritten[b > 0] = 2

    class Loader:
        def __iter__(self):
            yield {"image": torch.from_numpy(np.where(a > 0, 12.0, -12.0).astype(np.float32))[None, None],
                   "gt_sets": torch.from_numpy(overwritten)[None, None],
                   "name": ["observation"], "set_ids": [("annotator-observation",)]}

    loader = Loader()
    loader.dataset = dataset
    result = evaluate_model(IdentityLogits(), loader, "cpu", {"inference": {"amp": False}})
    assert result["PQ_dataset"] == pytest.approx(2 / 3)
    assert result["n_raw_annotation_sets"] == 1
    assert (result["tp"], result["fn"]) == (1, 1)
