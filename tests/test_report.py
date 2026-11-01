"""Report integration with the evaluator's actual saved-artifact schema."""
import json

import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image

from config import load_config, save_config
from evaluate import evaluate_model
from report import create_report, prediction_examples


def test_reports_render_actual_evaluation_artifacts_without_mislabelling_pq(tmp_path, monkeypatch):
    run = tmp_path / "run"
    run.mkdir()
    gt = torch.zeros((32, 32), dtype=torch.int32)
    gt[5:12, 8:20] = 1
    good = torch.where(gt > 0, 12.0, -12.0)[None, None]
    names = ["20150101000000Mh", "20150102000000Mh"]
    batches = [
        {"image": good, "gt_sets": gt[None, None], "name": [names[0]]},
        {"image": torch.full_like(good, -12), "gt_sets": gt[None, None].repeat(1, 3, 1, 1),
         "name": [names[1]]},
    ]
    metrics = evaluate_model(torch.nn.Identity(), batches, "cpu", {"inference": {"amp": False}},
                             run / "evaluation", save_predictions=True)
    summary = dict(run_name="synthetic", seed=42, primary_metric="observation_PQ_mean",
                   best_score=metrics["observation_PQ_mean"], best_validation=metrics,
                   parameters=1, weights_file_bytes=1024, training_seconds=1,
                   best_epoch=1, epochs_completed=1, overfit_images=0)
    (run / "summary.json").write_text(json.dumps(summary), encoding="utf8")
    pd.DataFrame([dict(epoch=1, train_loss=1, val_PQ_dataset=metrics["PQ_dataset"])]).to_csv(
        run / "history.csv", index=False)
    output = tmp_path / "report"
    frame = create_report(iter([run]), output)  # iterable must remain usable for distributions
    assert frame.iloc[0]["PQ"] == pytest.approx(0.4)
    assert frame.iloc[0]["checkpoint_selection_score"] == pytest.approx(0.5)
    assert pd.isna(frame.iloc[0]["Kaggle_PQ"])
    assert frame.iloc[0]["evaluation_provenance"] == "unverified_missing_checkpoint_hash"
    # The two saved distributions have different denominators and cannot be combined.
    assert np.load(run / "evaluation/filament_scores.npz")["iou"].tolist() == [1, 0, 0, 0]
    assert np.load(run / "evaluation/overlap_scores.npz")["iou"].tolist() == [1]

    cfg = load_config({"data": {"data_root": str(tmp_path / "raw"),
                                 "cache_dir": str(tmp_path / "cache")}})
    save_config(cfg, run / "config.yaml")
    import data as D
    catalog = pd.DataFrame([dict(obs=n, set_id=f"ann-{n}", file_name=f"{n}.png") for n in names])
    monkeypatch.setattr(D, "load_catalog", lambda _: catalog)
    monkeypatch.setattr(D, "load_image", lambda _: np.full((32, 32), 0.6, np.float32))
    monkeypatch.setattr(D, "load_instance_map", lambda *_: gt.numpy())
    prediction_examples(run, output, count=1)
    for filename in ["learning_curves.png", "filament_distributions.png",
                     "official_overlap_distributions.png", "overlap_relationships.png",
                     "validation_best.png", "validation_worst.png"]:
        with Image.open(output / filename) as image:
            assert image.width > 500 and image.height > 300
            image.verify()


def test_report_rejects_overfit_diagnostics_and_empty_run_list(tmp_path):
    (tmp_path / "summary.json").write_text('{"overfit_images":4}', encoding="utf8")
    with pytest.raises(ValueError, match="Diagnostic overfit"):
        create_report([tmp_path], tmp_path / "out")
    with pytest.raises(ValueError, match="at least one"):
        create_report([], tmp_path / "out")


def test_resumed_weights_reject_stale_evaluation_before_reporting(tmp_path):
    import hashlib
    from report import _evaluation_provenance

    weights = tmp_path / "best_weights.pt"
    weights.write_bytes(b"checkpoint before resumed training")
    summary = {"weights_checkpoint": str(weights)}
    (tmp_path / "summary.json").write_text(json.dumps(summary), encoding="utf8")
    evaluation = tmp_path / "evaluation"
    evaluation.mkdir()
    metadata = {"checkpoint_sha256": hashlib.sha256(weights.read_bytes()).hexdigest()}
    (evaluation / "summary.json").write_text(json.dumps(metadata), encoding="utf8")
    assert _evaluation_provenance(tmp_path, summary) == "verified_current_weights_sha256"

    # A resumed run overwrites the same weights filename, leaving old arrays behind.
    weights.write_bytes(b"new best checkpoint after resumed training")
    with pytest.raises(ValueError, match="Rerun evaluate.py"):
        create_report([tmp_path], tmp_path / "report")
    with pytest.raises(ValueError, match="Rerun evaluate.py"):
        prediction_examples(tmp_path, tmp_path / "examples")
    assert not (tmp_path / "report/model_comparison.csv").exists()

    metadata["checkpoint_sha256"] = hashlib.sha256(weights.read_bytes()).hexdigest()
    (evaluation / "summary.json").write_text(json.dumps(metadata), encoding="utf8")
    assert _evaluation_provenance(tmp_path, summary) == "verified_current_weights_sha256"
