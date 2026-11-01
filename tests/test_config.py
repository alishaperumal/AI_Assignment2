"""Configuration failures that could silently change an experiment."""
import pytest

from config import load_config


def test_typo_is_rejected():
    with pytest.raises(ValueError, match="learning_raet"):
        load_config({"train": {"learning_raet": 0.01}})


def test_inheritance_and_seed(tmp_path):
    (tmp_path / "base.yaml").write_text("seed: 7\nmodel:\n  architecture: unet\n")
    (tmp_path / "child.yaml").write_text("extends: base.yaml\nmodel:\n  architecture: unetplusplus\n")
    cfg = load_config(tmp_path / "child.yaml")
    assert cfg["seed"] == cfg["data"]["seed"] == 7
    assert cfg["model"]["architecture"] == "unetplusplus"
    assert cfg["model"]["encoder_name"] == "resnet34"


def test_circular_inheritance(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text("extends: bad.yaml\n")
    with pytest.raises(ValueError, match="Circular"):
        load_config(path)


def test_invalid_matching_threshold():
    with pytest.raises(ValueError, match="IoU threshold"):
        load_config({"evaluation": {"iou_threshold": 0.3}})


def test_environment_path_survives_inheritance(tmp_path, monkeypatch):
    monkeypatch.setenv("FILAMENT_DATA", str(tmp_path / "images"))
    (tmp_path / "base.yaml").write_text("seed: 7\n")
    child = tmp_path / "child.yaml"
    child.write_text("extends: base.yaml\n")
    assert load_config(child)["data"]["data_root"] == str((tmp_path / "images").resolve())
