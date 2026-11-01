"""CPU-only handoff integrity checks using small, complete prediction exports."""
from copy import deepcopy
import csv
import json

import numpy as np
import pytest
import yaml

import package_person_b as P

coco = pytest.importorskip("pycocotools.mask")
SHAPE = (7, 9)
VALIDATION = ["20150125172714Mh", "20150126172714Mh"]
TEST = ["20160125172714Mh", "20160126172714Mh"]


def write_json(path, value):
    path.write_text(json.dumps(value), encoding="utf8")


def save_prediction(path, observation, instances, probability=None):
    if probability is None:
        probability = np.where(instances > 0, 0.8, 0.1).astype(np.float32)
    np.savez_compressed(path, observation=observation, instances=instances, probability=probability)


@pytest.fixture
def finished_run(tmp_path):
    run = tmp_path / "run"
    (run / "evaluation/predictions").mkdir(parents=True)
    (run / "test_predictions").mkdir()
    images = tmp_path / "dataset/test/test_images"
    images.mkdir(parents=True)
    for name in TEST:
        (images / f"{name}.jpeg").write_bytes(b"not loaded by packaging")
    split = tmp_path / "frozen_split.json"
    write_json(split, dict(train=["20140125172714Mh"], val=VALIDATION))
    cfg = dict(run=dict(name="selected_seed44", output_dir=str(run)), seed=44,
               data=dict(data_root=str(tmp_path / "dataset"), cache_dir=str(tmp_path / "cache"),
                         split=str(split), crop=512, preprocess=dict(mean=0.3, std=0.2)),
               model=dict(architecture="unet", encoder_name="resnet34", in_channels=1, classes=1),
               loss=dict(name="bce_dice"), train=dict(epochs=20, learning_rate=0.0003),
               inference=dict(resize=None), evaluation=dict(threshold=0.5, primary_metric="PQ_dataset"))
    for path in (run / "config.yaml", run / "evaluation/config.yaml"):
        path.write_text(yaml.safe_dump(cfg), encoding="utf8")
    (run / "best_weights.pt").write_bytes(b"opaque weights; packaging never unpickles or loads a model")
    (run / "last.pt").write_bytes(b"optimizer checkpoint must not be copied")
    checkpoint_hash, split_hash = P.file_sha256(run / "best_weights.pt"), P.file_sha256(split)
    write_json(run / "summary.json", dict(epochs_completed=20, stop_reason="completed", overfit_images=0,
                                          split_sha256=split_hash, best_validation=dict(PQ_dataset=0.3)))
    write_json(run / "run_metadata.json", dict(split_sha256=split_hash,
                                              source_sha256={"models.py": P.file_sha256(P.ROOT / "models.py")}))
    write_json(run / "history.json", [dict(epoch=20, train_loss=0.1)])
    (run / "history.csv").write_text("epoch,train_loss\n20,0.1\n", encoding="utf8")
    write_json(run / "evaluation/summary.json", dict(checkpoint_sha256=checkpoint_hash,
                                                    foreground_threshold=0.5, n_observations=2, PQ_dataset=0.3))
    instances = np.zeros(SHAPE, np.uint32)
    instances[1:3, 2:4] = 1
    instances[5, 7] = 2
    empty = np.zeros_like(instances)
    for name in VALIDATION:
        save_prediction(run / f"evaluation/predictions/{name}.npz", name, instances)
    for name, inst in zip(TEST, (instances, empty)):
        save_prediction(run / f"test_predictions/{name}.npz", name, inst)
    with (run / "submission.csv").open("w", newline="", encoding="utf8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["filament_id", "segmentation_rle"])
        writer.writeheader()
        for index in (1, 2):
            encoded = coco.encode(np.asfortranarray(instances == index, dtype=np.uint8))
            writer.writerow(dict(filament_id=f"{TEST[0]}_{index}", segmentation_rle=encoded["counts"].decode()))
    write_json(run / "submission.summary.json", dict(
        checkpoint_sha256=checkpoint_hash, config=cfg, foreground_threshold=0.5,
        n_images=2, n_filaments=2, zero_detection_images=[TEST[1]], kaggle_acceptance_verified=False,
        images=[dict(observation=TEST[0], n_instances=2), dict(observation=TEST[1], n_instances=0)]))
    return run


def verify(run):
    return P.verify_run(run, expected_shape=SHAPE, expected_val_count=2, expected_test_count=2)


def package(run, output, study=None):
    return P.package_run(run, output, study, expected_shape=SHAPE,
                         expected_val_count=2, expected_test_count=2)


def test_package_is_portable_complete_idempotent_and_preserves_input(finished_run, tmp_path):
    source_hashes = {p.relative_to(finished_run): P.file_sha256(p) for p in finished_run.rglob("*") if p.is_file()}
    study = tmp_path / "study"
    study.mkdir()
    write_json(study / "summary.json", dict(selected_config="baseline", final_seeds=[43, 44, 45]))
    write_json(study / "progress.json", dict(status="packaging in progress"))
    stable_files = ["protocol.json", "selection.json", "raw_dataset_manifest.json", "failure.json",
                    "methodology.md", "screening.csv", "figure.png"]
    for name in stable_files:
        (study / name).write_text("stable study artifact", encoding="utf8")
    output = tmp_path / "handoff"
    handoff = package(finished_run, output, study)
    assert handoff["checks"]["test"]["csv_masks_exactly_verified"]
    assert handoff["checks"]["test"]["zero_detection_images"] == [TEST[1]]
    assert not handoff["checks"]["kaggle_acceptance_verified"]
    assert not (output / "last.pt").exists() and not (output / "data").exists()
    original = yaml.safe_load((output / "config.original.yaml").read_text())
    portable = yaml.safe_load((output / "config.yaml").read_text())
    assert portable["data"]["split"] == "splits/split_v1.json"
    assert portable["data"]["data_root"] == "data"
    assert P._settings(original) == P._settings(portable)
    manifest_before = (output / "manifest.json").read_bytes()
    assert not (output / "study/progress.json").exists()
    assert all((output / "study" / name).is_file() for name in stable_files)
    write_json(study / "progress.json", dict(status="completed"))
    assert package(finished_run, output, study) == handoff
    assert (output / "manifest.json").read_bytes() == manifest_before
    manifest = json.loads(manifest_before)
    assert "study/progress.json" not in manifest["files"]
    for name, record in manifest["files"].items():
        assert P.file_sha256(output / name) == record["sha256"]
    assert source_hashes == {p.relative_to(finished_run): P.file_sha256(p)
                            for p in finished_run.rglob("*") if p.is_file()}
    (output / "best_weights.pt").write_bytes(b"tampered")
    with pytest.raises(FileExistsError, match="content changed"):
        package(finished_run, output, study)
    assert (output / "best_weights.pt").read_bytes() == b"tampered"


def test_stale_checkpoint_artifacts_are_rejected(finished_run):
    (finished_run / "best_weights.pt").write_bytes(b"newer trained checkpoint")
    with pytest.raises(ValueError, match="checkpoint hash"):
        verify(finished_run)


def test_missing_prediction_or_wrong_observation_is_rejected(finished_run):
    path = finished_run / f"evaluation/predictions/{VALIDATION[0]}.npz"
    save_prediction(path, VALIDATION[1], np.zeros(SHAPE, np.uint32))
    with pytest.raises(ValueError, match="identity"):
        verify(finished_run)
    path.unlink()
    with pytest.raises(ValueError, match="coverage"):
        verify(finished_run)


@pytest.mark.parametrize("error", ["shape", "range", "threshold"])
def test_invalid_prediction_arrays_are_rejected(finished_run, error):
    instances = np.zeros(SHAPE, np.uint32)
    probability = np.zeros(SHAPE, np.float32)
    if error == "shape":
        probability = np.zeros((2, 3), np.float32)
    elif error == "range":
        probability[0, 0] = np.nan
    else:
        probability[0, 0] = 0.9
    save_prediction(finished_run / f"evaluation/predictions/{VALIDATION[0]}.npz",
                    VALIDATION[0], instances, probability)
    with pytest.raises(ValueError, match={"shape": "shape", "range": "finite", "threshold": "threshold"}[error]):
        verify(finished_run)


def test_csv_rle_and_sidecar_coverage_are_independently_verified(finished_run):
    path = finished_run / "submission.csv"
    with path.open(newline="", encoding="utf8") as stream:
        rows = list(csv.DictReader(stream))
    rows[0]["segmentation_rle"] = rows[1]["segmentation_rle"]
    with path.open("w", newline="", encoding="utf8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["filament_id", "segmentation_rle"])
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(ValueError, match="RLE mask differs"):
        verify(finished_run)


def test_partial_training_and_config_drift_are_rejected(finished_run):
    path = finished_run / "evaluation/config.yaml"
    cfg = yaml.safe_load(path.read_text())
    changed = deepcopy(cfg)
    changed["inference"]["resize"] = 1024
    path.write_text(yaml.safe_dump(changed), encoding="utf8")
    with pytest.raises(ValueError, match="numerical settings"):
        verify(finished_run)
    summary = json.loads((finished_run / "summary.json").read_text())
    summary["stop_reason"] = "invocation_limit"
    write_json(finished_run / "summary.json", summary)
    with pytest.raises(ValueError, match="full-budget"):
        verify(finished_run)


def test_csv_duplicate_ids_and_false_empty_sidecar_are_rejected(finished_run):
    path = finished_run / "submission.csv"
    original = path.read_text(encoding="utf8")
    path.write_text(original + original.splitlines()[1] + "\n", encoding="utf8")
    with pytest.raises(ValueError, match="duplicate"):
        verify(finished_run)
    path.write_text(original, encoding="utf8")
    sidecar_path = finished_run / "submission.summary.json"
    sidecar = json.loads(sidecar_path.read_text())
    sidecar["zero_detection_images"] = []
    write_json(sidecar_path, sidecar)
    with pytest.raises(ValueError, match="zero-detection"):
        verify(finished_run)


def test_recorded_source_drift_refuses_packaging_before_writing(finished_run, tmp_path):
    path = finished_run / "run_metadata.json"
    metadata = json.loads(path.read_text())
    metadata["source_sha256"]["models.py"] = "changed"
    write_json(path, metadata)
    output = tmp_path / "must_not_be_created"
    with pytest.raises(ValueError, match="source snapshot differs"):
        package(finished_run, output)
    assert not output.exists()


@pytest.mark.parametrize("winerror", [5, 32, 33])
def test_publish_retries_windows_access_or_sharing_errors(tmp_path, monkeypatch, winerror):
    stage, output = tmp_path / "stage", tmp_path / "output"
    calls, sleeps = [], []
    error = OSError("Windows access/sharing failure")
    error.winerror = winerror

    def rename(source, target):
        calls.append((source, target))
        if len(calls) < 3:
            raise error

    monkeypatch.setattr(P.Path, "rename", rename)
    monkeypatch.setattr(P.time, "sleep", sleeps.append)
    P._publish_directory(stage, output)
    assert calls == [(stage, output)] * 3 and sleeps == [1.0, 1.0]


@pytest.mark.parametrize("winerror", [None, 112])
def test_publish_does_not_retry_other_errors(tmp_path, monkeypatch, winerror):
    error = OSError("Permanent filesystem failure")
    if winerror is not None:
        error.winerror = winerror
    calls, sleeps = [], []

    def rename(*args):
        calls.append(args)
        raise error

    monkeypatch.setattr(P.Path, "rename", rename)
    monkeypatch.setattr(P.time, "sleep", sleeps.append)
    with pytest.raises(OSError, match="Permanent"):
        P._publish_directory(tmp_path / "stage", tmp_path / "output")
    assert len(calls) == 1 and not sleeps


def test_publish_retry_is_bounded_and_rechecks_destination(tmp_path, monkeypatch):
    stage, output = tmp_path / "stage", tmp_path / "output"
    error = PermissionError("Still locked")
    error.winerror = 5
    calls, sleeps = [], []

    def rename(*args):
        calls.append(args)
        raise error

    monkeypatch.setattr(P.Path, "rename", rename)
    monkeypatch.setattr(P.time, "sleep", sleeps.append)
    with pytest.raises(PermissionError, match="Still locked"):
        P._publish_directory(stage, output)
    assert len(calls) == 6 and sleeps == [1.0] * 5
    calls.clear()

    def create_competing_destination(seconds):
        output.mkdir()
        (output / "existing.txt").write_text("keep", encoding="utf8")

    monkeypatch.setattr(P.time, "sleep", create_competing_destination)
    with pytest.raises(FileExistsError, match="Destination"):
        P._publish_directory(stage, output)
    assert len(calls) == 1
    assert (output / "existing.txt").read_text(encoding="utf8") == "keep"
