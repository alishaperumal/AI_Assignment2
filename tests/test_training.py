"""Fast CPU checks for checkpoint continuation and gradient accumulation.

The continuation test uses stochastic dropout and shuffled, epoch-dependent
samples. Matching the final weights therefore checks meaningful resume state,
not just whether a checkpoint file can be opened.
"""
from copy import deepcopy

import pytest
import torch
from torch.utils.data import DataLoader, Dataset

import train
from config import load_config
from losses import build_loss


class ToyDataset(Dataset):
    def __init__(self, n=7):
        self.n, self.epoch = n, 0

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __len__(self):
        return self.n

    def __getitem__(self, index):
        rng = torch.Generator().manual_seed(1000 + self.epoch * 100 + index)
        image = torch.rand(1, 8, 8, generator=rng)
        return {"image": image, "mask": (image > 0.65).float(), "name": str(index)}


def toy_loaders(cfg):
    return (
        DataLoader(ToyDataset(), batch_size=2, shuffle=True,
                   generator=torch.Generator().manual_seed(cfg["seed"])),
        DataLoader(ToyDataset(2), batch_size=1, shuffle=False),
    )


def toy_model(cfg):
    return torch.nn.Sequential(torch.nn.Conv2d(1, 3, 3, padding=1), torch.nn.ReLU(),
                               torch.nn.Dropout2d(0.2), torch.nn.Conv2d(3, 1, 1))


def toy_evaluation(model, loader, device, cfg, **kwargs):
    model.eval()
    values = []
    for batch in loader:
        loss = torch.nn.functional.binary_cross_entropy_with_logits(model(batch["image"]), batch["mask"])
        values.append(float(loss.detach()))
    return {"observation_PQ_mean": 1.0 / (1.0 + sum(values) / len(values))}


@pytest.fixture
def cpu_training(monkeypatch):
    previous_threads = torch.get_num_threads()
    previous_determinism = torch.are_deterministic_algorithms_enabled()
    torch.set_num_threads(1)
    monkeypatch.setattr(train, "_make_loaders", toy_loaders)
    monkeypatch.setattr(train, "build_model", toy_model)
    monkeypatch.setattr(train, "_evaluate_model", toy_evaluation)
    yield
    torch.set_num_threads(previous_threads)
    torch.use_deterministic_algorithms(previous_determinism)


def experiment(tmp_path, name):
    split = tmp_path / "split.json"
    if not split.exists():
        split.write_text('{"train":["a"],"val":["b"]}', encoding="utf8")
    return load_config({"run": {"name": name, "output_dir": str(tmp_path / name)},
                        "seed": 17,
                        "data": {"split": str(split), "num_workers": 0, "val_num_workers": 0},
                        "model": {"encoder_weights": None},
                        "evaluation": {"primary_metric": "observation_PQ_mean"},
                        "train": {"device": "cpu", "epochs": 3, "learning_rate": 0.01,
                                  "amp": False, "patience": 0, "accumulation_steps": 2}})


def test_epoch_resume_matches_continuous_training(cpu_training, tmp_path):
    full_cfg = experiment(tmp_path, "continuous")
    resumed_cfg = experiment(tmp_path, "resumed")
    full_result = train.run_training(full_cfg)
    partial = train.run_training(resumed_cfg, max_epochs=1)
    assert partial["epochs_completed"] == 1 and partial["stop_reason"] == "invocation_limit"
    resumed_result = train.run_training(resumed_cfg, resume=tmp_path / "resumed" / "last.pt")
    full = torch.load(tmp_path / "continuous" / "last.pt", weights_only=False)
    resumed = torch.load(tmp_path / "resumed" / "last.pt", weights_only=False)
    assert full_result["optimizer_steps"] == resumed_result["optimizer_steps"] == 6
    assert resumed_result["epochs_completed"] == 3
    for key in full["model_state_dict"]:
        assert torch.equal(full["model_state_dict"][key], resumed["model_state_dict"][key])
    assert full["scheduler_state_dict"] == resumed["scheduler_state_dict"]
    assert [r["train_loss"] for r in full["history"]] == [r["train_loss"] for r in resumed["history"]]
    assert full["best_score"] == resumed["best_score"]
    assert (tmp_path / "resumed" / "best_weights.pt").exists()


def test_resume_rejects_changed_horizon_or_split(cpu_training, tmp_path):
    cfg = experiment(tmp_path, "resume_checks")
    train.run_training(cfg, max_epochs=1)
    checkpoint = tmp_path / "resume_checks" / "last.pt"
    changed = deepcopy(cfg)
    changed["train"]["epochs"] = 4
    with pytest.raises(ValueError, match="settings differ"):
        train.run_training(changed, resume=checkpoint)
    (tmp_path / "split.json").write_text('{"train":["changed"],"val":["b"]}', encoding="utf8")
    with pytest.raises(ValueError, match="split contents changed"):
        train.run_training(cfg, resume=checkpoint)


def test_new_run_cannot_overwrite_existing_results(cpu_training, tmp_path):
    cfg = experiment(tmp_path, "existing")
    train.run_training(cfg, max_epochs=1)
    with pytest.raises(FileExistsError):
        train.run_training(cfg)


def test_accumulation_handles_partial_windows_and_short_batches():
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        torch.manual_seed(2)
        small = torch.nn.Conv2d(1, 1, 1)
        large = deepcopy(small)
        dataset = ToyDataset(7)
        # batch2*accum2 -> batches (2+2), (2+1), equivalent to batch4 -> (4),(3).
        for model, batch_size, accumulation in ((small, 2, 2), (large, 4, 1)):
            loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
            optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
            scaler = torch.amp.GradScaler("cuda", enabled=False)
            result = train.train_one_epoch(model, loader, build_loss(), optimizer,
                                           torch.device("cpu"), scaler, gradient_clip=0,
                                           accumulation_steps=accumulation)
            assert result["optimizer_steps"] == 2 and result["train_examples"] == 7
        for a, b in zip(small.parameters(), large.parameters()):
            assert torch.allclose(a, b, atol=1e-7, rtol=1e-6)
    finally:
        torch.set_num_threads(previous_threads)
