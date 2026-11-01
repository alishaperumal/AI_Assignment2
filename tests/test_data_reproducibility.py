"""Regression checks for frozen splits and reproducible Windows data loading."""
import copy
from collections import Counter
import json
import pickle
from pathlib import Path

import numpy as np
import pytest
import torch

import data as D
from synthetic import make_synthetic


@pytest.fixture(scope="module")
def small_data(tmp_path_factory):
    root = make_synthetic(tmp_path_factory.mktemp("repro_raw"), n_obs=12, size=64)
    cache = tmp_path_factory.mktemp("repro_cache")
    cat = D.build_cache(
        root / "train" / "MAGFiLO_1.0_Annotations_kaggle2026_train.json",
        cache, workers=1, verbose=False,
    )
    obs = sorted(cat.obs.unique())
    split = dict(train=obs[:8], val=obs[8:], dropped=[])
    path = cache / "synthetic_split.json"
    D.save_split(split, path)
    return root, cache, cat, path


def loader_config(small_data, **overrides):
    root, cache, _, split = small_data
    cfg = dict(data_root=str(root), cache_dir=str(cache), split=str(split),
               crop=32, batch_size=2, num_workers=0, val_num_workers=0,
               epoch_length=8, seed=23, pin_memory=False)
    cfg.update(overrides)
    return cfg


def test_samples_repeat_by_seed_epoch_index_and_change_between_epochs(small_data):
    first, _ = D.make_loaders(loader_config(small_data))
    replica, _ = D.make_loaders(loader_config(small_data))
    ds = first.dataset
    original = ds[0]
    # Visiting other samples and changing global RNG state does not alter sample 0.
    ds[3]
    D.seed_everything(999)
    for candidate in (ds[0], replica.dataset[0]):
        assert torch.equal(original["image"], candidate["image"])
        assert torch.equal(original["instances"], candidate["instances"])
    ds.set_epoch(1)
    assert not torch.equal(original["image"], ds[0]["image"])
    ds.set_epoch(0)
    assert torch.equal(original["image"], ds[0]["image"])


def test_paths_and_workers_are_honored_and_spawn_is_reproducible(small_data):
    direct, direct_val = D.make_loaders(loader_config(small_data))
    workers, worker_val = D.make_loaders(
        loader_config(small_data, num_workers=2, val_num_workers=2)
    )
    assert direct.num_workers == direct_val.num_workers == 0
    assert workers.num_workers == worker_val.num_workers == 2
    pickle.dumps(workers.worker_init_fn)  # Regression: a local lambda cannot be pickled.
    # Use spawn on every platform to exercise the Windows startup contract.
    workers.multiprocessing_context = "spawn"
    worker_val.multiprocessing_context = "spawn"
    direct.dataset.set_epoch(3)
    workers.dataset.set_epoch(3)
    a, b = next(iter(direct)), next(iter(workers))
    assert a["name"] == b["name"]
    assert torch.equal(a["image"], b["image"])
    assert torch.equal(a["instances"], b["instances"])
    val = next(iter(worker_val))
    assert val["image"].shape == (1, 1, 64, 64)
    assert val["gt_sets"].shape[-2:] == (64, 64)
    expected_ids = worker_val.dataset.groups[0][1]["set_id"].tolist()
    assert [entry[0] for entry in val["set_ids"]] == expected_ids
    assert len(val["set_ids"]) == val["gt_sets"].shape[1]


def test_saved_split_is_never_rewritten(tmp_path):
    path = tmp_path / "split.json"
    split = dict(train=["a"], val=["b"], dropped=[])
    # Deliberately use non-default formatting to check preservation of original bytes.
    path.write_text(json.dumps(split, indent=4), encoding="utf8")
    before = path.read_bytes()
    D.save_split(split, path)
    assert path.read_bytes() == before
    with pytest.raises(FileExistsError, match="frozen split"):
        D.save_split(dict(train=["b"], val=["a"], dropped=[]), path)
    assert path.read_bytes() == before


def test_unknown_split_ids_and_config_keys_fail_early(small_data, tmp_path):
    _, _, cat, source = small_data
    bad = copy.deepcopy(D.load_split(source))
    bad["train"].append("missing_observation")
    with pytest.raises(AssertionError, match="absent from catalog"):
        D.assert_no_leak(bad, cat)
    bad = copy.deepcopy(D.load_split(source))
    bad["train"].append(bad["train"][0])
    with pytest.raises(AssertionError, match="duplicate"):
        D.assert_no_leak(bad, cat)
    with pytest.raises(ValueError, match="unknown data settings"):
        D.make_loaders(loader_config(small_data, typo_setting=1))
    with pytest.raises(FileNotFoundError):
        D.make_loaders(loader_config(small_data, split=str(tmp_path / "missing.json")))
    with pytest.raises(FileNotFoundError, match="source images"):
        D.make_loaders(loader_config(small_data, data_root=str(tmp_path)))


def test_resize_keeps_instance_ids_and_validation_resolution(small_data):
    tr, va = D.make_loaders(
        loader_config(small_data, train_resize=32, crop=0, augment=False, epoch_length=None)
    )
    batch = tr.dataset[0]
    assert batch["image"].shape == (1, 32, 32)
    assert batch["mask"].shape == (1, 32, 32)
    source = D.load_instance_map(tr.dataset.items[0]["set_ids"][0], tr.dataset.cache_dir)
    assert set(batch["instances"].unique().tolist()) <= set(np.unique(source).tolist())
    assert set(batch["mask"].unique().tolist()) <= {0.0, 1.0}
    assert va.dataset[0]["image"].shape == (1, 64, 64)
    assert va.dataset[0]["gt_sets"].shape[-2:] == (64, 64)


def test_statistics_use_only_requested_observations_and_hard_cap(small_data):
    root, cache, cat, split_path = small_data
    train = D.load_split(split_path)["train"]
    stats = D.compute_norm_stats(cat, train, root / "train" / "train_images", 3, cache)
    assert stats["n_images"] == 3
    assert stats["fg_frac"] == pytest.approx(cat[cat.obs.isin(train)].fg_frac.mean())
    assert stats == D.compute_norm_stats(cat, train, root / "train" / "train_images", 3, cache)


def test_empty_training_loader_is_rejected(small_data):
    with pytest.raises(ValueError, match="empty training loader"):
        D.make_loaders(loader_config(small_data, epoch_length=1, batch_size=2))
    tr, _ = D.make_loaders(loader_config(small_data, epoch_length=1, batch_size=2, drop_last=False))
    assert len(tr) == 1


def test_short_epochs_sample_beyond_prefix_and_oversampling_is_balanced(small_data):
    tr, _ = D.make_loaders(loader_config(small_data, epoch_length=2, augment=False))
    ds = tr.dataset
    prefix = {item["obs"] for item in ds.items[:2]}
    observed = set()
    for epoch in range(6):
        ds.set_epoch(epoch)
        observed.update(ds[i]["name"] for i in range(len(ds)))
    assert observed - prefix
    ds.set_epoch(2)
    expected = [ds[i]["name"] for i in range(len(ds))]
    ds.set_epoch(2)
    assert [ds[i]["name"] for i in range(len(ds))] == expected

    # Consensus supplies one item per observation, so counts are observable via names.
    tr, _ = D.make_loaders(
        loader_config(small_data, target="consensus", epoch_length=20, augment=False)
    )
    counts = Counter(tr.dataset[i]["name"] for i in range(len(tr.dataset)))
    assert len(counts) == len(tr.dataset.items)
    assert max(counts.values()) - min(counts.values()) <= 1
