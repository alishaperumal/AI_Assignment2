"""Tests for data.py on a synthetic dataset (run: pytest -q)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent))
import data as D
from synthetic import make_synthetic


@pytest.fixture(scope="module")
def env(tmp_path_factory):
    root = make_synthetic(tmp_path_factory.mktemp("raw"), n_obs=60, size=256)
    cache = tmp_path_factory.mktemp("cache")
    cat = D.build_cache(root / "train" / "MAGFiLO_1.0_Annotations_kaggle2026_train.json", cache, verbose=False)
    return root, cache, cat


def test_parse_file_name():
    r = D.parse_file_name("20150304123456Bh.png")
    assert r["instrument"] == "Bh" and r["time"] == pd.Timestamp("2015-03-04 12:34:56") and r["obs"] == "20150304123456Bh"
    with pytest.raises(ValueError):
        D.parse_file_name("garbage.png")


def test_catalog_and_degenerate_polygons(env):
    _, cache, cat = env
    assert cat["obs"].nunique() < len(cat)               # several annotators per observation
    assert (cat["n_degenerate_rings"] > 0).any()          # injected 2-point polygons were dropped
    inst = D.load_instance_map(cat.set_id.iloc[0], cache)
    assert inst.shape == (256, 256) and inst.max() == cat.n_inst.iloc[0]


def test_decode_rle_and_polygon():
    m, d = D.decode_annotation([[2, 2, 10, 2, 10, 10, 2, 10]], 16, 16)
    assert m.sum() > 0 and d == 0
    m, d = D.decode_annotation([[1, 1, 2, 2]], 16, 16)
    assert m.sum() == 0 and d == 1


def test_split_has_no_leakage_and_is_time_blocked(env):
    root, _, cat = env
    tcat = D.test_catalog(root / "test" / "test_images")
    s = D.make_split(cat, tcat, val_frac=0.2, block_days=14, gap_days=1, seed=1)
    tr, va, dr = set(s["train"]), set(s["val"]), set(s["dropped"])
    assert not tr & va and not tr & dr and not va & dr
    assert tr | va | dr == set(cat["obs"])
    # all sets of an observation share a side by construction (grouping is by obs)
    obs_t = cat.groupby("obs")["time"].first()
    vt = obs_t[list(va)].sort_values()
    # val times form contiguous blocks: no train observation lies strictly inside a val block
    assert 0.1 < s["report"]["val_fraction"] < 0.35
    gap = pd.Timedelta(days=1)
    assert all(((vt - t).abs() > gap).all() for t in obs_t[list(tr)])  # embargo respected
    s2 = D.make_split(cat, tcat, val_frac=0.2, block_days=14, gap_days=1, seed=1)
    assert s == s2                                         # deterministic


def test_consensus_majority_and_single():
    a = np.zeros((10, 10), np.int32); a[2:5, 2:8] = 1
    b = a.copy(); c = np.zeros_like(a); c[2:5, 2:5] = 1
    cons = D.consensus_instances([a, b, c])                # majority of 3 -> == a
    assert ((cons > 0) == (a > 0)).all()
    assert (D.consensus_instances([a]) == a).all()
    union = D.consensus_instances([a, c])                  # n=2 -> union
    assert ((union > 0) == ((a > 0) | (c > 0))).all()


def test_preprocessing_disc_clahe_local(env):
    root, _, cat = env
    img = D.load_image(root / "train" / "train_images" / cat.file_name.iloc[0])
    assert img.dtype == np.float32 and 0 <= img.min() and img.max() <= 1
    disc = D.estimate_disc_mask(img)
    assert 0.3 < disc.mean() < 0.9 and not disc[0, 0] and disc[128, 128]
    out = D.preprocess_image(img, dict(mask_disc=True, clahe=True, local_norm=True))
    assert out.shape == img.shape and out.dtype == np.float32 and 0 <= out.min() <= out.max() <= 1
    assert out[~disc].max() == 0


def test_datasets_shapes_and_positive_sampling(env):
    root, cache, cat = env
    tcat = D.test_catalog(root / "test" / "test_images")
    s = D.make_split(cat, tcat, val_frac=0.2, block_days=14, seed=0)
    img_dir = root / "train" / "train_images"
    items = D.build_items(cat, s["train"], img_dir, "per_set")
    n_sets = cat[cat.obs.isin(s["train"])].shape[0]
    assert len(items) == n_sets
    # take only items that contain filament to test forced-positive crops
    pos_items = [it for it in items if D.load_instance_map(it["set_ids"][0], cache).max() > 0]
    ds = D.FilamentDataset(pos_items, crop=64, pos_frac=1.0, augment=True, cache_dir=cache)
    hits = 0
    for i in range(30):
        b = ds[i % len(ds)]
        assert b["image"].shape == (1, 64, 64) and b["mask"].shape == (1, 64, 64)
        assert b["image"].dtype.is_floating_point
        hits += int(b["mask"].sum() > 0)
    assert hits >= 24                                      # most crops contain filament (augment may shift some out)
    cons_items = D.build_items(cat, s["train"], img_dir, "consensus")
    assert len(cons_items) == len(s["train"])
    vds = D.ObservationDataset(cat, s["val"], img_dir, cache_dir=cache)
    v = vds[0]
    assert v["image"].shape == (1, 256, 256) and v["gt_sets"].ndim == 3


def test_norm_stats(env):
    root, cache, cat = env
    s = D.compute_norm_stats(cat, list(cat.obs.unique()), root / "train" / "train_images", 10, cache)
    assert 0 < s["mean"] < 1 and s["std"] > 0 and s["pos_weight"] > 1