"""Data pipeline for the solar filament segmentation task (Person A).

Everything between the raw Kaggle download and a PyTorch DataLoader lives here:

1. parsing file names (capture time + instrument) and the COCO json
2. decoding polygons -> integer instance maps, cached to disk as 16-bit PNGs
3. an observation-grouped, contiguous-time-block train/validation split
4. consensus targets (union for two sets, majority vote for three)
5. preprocessing (solar-disc mask, CLAHE, local normalisation) and augmentation
6. ``FilamentDataset`` (random crops, training) and ``ObservationDataset``
   (whole image + all annotation sets, validation)

Command line (run from the repo root)::

    python data.py build-cache   # decode polygons, write cache/ + cache/catalog.csv
    python data.py split         # load the frozen split; create only if absent
    python data.py consensus     # optional: cache majority-vote targets
    python data.py stats         # mean / std of training images -> cache/norm_stats.json

Terminology
-----------
* *image entry / annotation set*: one COCO ``images`` record (one annotation set for
  one observation). Its ``id`` is a STRING such as ``040301-20140609195854Bh`` and is
  the unique key (``set_id``); the part before the dash is stored as ``set_code``.
* *observation*: all entries sharing the same ``file_name`` stem. The split is
  done on observations so annotation sets of one image never straddle the split.

Only segmentation masks are used for training. Nothing from the file name (time,
instrument) is ever fed to a model; it is only used to build the split.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import pandas as pd
from scipy import ndimage as ndi

# --------------------------------------------------------------------------- #
# Paths and defaults
# --------------------------------------------------------------------------- #
DATA_ROOT = Path(os.environ.get("FILAMENT_DATA", "data"))
TRAIN_IMG_DIR = DATA_ROOT / "train" / "train_images"
TEST_IMG_DIR = DATA_ROOT / "test" / "test_images"
TRAIN_JSON = DATA_ROOT / "train" / "MAGFiLO_1.0_Annotations_kaggle2026_train.json"
CACHE_DIR = Path(os.environ.get("FILAMENT_CACHE", "cache"))
SPLIT_PATH = Path("splits/split_v1.json")

IMG_SIZE = 2048
TINY_AREA_PX = 20  # an instance smaller than this is reported as "tiny"

DEFAULT_PREPROCESS = {
    "mask_disc": False,       # zero everything outside the solar disc
    "clahe": False,           # contrast-limited adaptive histogram equalisation
    "clahe_clip": 2.0,
    "clahe_tile": 8,
    "local_norm": False,      # remove limb darkening / slow background trends
    "local_sigma": 60.0,      # px at full resolution
    "mean": 0.5,              # final fixed normalisation (see `python data.py stats`)
    "std": 0.25,
}

DEFAULT_AUG = {
    "hflip": 0.5,
    "vflip": 0.5,
    "rot90": 0.5,
    "affine": 0.5,            # small rotation / scale / shift
    "rotate_limit": 30,
    "scale_limit": 0.2,
    "brightness_contrast": 0.5,
    "brightness_limit": 0.2,
    "contrast_limit": 0.2,
    "noise": 0.3,
    "blur": 0.2,
}


def seed_everything(seed: int) -> None:
    """Seed python, numpy and torch (if installed) for reproducible runs."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:  # pragma: no cover
        pass


# --------------------------------------------------------------------------- #
# File names and COCO parsing
# --------------------------------------------------------------------------- #
_NAME_RE = re.compile(r"(\d{14})([A-Za-z]{2})")


def obs_id_from_name(file_name: str) -> str:
    """Observation id = file name without directory and extension."""
    return os.path.basename(file_name).split(".")[0]


def parse_file_name(file_name: str) -> Dict[str, object]:
    """Parse ``YYYYMMDDHHMMSSII`` into capture time and two-letter instrument code.

    Returns:
        dict(obs, time (pandas Timestamp), instrument (str)). Raises ValueError if
        the name does not follow the pattern.
    """
    obs = obs_id_from_name(file_name)
    m = _NAME_RE.search(obs)
    if not m:
        raise ValueError(f"file name does not match YYYYMMDDHHMMSSII: {file_name}")
    return dict(obs=obs, time=pd.to_datetime(m.group(1), format="%Y%m%d%H%M%S"),
                instrument=m.group(2))


def load_coco(json_path: Path = TRAIN_JSON) -> dict:
    """Read the COCO-style annotation json."""
    with open(json_path) as f:
        return json.load(f)


def annotations_by_set(coco: dict) -> Dict[str, list]:
    """Group COCO annotations by image-entry id (one entry = one annotation set)."""
    out: Dict[str, list] = defaultdict(list)
    for a in coco["annotations"]:
        out[a["image_id"]].append(a)
    return out


# --------------------------------------------------------------------------- #
# Polygon -> mask -> instance map
# --------------------------------------------------------------------------- #
def decode_annotation(seg, h: int = IMG_SIZE, w: int = IMG_SIZE) -> Tuple[np.ndarray, int]:
    """Decode one COCO ``segmentation`` into a binary mask.

    Handles polygon lists (possibly several rings per filament) and RLE dicts.
    Degenerate rings (fewer than 3 points / odd length) are dropped.

    Returns:
        (mask uint8 HxW, number of degenerate rings dropped)
    """
    from pycocotools import mask as mu

    dropped = 0
    if isinstance(seg, dict):  # RLE
        rle = mu.frPyObjects(seg, h, w) if isinstance(seg.get("counts"), list) else seg
        return mu.decode(rle).astype(np.uint8), 0
    rings = []
    for ring in seg or []:
        if len(ring) >= 6 and len(ring) % 2 == 0:
            rings.append(ring)
        else:
            dropped += 1
    if not rings:
        return np.zeros((h, w), np.uint8), dropped
    rle = mu.merge(mu.frPyObjects(rings, h, w))
    return mu.decode(rle).astype(np.uint8), dropped


def build_instance_map(anns: Sequence[dict], h: int = IMG_SIZE, w: int = IMG_SIZE):
    """Integer instance map for one annotation set (0 = background, 1..N filaments).

    Where two filaments overlap, the later annotation wins (overlap pixels are
    counted and returned so the EDA can report how often this happens).

    Returns:
        inst (uint16 HxW), stats dict(n_ann, n_inst, n_empty, n_degenerate_rings,
        overlap_px, areas list)
    """
    inst = np.zeros((h, w), np.uint16)
    overlap_px, n_deg, n_empty, areas = 0, 0, 0, []
    next_id = 1
    for a in anns:
        m, dropped = decode_annotation(a["segmentation"], h, w)
        n_deg += dropped
        if m.sum() == 0:
            n_empty += 1
            continue
        sel = m > 0
        overlap_px += int((inst[sel] > 0).sum())
        inst[sel] = next_id
        next_id += 1
    ids, counts = np.unique(inst[inst > 0], return_counts=True)
    areas = counts.tolist()  # after overwrites
    stats = dict(n_ann=len(anns), n_inst=int(len(ids)), n_empty=n_empty,
                 n_degenerate_rings=n_deg, overlap_px=overlap_px, areas=areas)
    return inst, stats


def instance_path(set_id, cache_dir: Path = CACHE_DIR) -> Path:
    """Where the cached instance map of an annotation set lives."""
    return Path(cache_dir) / "instances" / f"{set_id}.png"


def load_instance_map(set_id, cache_dir: Path = CACHE_DIR) -> np.ndarray:
    """Load a cached instance map as int32."""
    arr = cv2.imread(str(instance_path(set_id, cache_dir)), cv2.IMREAD_UNCHANGED)
    if arr is None:
        raise FileNotFoundError(f"run `python data.py build-cache` first: {instance_path(set_id, cache_dir)}")
    return arr.astype(np.int32)


def _decode_one(args):
    """Worker: decode one annotation set, write PNG + stats json, return the stats."""
    set_id, anns, h, w, cache_dir, overwrite = args
    path = instance_path(set_id, cache_dir)
    sp = path.with_suffix(".json")
    if path.exists() and sp.exists() and not overwrite:
        with open(sp) as f:
            return set_id, json.load(f)
    inst, st = build_instance_map(anns, h, w)
    if not cv2.imwrite(str(path), inst):
        raise OSError(f"could not write instance cache {path}")
    with open(sp, "w") as f:
        json.dump(st, f)
    return set_id, st


def build_cache(
    json_path: Path = TRAIN_JSON,
    cache_dir: Path = CACHE_DIR,
    overwrite: bool = False,
    verbose: bool = True,
    workers: Optional[int] = None,
) -> pd.DataFrame:
    """Decode every annotation set to a cached 16-bit PNG and write the catalog.

    Resumable (each set also gets a small stats json, so an interrupted run can be
    restarted without losing statistics) and parallel over ``workers`` processes
    (default: at most two, to bound memory use). The catalog (``cache/catalog.csv``) has one row per
    annotation set with the parsed capture time / instrument and per-set
    statistics (number of filaments, foreground fraction, tiny/degenerate
    polygons, overlaps). It is the input to the EDA notebook and to the split.
    """
    from concurrent.futures import ProcessPoolExecutor

    coco = load_coco(json_path)
    by_set = annotations_by_set(coco)
    Path(cache_dir, "instances").mkdir(parents=True, exist_ok=True)
    jobs = [(str(im["id"]), by_set.get(im["id"], []), im.get("height", IMG_SIZE),
             im.get("width", IMG_SIZE), str(cache_dir), overwrite) for im in coco["images"]]
    workers = min(2, os.cpu_count() or 1) if workers is None else int(workers)
    if workers <= 0:
        raise ValueError("cache workers must be positive")
    stats = {}
    if workers > 1 and len(jobs) > 8:
        with ProcessPoolExecutor(workers) as ex:
            for i, (sid, st) in enumerate(ex.map(_decode_one, jobs, chunksize=8)):
                stats[sid] = st
                if verbose and (i + 1) % 100 == 0:
                    print(f"  decoded {i + 1}/{len(jobs)}", flush=True)
    else:
        for i, j in enumerate(jobs):
            sid, st = _decode_one(j)
            stats[sid] = st
            if verbose and (i + 1) % 100 == 0:
                print(f"  decoded {i + 1}/{len(jobs)}", flush=True)
    rows = []
    for im in coco["images"]:
        set_id = str(im["id"])
        h, w = im.get("height", IMG_SIZE), im.get("width", IMG_SIZE)
        info = parse_file_name(im["file_name"])
        st = stats[set_id]
        areas = np.array(st["areas"]) if st["areas"] else np.zeros(0)
        rows.append(dict(
            set_id=set_id, set_code=set_id.split("-")[0] if "-" in set_id else "",
            file_name=im["file_name"], obs=info["obs"], time=info["time"],
            instrument=info["instrument"], height=h, width=w,
            n_ann=st["n_ann"], n_inst=st["n_inst"], n_empty_masks=st["n_empty"],
            n_degenerate_rings=st["n_degenerate_rings"], overlap_px=st["overlap_px"],
            fg_frac=float(areas.sum() / (h * w)) if areas.size else 0.0,
            n_tiny=int((areas < TINY_AREA_PX).sum()),
            min_area=int(areas.min()) if areas.size else 0,
            max_area=int(areas.max()) if areas.size else 0,
            mean_area=float(areas.mean()) if areas.size else 0.0,
        ))
    cat = pd.DataFrame(rows).sort_values(["time", "set_id"]).reset_index(drop=True)
    cat.to_csv(Path(cache_dir) / "catalog.csv", index=False)
    if verbose:
        print(f"catalog: {len(cat)} annotation sets, {cat.obs.nunique()} observations")
    return cat


def load_catalog(cache_dir: Path = CACHE_DIR) -> pd.DataFrame:
    """Read ``cache/catalog.csv`` (parsed times)."""
    p = Path(cache_dir) / "catalog.csv"
    if not p.exists():
        raise FileNotFoundError("run `python data.py build-cache` first")
    return pd.read_csv(p, parse_dates=["time"], dtype={"set_id": str, "set_code": str}, keep_default_na=False)


# --------------------------------------------------------------------------- #
# Test file names (instrument mix only; never labels)
# --------------------------------------------------------------------------- #
def test_catalog(test_dir: Path = TEST_IMG_DIR) -> pd.DataFrame:
    """Parse capture time and instrument from the *test file names only*."""
    rows = []
    for p in sorted(Path(test_dir).glob("*")):
        if p.is_file():
            try:
                rows.append(parse_file_name(p.name))
            except ValueError:
                continue
    return pd.DataFrame(rows)

# --------------------------------------------------------------------------- #
# Observation-grouped, time-blocked split
# --------------------------------------------------------------------------- #
def make_split(
    catalog: pd.DataFrame,
    test_cat: Optional[pd.DataFrame] = None,
    val_frac: float = 0.17,
    block_days: float = 14.0,
    gap_days: float = 1.0,
    seed: int = 42,
    n_trials: int = 5000,
) -> dict:
    """Split observations into train/val using contiguous time blocks.

    Rules (from the team plan):
    * all annotation sets of an observation go to the same side;
    * validation consists of whole contiguous time blocks of ``block_days`` days,
      because images close in time / from one instrument are not independent;
    * observations within ``gap_days`` of any validation observation are removed
      from training ("embargo") so block edges do not leak;
    * among ``n_trials`` random block subsets whose size is within 3 percentage
      points of ``val_frac``, choose the one whose instrument mix is closest (L1)
      to the test file names (or to the whole training set if ``test_cat`` is
      None).

    Returns:
        dict with ``train`` / ``val`` / ``dropped`` observation id lists, the
        parameters and a small ``report`` (sizes, instrument mixes, date ranges).
    """
    obs = (catalog.groupby("obs")
           .agg(time=("time", "first"), instrument=("instrument", "first"),
                n_sets=("set_id", "count"))
           .reset_index().sort_values("time").reset_index(drop=True))
    t0 = obs["time"].min()
    obs["block"] = ((obs["time"] - t0).dt.total_seconds() // (block_days * 86400)).astype(int)
    blocks = sorted(obs["block"].unique())
    instruments = sorted(obs["instrument"].unique())
    if test_cat is not None and len(test_cat):
        ref = test_cat["instrument"].value_counts(normalize=True)
        instruments = sorted(set(instruments) | set(ref.index))
    else:
        ref = obs["instrument"].value_counts(normalize=True)
    ref_vec = np.array([ref.get(i, 0.0) for i in instruments])

    n_obs = len(obs)
    size = np.array([(obs["block"] == b).sum() for b in blocks], dtype=float)
    inst_counts = np.array([[((obs["block"] == b) & (obs["instrument"] == i)).sum()
                             for i in instruments] for b in blocks], dtype=float)

    rng = np.random.default_rng(seed)
    best, best_score = None, math.inf
    target = val_frac * n_obs
    for _ in range(n_trials):
        perm = rng.permutation(len(blocks))
        csum = np.cumsum(size[perm])
        k = int(np.searchsorted(csum, target)) + 1
        k = min(k, len(blocks) - 1)  # always keep at least one training block
        chosen = perm[:k]
        frac = size[chosen].sum() / n_obs
        if abs(frac - val_frac) > 0.03:
            continue
        mix = inst_counts[chosen].sum(0)
        mix = mix / mix.sum() if mix.sum() else mix
        score = np.abs(mix - ref_vec).sum() + 2.0 * abs(frac - val_frac)
        if score < best_score:
            best, best_score = chosen, score
    if best is None:  # fall back: take the closest single chunk of blocks
        best = np.arange(max(1, int(round(val_frac * len(blocks)))))
    val_blocks = {blocks[i] for i in best}

    obs["is_val"] = obs["block"].isin(val_blocks)
    val_times = np.sort(obs.loc[obs["is_val"], "time"].values)
    gap = np.timedelta64(int(gap_days * 86400), "s")

    def near_val(t) -> bool:
        if len(val_times) == 0:
            return False
        i = np.searchsorted(val_times, t)
        cands = []
        if i < len(val_times):
            cands.append(val_times[i] - t)
        if i > 0:
            cands.append(t - val_times[i - 1])
        return min(cands) <= gap

    train_mask = ~obs["is_val"]
    dropped_mask = train_mask & obs["time"].apply(lambda t: near_val(np.datetime64(t)))
    train_mask = train_mask & ~dropped_mask

    def mix_of(df):
        return df["instrument"].value_counts(normalize=True).round(4).to_dict()

    def span(df):
        return [str(df["time"].min()), str(df["time"].max())] if len(df) else None

    split = dict(
        train=obs.loc[train_mask, "obs"].tolist(),
        val=obs.loc[obs["is_val"], "obs"].tolist(),
        dropped=obs.loc[dropped_mask, "obs"].tolist(),
        params=dict(val_frac=val_frac, block_days=block_days, gap_days=gap_days,
                    seed=seed, n_trials=n_trials),
        val_blocks=[int(b) for b in sorted(val_blocks)],
        report=dict(
            n_obs=n_obs, n_train=int(train_mask.sum()), n_val=int(obs["is_val"].sum()),
            n_dropped=int(dropped_mask.sum()),
            val_fraction=float(obs["is_val"].mean()),
            instrument_mix_train=mix_of(obs[train_mask]),
            instrument_mix_val=mix_of(obs[obs["is_val"]]),
            instrument_mix_test=({k: round(float(v), 4) for k, v in ref.items()}
                                 if test_cat is not None and len(test_cat) else None),
            train_span=span(obs[train_mask]), val_span=span(obs[obs["is_val"]]),
        ),
    )
    assert_no_leak(split, catalog)
    return split


def assert_no_leak(split: dict, catalog: Optional[pd.DataFrame] = None) -> None:
    """Raise if an observation is on both sides, or (given the catalog) if any
    observation is missing from train/val/dropped. Since the split is keyed by
    observation, all annotation sets of one image are on the same side."""
    tr, va, dr = set(split["train"]), set(split["val"]), set(split.get("dropped", []))
    for name in ("train", "val", "dropped"):
        ids = split.get(name, [])
        if len(ids) != len(set(ids)):
            raise AssertionError(f"duplicate observations in {name}")
    for a, b, name in ((tr, va, "train/val"), (tr, dr, "train/dropped"), (va, dr, "val/dropped")):
        if a & b:
            raise AssertionError(f"{len(a & b)} observations in both {name}")
    if catalog is not None:
        unknown = (tr | va | dr) - set(catalog["obs"])
        if unknown:
            raise AssertionError(f"{len(unknown)} split observations absent from catalog: {sorted(unknown)[:3]}")
        missing = set(catalog["obs"]) - (tr | va | dr)
        if missing:
            raise AssertionError(f"{len(missing)} observations not assigned: {sorted(missing)[:3]}")


def save_split(split: dict, path: Path = SPLIT_PATH) -> None:
    """Create a split once; never overwrite an existing experiment partition.

    Saving the same contents is a no-op, preserving the original file bytes.
    Use a new versioned filename for an intentionally different split.
    """
    path = Path(path)
    assert_no_leak(split)
    if path.exists():
        if load_split(path) == split:
            return
        raise FileExistsError(f"Refusing to overwrite frozen split {path}; use a new versioned path")
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "x") as f:
        json.dump(split, f, indent=1)


def load_split(path: Path = SPLIT_PATH) -> dict:
    """Load the fixed split."""
    with open(path) as f:
        return json.load(f)


# --------------------------------------------------------------------------- #
# Multiple annotators: consensus target
# --------------------------------------------------------------------------- #
def consensus_instances(
    instance_maps: Sequence[np.ndarray], min_votes: Optional[int] = None, connectivity: int = 2
) -> np.ndarray:
    """Majority-vote target from several annotation sets of one observation.

    A pixel is foreground if at least ``min_votes`` annotators marked it. Default
    ``min_votes = ceil(n/2)``: majority for odd n, *union* for n = 2 (so a single
    annotation set is returned unchanged). Instances are the connected components
    of the voted mask (touching filaments therefore merge; this is a deliberate
    simplification for the consensus ablation).
    """
    n = len(instance_maps)
    if n == 1:
        return instance_maps[0].astype(np.int32)
    if min_votes is None:
        min_votes = math.ceil(n / 2)
    votes = sum((m > 0).astype(np.int32) for m in instance_maps)
    structure = ndi.generate_binary_structure(2, connectivity)
    lab, _ = ndi.label(votes >= min_votes, structure=structure)
    return lab.astype(np.int32)


def consensus_path(obs: str, cache_dir: Path = CACHE_DIR) -> Path:
    return Path(cache_dir) / "consensus" / f"{obs}.png"


def build_consensus_cache(catalog: pd.DataFrame, cache_dir: Path = CACHE_DIR, **kw) -> None:
    """Cache the majority-vote instance map of every observation."""
    out = Path(cache_dir) / "consensus"
    out.mkdir(parents=True, exist_ok=True)
    for obs, g in catalog.groupby("obs"):
        maps = [load_instance_map(s, cache_dir) for s in g["set_id"]]
        cv2.imwrite(str(consensus_path(obs, cache_dir)), consensus_instances(maps, **kw).astype(np.uint16))

# --------------------------------------------------------------------------- #
# Image loading and preprocessing
# --------------------------------------------------------------------------- #
def load_image(path: Path) -> np.ndarray:
    """Read a grayscale image as float32 in [0, 1] (handles 8/16-bit and RGB-stored gray)."""
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(path)
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY if img.shape[2] == 3 else cv2.COLOR_BGRA2GRAY)
    if img.dtype == np.uint8:
        return img.astype(np.float32) / 255.0
    if img.dtype == np.uint16:
        return img.astype(np.float32) / 65535.0
    return img.astype(np.float32)


def estimate_disc_mask(img: np.ndarray, work_size: int = 256) -> np.ndarray:
    """Boolean mask of the solar disc (bright disc on dark sky), full resolution.

    Works at low resolution: Otsu threshold -> largest connected component ->
    fill holes -> opening, then nearest-neighbour upsample. If the result is
    implausibly small (< 20 % of the frame) the whole frame is returned, so
    this never silently blanks an image.
    """
    from skimage.filters import threshold_otsu

    h, w = img.shape
    small = cv2.resize(img, (work_size, work_size), interpolation=cv2.INTER_AREA)
    small = cv2.GaussianBlur(small, (0, 0), 1.5)
    if float(small.max() - small.min()) < 1e-6:
        return np.ones((h, w), bool)
    binary = small > threshold_otsu(small)
    lab, n = ndi.label(binary)
    if n == 0:
        return np.ones((h, w), bool)
    sizes = ndi.sum(binary, lab, range(1, n + 1))
    disc = lab == (1 + int(np.argmax(sizes)))
    disc = ndi.binary_fill_holes(disc)
    disc = ndi.binary_opening(disc, iterations=2)
    if disc.mean() < 0.2:
        return np.ones((h, w), bool)
    return cv2.resize(disc.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)


def local_normalise(img: np.ndarray, disc: Optional[np.ndarray], sigma: float = 60.0) -> np.ndarray:
    """Remove slow intensity trends (limb darkening) and rescale local contrast.

    Subtracts a heavily blurred copy and divides by the local standard deviation,
    computed at 1/8 resolution for speed, then maps back to [0, 1] around 0.5 so
    photometric augmentation still behaves.
    """
    h, w = img.shape
    f = 8
    small = cv2.resize(img, (w // f, h // f), interpolation=cv2.INTER_AREA)
    wgt = (cv2.resize(disc.astype(np.float32), (w // f, h // f), interpolation=cv2.INTER_AREA)
           if disc is not None else np.ones_like(small))
    s = sigma / f
    num = cv2.GaussianBlur(small * wgt, (0, 0), s)
    den = cv2.GaussianBlur(wgt, (0, 0), s) + 1e-6
    mu = num / den
    var = cv2.GaussianBlur(((small - mu) ** 2) * wgt, (0, 0), s) / den
    mu_f = cv2.resize(mu, (w, h), interpolation=cv2.INTER_LINEAR)
    sd_f = cv2.resize(np.sqrt(var) + 1e-3, (w, h), interpolation=cv2.INTER_LINEAR)
    z = (img - mu_f) / sd_f
    return np.clip(0.5 + 0.2 * z, 0.0, 1.0).astype(np.float32)


def preprocess_image(img: np.ndarray, cfg: Optional[dict] = None) -> np.ndarray:
    """Apply the configured preprocessing; output stays in [0, 1] float32.

    Steps in order: disc estimate (if needed) -> local normalisation ->
    CLAHE -> zero outside the disc. The final mean/std normalisation is applied
    separately, *after* augmentation (``normalise``).
    """
    c = {**DEFAULT_PREPROCESS, **(cfg or {})}
    disc = estimate_disc_mask(img) if (c["mask_disc"] or c["local_norm"]) else None
    out = img
    if c["local_norm"]:
        out = local_normalise(out, disc, c["local_sigma"])
    if c["clahe"]:
        clahe = cv2.createCLAHE(clipLimit=float(c["clahe_clip"]),
                                tileGridSize=(int(c["clahe_tile"]),) * 2)
        out = clahe.apply((np.clip(out, 0, 1) * 65535).astype(np.uint16)).astype(np.float32) / 65535.0
    if c["mask_disc"]:
        out = out * disc
    return out.astype(np.float32)


def normalise(img: np.ndarray, cfg: Optional[dict] = None) -> np.ndarray:
    """Final fixed (x - mean) / std normalisation using constants from the config."""
    c = {**DEFAULT_PREPROCESS, **(cfg or {})}
    if not np.isfinite(c["mean"]) or not np.isfinite(c["std"]) or c["std"] <= 0:
        raise ValueError("normalisation mean must be finite and std must be positive and finite")
    return ((img - c["mean"]) / c["std"]).astype(np.float32)


# --------------------------------------------------------------------------- #
# Augmentation
# --------------------------------------------------------------------------- #
def build_augmentation(cfg: Optional[dict] = None, seed: int = 42):
    """albumentations pipeline for a crop (image in [0,1], mask as float32).

    Flips / 90-degree rotations are safe: the target is filament *shape*, and the
    Left/Right chirality class is not used. Noise, blur and brightness/contrast
    imitate seeing conditions of the ground-based GONG network.
    Requires albumentations >= 2.0.
    """
    import albumentations as A

    c = {**DEFAULT_AUG, **(cfg or {})}
    ts = []
    if c["hflip"]: ts.append(A.HorizontalFlip(p=c["hflip"]))
    if c["vflip"]: ts.append(A.VerticalFlip(p=c["vflip"]))
    if c["rot90"]: ts.append(A.RandomRotate90(p=c["rot90"]))
    if c["affine"]:
        ts.append(A.Affine(scale=(1 - c["scale_limit"], 1 + c["scale_limit"]),
                           rotate=(-c["rotate_limit"], c["rotate_limit"]),
                           translate_percent=(-0.05, 0.05), border_mode=cv2.BORDER_CONSTANT,
                           fill=0, fill_mask=0, interpolation=cv2.INTER_LINEAR,
                           mask_interpolation=cv2.INTER_NEAREST, p=c["affine"]))
    if c["brightness_contrast"]:
        ts.append(A.RandomBrightnessContrast(brightness_limit=c["brightness_limit"],
                                             contrast_limit=c["contrast_limit"],
                                             p=c["brightness_contrast"]))
    if c["noise"]: ts.append(A.GaussNoise(std_range=(0.01, 0.06), p=c["noise"]))
    if c["blur"]: ts.append(A.GaussianBlur(blur_limit=(3, 7), p=c["blur"]))
    return A.Compose(ts, seed=seed)


# --------------------------------------------------------------------------- #
# Items and datasets
# --------------------------------------------------------------------------- #
def build_items(
    catalog: pd.DataFrame,
    obs_list: Sequence[str],
    img_dir: Path = TRAIN_IMG_DIR,
    target: str = "per_set",
) -> List[dict]:
    """Turn observations into dataset items.

    Args:
        target: ``"per_set"`` = every annotation set is its own sample (Kaggle's
            suggestion); ``"consensus"`` = one majority-vote sample per observation.
    """
    obs_set = set(obs_list)
    sub = catalog[catalog["obs"].isin(obs_set)]
    items = []
    for obs, g in sub.groupby("obs", sort=True):
        img_path = Path(img_dir) / g["file_name"].iloc[0]
        set_ids = g["set_id"].tolist()
        if target == "per_set":
            items += [dict(obs=obs, set_ids=[s], image=img_path) for s in set_ids]
        elif target == "consensus":
            items.append(dict(obs=obs, set_ids=set_ids, image=img_path, consensus=True))
        else:
            raise ValueError(target)
    return items


def _load_target(item: dict, cache_dir: Path) -> np.ndarray:
    if item.get("consensus"):
        p = consensus_path(item["obs"], cache_dir)
        if p.exists():
            return cv2.imread(str(p), cv2.IMREAD_UNCHANGED).astype(np.int32)
        return consensus_instances([load_instance_map(s, cache_dir) for s in item["set_ids"]])
    return load_instance_map(item["set_ids"][0], cache_dir)


try:
    import torch
    from torch.utils.data import Dataset
except ImportError:  # allow data.py to be imported for EDA without torch
    torch = None
    Dataset = object


class FilamentDataset(Dataset):
    """Training dataset: random crops with forced-positive sampling and augmentation.

    Each ``__getitem__`` returns a dict with
    ``image`` (1,H,W float32, normalised), ``mask`` (1,H,W float32 binary),
    ``instances`` (H,W int32, original instance ids retained in the crop; for
    distance-transform / watershed targets) and ``name``.

    Args:
        items: from :func:`build_items`.
        crop: crop size in px (0 = whole image).
        pos_frac: probability that the crop is centred near a random filament
            pixel (filaments cover a tiny fraction of the image).
        augment: apply albumentations pipeline.
        preprocess / aug: config dicts (see ``DEFAULT_PREPROCESS``, ``DEFAULT_AUG``).
        epoch_length: samples per epoch (default len(items)); use a larger value
            to draw several crops per image per epoch.
        seed: controls sampling and augmentation independently of worker order.
        train_resize: optional square image/target size before crop sampling.
            Validation targets always retain their original resolution.

    Call ``set_epoch(epoch)`` before creating each epoch's loader iterator. A
    sample is determined by (seed, epoch, index); repeated visits to that tuple
    reproduce it, while subsequent epochs provide new crops and augmentations.
    """

    def __init__(self, items, crop=512, pos_frac=0.7, augment=True, preprocess=None,
                 aug=None, cache_dir=CACHE_DIR, epoch_length=None, seed=42,
                 train_resize=None):
        self.items, self.crop, self.pos_frac = list(items), int(crop), float(pos_frac)
        if not self.items:
            raise ValueError("training dataset has no items")
        if self.crop < 0 or not 0 <= self.pos_frac <= 1:
            raise ValueError("crop must be nonnegative and pos_frac must be between 0 and 1")
        self.pre, self.cache_dir = {**DEFAULT_PREPROCESS, **(preprocess or {})}, Path(cache_dir)
        self.seed, self.epoch = int(seed), 0
        if self.seed < 0:
            raise ValueError("seed must be nonnegative")
        self.train_resize = int(train_resize) if train_resize is not None else None
        if self.train_resize is not None and self.train_resize <= 0:
            raise ValueError("train_resize must be positive or None")
        self.tf = build_augmentation(aug, seed=self.seed) if augment else None
        self.epoch_length = len(self.items) if epoch_length is None else int(epoch_length)
        if self.epoch_length <= 0:
            raise ValueError("epoch_length must be positive")
        self.set_epoch(0)

    def __len__(self) -> int:
        return self.epoch_length

    def set_epoch(self, epoch: int) -> None:
        """Select deterministic samples/crops; shortened epochs cover the full dataset.

        A full pass retains normal item indexing. Other epoch lengths use
        shuffled cycles: every item appears once per complete cycle, with a
        random subset for the remainder. This avoids repeatedly training on a
        chronological prefix when limiting the number of samples per epoch.
        """
        if int(epoch) < 0:
            raise ValueError("epoch must be nonnegative")
        self.epoch = int(epoch)
        n_items = len(self.items)
        if n_items == 0 or self.epoch_length <= 0:
            raise ValueError("training dataset and epoch_length must be nonempty")
        if self.epoch_length == n_items:
            self._item_indices = np.arange(n_items)
        else:
            order_rng = np.random.default_rng(np.random.SeedSequence([self.seed, self.epoch, 0x5A17]))
            cycles = [order_rng.permutation(n_items)
                      for _ in range(math.ceil(self.epoch_length / n_items))]
            self._item_indices = np.concatenate(cycles)[:self.epoch_length]

    def _crop_origin(self, inst: np.ndarray, rng: np.random.Generator) -> Tuple[int, int]:
        h, w = inst.shape
        c = self.crop
        if rng.random() < self.pos_frac:
            fg = np.flatnonzero(inst > 0)
            if fg.size:
                y, x = divmod(int(rng.choice(fg)), w)
                y0 = int(np.clip(y - rng.integers(0, c), 0, h - c))
                x0 = int(np.clip(x - rng.integers(0, c), 0, w - c))
                return y0, x0
        return int(rng.integers(0, h - c + 1)), int(rng.integers(0, w - c + 1))

    def __getitem__(self, idx: int) -> dict:
        if not 0 <= idx < len(self):
            raise IndexError(idx)
        item = self.items[int(self._item_indices[idx])]
        crop_seed, aug_seed = np.random.SeedSequence(
            [self.seed, self.epoch, int(idx)]
        ).generate_state(2)
        rng = np.random.default_rng(int(crop_seed))
        img = preprocess_image(load_image(item["image"]), self.pre)
        inst = _load_target(item, self.cache_dir)
        if img.shape != inst.shape:
            raise ValueError(f"image/target shape mismatch for {item['obs']}: {img.shape} vs {inst.shape}")
        if self.train_resize is not None:
            size = (self.train_resize, self.train_resize)
            interpolation = cv2.INTER_AREA if self.train_resize < min(img.shape) else cv2.INTER_LINEAR
            img = cv2.resize(img, size, interpolation=interpolation)
            inst = cv2.resize(inst, size, interpolation=cv2.INTER_NEAREST)
        if self.crop > min(inst.shape):
            raise ValueError(f"crop {self.crop} exceeds image shape {inst.shape}")
        if self.crop and inst.shape != (self.crop, self.crop):
            y0, x0 = self._crop_origin(inst, rng)
            img = img[y0:y0 + self.crop, x0:x0 + self.crop]
            inst = inst[y0:y0 + self.crop, x0:x0 + self.crop]
        inst_f = inst.astype(np.float32)
        if self.tf is not None:
            self.tf.set_random_seed(int(aug_seed))
            out = self.tf(image=img, mask=inst_f)
            img, inst_f = out["image"], out["mask"]
        inst = np.rint(inst_f).astype(np.int32)
        x = normalise(img, self.pre)[None]
        return dict(image=torch.from_numpy(np.ascontiguousarray(x)),
                    mask=torch.from_numpy((inst > 0).astype(np.float32)[None]),
                    instances=torch.from_numpy(np.ascontiguousarray(inst)), name=item["obs"])


class ObservationDataset(Dataset):
    """Validation dataset: whole image plus ALL annotation sets of the observation.

    ``__getitem__`` returns ``image`` (1,H,W), ``mask`` (1,H,W; majority-vote
    foreground, for a validation loss), ``gt_sets`` (K,H,W int32, one instance map
    per annotator), ``set_ids`` (the corresponding annotation IDs) and ``name``.
    K varies between observations, so use
    ``batch_size=1`` when K differs.
    """

    def __init__(self, catalog, obs_list, img_dir=TRAIN_IMG_DIR, preprocess=None,
                 cache_dir=CACHE_DIR):
        self.cat = catalog
        self.groups = [(o, g) for o, g in catalog[catalog["obs"].isin(set(obs_list))].groupby("obs", sort=True)]
        self.img_dir, self.cache_dir = Path(img_dir), Path(cache_dir)
        self.pre = {**DEFAULT_PREPROCESS, **(preprocess or {})}

    def __len__(self) -> int:
        return len(self.groups)

    def __getitem__(self, idx: int) -> dict:
        obs, g = self.groups[idx]
        img = preprocess_image(load_image(self.img_dir / g["file_name"].iloc[0]), self.pre)
        sets = [load_instance_map(s, self.cache_dir) for s in g["set_id"]]
        cons = consensus_instances(sets) > 0
        return dict(image=torch.from_numpy(normalise(img, self.pre)[None]),
                    mask=torch.from_numpy(cons.astype(np.float32)[None]),
                    gt_sets=torch.from_numpy(np.stack(sets).astype(np.int32)),
                    set_ids=g["set_id"].tolist(), name=obs)


def seed_worker(worker_id: int) -> None:
    """Picklable Windows worker setup; sample RNGs are controlled by the dataset."""
    worker_seed = torch.initial_seed() % 2**32
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    cv2.setNumThreads(1)


def validate_data_inputs(catalog: pd.DataFrame, split: dict,
                         img_dir: Path, cache_dir: Path) -> None:
    """Fail before training if a split, source image, or cached target is missing."""
    assert_no_leak(split, catalog)
    if not split["train"] or not split["val"]:
        raise ValueError("the training and validation partitions must both be nonempty")
    if catalog["set_id"].duplicated().any():
        raise ValueError("annotation set IDs must be unique in the catalog")
    selected = catalog[catalog["obs"].isin(set(split["train"]) | set(split["val"]))]
    missing_images = [str(Path(img_dir) / fn) for fn in selected["file_name"].unique()
                      if not (Path(img_dir) / fn).is_file()]
    missing_masks = [str(instance_path(s, cache_dir)) for s in selected["set_id"]
                     if not instance_path(s, cache_dir).is_file()]
    if missing_images or missing_masks:
        raise FileNotFoundError(
            f"Data incomplete: {len(missing_images)} source images and "
            f"{len(missing_masks)} cached masks missing. Examples: "
            f"{(missing_images + missing_masks)[:3]}. Check data_root and run build-cache."
        )


def make_loaders(cfg: Optional[dict] = None):
    """Build train/validation loaders, honoring explicit paths and worker settings.

    Call ``train_loader.dataset.set_epoch(epoch)`` and seed its ``generator``
    consistently at each epoch boundary. Workers are intentionally nonpersistent
    so they receive the selected epoch before loading that epoch's samples.
    Validation always returns original-resolution images and annotation maps.
    """
    from torch.utils.data import DataLoader

    c = dict(crop=512, pos_frac=0.7, batch_size=8, num_workers=0,
             val_num_workers=0, target="per_set", preprocess={}, aug={},
             augment=True, epoch_length=None, seed=42, train_resize=None,
             data_root=str(DATA_ROOT), cache_dir=str(CACHE_DIR), split=str(SPLIT_PATH),
             pin_memory=None, drop_last=True)
    unknown = set(cfg or {}) - set(c)
    if unknown:
        raise ValueError(f"unknown data settings: {sorted(unknown)}")
    c.update(cfg or {})
    for name, allowed in (("preprocess", DEFAULT_PREPROCESS), ("aug", DEFAULT_AUG)):
        unknown = set(c[name] or {}) - set(allowed)
        if unknown:
            raise ValueError(f"unknown {name} settings: {sorted(unknown)}")
    if c["batch_size"] <= 0 or c["num_workers"] < 0 or c["val_num_workers"] < 0:
        raise ValueError("batch_size must be positive and worker counts nonnegative")
    root, cache_dir = Path(c["data_root"]), Path(c["cache_dir"])
    img_dir = root / "train" / "train_images"
    cat, split = load_catalog(cache_dir), load_split(Path(c["split"]))
    validate_data_inputs(cat, split, img_dir, cache_dir)
    train_items = build_items(cat, split["train"], img_dir, target=c["target"])
    train_ds = FilamentDataset(
        train_items, c["crop"], c["pos_frac"], c["augment"], c["preprocess"],
        c["aug"], cache_dir=cache_dir, epoch_length=c["epoch_length"],
        seed=c["seed"], train_resize=c["train_resize"],
    )
    val_ds = ObservationDataset(cat, split["val"], img_dir,
                                preprocess=c["preprocess"], cache_dir=cache_dir)
    if c["drop_last"] and len(train_ds) < c["batch_size"]:
        raise ValueError("drop_last would produce an empty training loader; reduce batch_size or disable it")
    g = torch.Generator().manual_seed(c["seed"])
    vg = torch.Generator().manual_seed(c["seed"] + 1)
    pin = torch.cuda.is_available() if c["pin_memory"] is None else bool(c["pin_memory"])
    return (
        DataLoader(train_ds, batch_size=c["batch_size"], shuffle=True,
                   drop_last=c["drop_last"], num_workers=c["num_workers"],
                   generator=g, pin_memory=pin, worker_init_fn=seed_worker),
        DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=c["val_num_workers"],
                   generator=vg, pin_memory=pin, worker_init_fn=seed_worker),
    )


# --------------------------------------------------------------------------- #
# Dataset statistics for normalisation / class balance
# --------------------------------------------------------------------------- #
def compute_norm_stats(catalog: pd.DataFrame, obs_list: Sequence[str],
                       img_dir: Path = TRAIN_IMG_DIR, max_images: int = 200,
                       cache_dir: Path = CACHE_DIR, preprocess: Optional[dict] = None) -> dict:
    """Mean/std of supplied training observations, plus foreground pixel fraction.

    Pass only the frozen split's training IDs. A deterministic sample includes at
    most ``max_images`` observations; no validation images are added implicitly.
    """
    if max_images <= 0:
        raise ValueError("max_images must be positive")
    obs = sorted(set(obs_list))
    if not obs:
        raise ValueError("cannot compute normalisation statistics on no observations")
    unknown = set(obs) - set(catalog["obs"])
    if unknown:
        raise ValueError(f"normalisation observations absent from catalog: {sorted(unknown)[:3]}")
    if len(obs) > max_images:
        indices = np.random.default_rng(0).choice(len(obs), size=max_images, replace=False)
        obs = [obs[i] for i in sorted(indices)]
    first = catalog[catalog["obs"].isin(obs)].groupby("obs").first()
    s = s2 = n = 0.0
    for fn in first["file_name"]:
        img = preprocess_image(load_image(Path(img_dir) / fn), preprocess)
        s += float(img.sum()); s2 += float((img.astype(np.float64) ** 2).sum()); n += img.size
    mean = s / n
    fg = float(catalog[catalog["obs"].isin(set(obs_list))]["fg_frac"].mean())
    stats = dict(mean=mean, std=math.sqrt(max(s2 / n - mean ** 2, 1e-12)), fg_frac=fg,
                 pos_weight=(1 - fg) / max(fg, 1e-9), n_images=len(first))
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    with open(Path(cache_dir) / "norm_stats.json", "w") as f:
        json.dump(stats, f, indent=1)
    return stats


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main(argv: Optional[Sequence[str]] = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["build-cache", "split", "consensus", "stats", "all"])
    ap.add_argument("--val-frac", type=float, default=0.17)
    ap.add_argument("--block-days", type=float, default=14.0)
    ap.add_argument("--gap-days", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--data-root", type=Path, default=DATA_ROOT)
    ap.add_argument("--cache-dir", type=Path, default=CACHE_DIR)
    ap.add_argument("--split", type=Path, default=SPLIT_PATH)
    ap.add_argument("--workers", type=int, default=2, help="cache-building processes")
    ap.add_argument("--max-images", type=int, default=200, help="training images for normalisation stats")
    ap.add_argument("--overwrite", action="store_true", help="rebuild masks only; never overwrite a split")
    a = ap.parse_args(argv)
    train_json = a.data_root / "train" / "MAGFiLO_1.0_Annotations_kaggle2026_train.json"
    train_dir = a.data_root / "train" / "train_images"
    test_dir = a.data_root / "test" / "test_images"

    if a.command in ("build-cache", "all"):
        build_cache(train_json, a.cache_dir, overwrite=a.overwrite, workers=a.workers)
    if a.command in ("split", "all"):
        cat = load_catalog(a.cache_dir)
        if a.split.exists():
            split = load_split(a.split)
            assert_no_leak(split, cat)
            print(f"Using existing frozen split {a.split}; creation parameters are not reapplied.")
        else:
            tcat = test_catalog(test_dir) if test_dir.exists() else None
            split = make_split(cat, tcat, a.val_frac, a.block_days, a.gap_days, a.seed)
            save_split(split, a.split)
        print(json.dumps(split["report"], indent=1))
    if a.command in ("consensus", "all"):
        build_consensus_cache(load_catalog(a.cache_dir), a.cache_dir)
    if a.command in ("stats", "all"):
        cat, split = load_catalog(a.cache_dir), load_split(a.split)
        assert_no_leak(split, cat)
        print(compute_norm_stats(cat, split["train"], train_dir,
                                 max_images=a.max_images, cache_dir=a.cache_dir))


if __name__ == "__main__":
    main()
