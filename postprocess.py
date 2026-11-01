"""Person C: configurable conversion of foreground probabilities into filaments.

The baseline (``inference.probability_to_instances``) keeps pixels strictly above
``threshold`` and labels each 8-connected region as one filament. This module
keeps that as the default and adds optional steps aimed at the error types seen
in validation (extra "islands", fragmented filaments and merged neighbours):

* ``seed_threshold`` (hysteresis): keep a region only if it contains at least one
  pixel above this higher cutoff. Weak extensions attached to a confident
  filament survive; weak isolated islands are removed.
* ``min_area``: remove predicted filaments with fewer pixels than this.
* ``close_radius``: morphological closing (adds pixels) to bridge small gaps.
* ``fill_holes``: fill background holes enclosed by a predicted region.
* ``bridge_radius``: give fragments separated by gaps of at most
  ``2 * bridge_radius`` pixels the same instance ID *without adding pixels*,
  repairing fragmentation while leaving the predicted extent unchanged.
* ``split_threshold``: split a region containing two or more separate confident
  cores (connected pixels above this cutoff, at least ``split_min_core_area``
  pixels each) with a watershed on the probability map. This targets touching
  neighbours that the baseline merges.

All steps operate on one complete image after any tiled inference has been
blended, so tile borders never create instance boundaries. With default
settings the output is identical to the baseline conversion.
"""
from __future__ import annotations

from typing import Mapping

import cv2
import numpy as np
from scipy import ndimage as ndi

from metrics import label_components

DEFAULT_POSTPROCESS = {
    "seed_threshold": None,
    "min_area": 0,
    "close_radius": 0,
    "fill_holes": False,
    "bridge_radius": 0,
    "split_threshold": None,
    "split_min_core_area": 50,
    "connectivity": 2,
}


def resolve_postprocess(cfg: Mapping | None = None, threshold: float = 0.5) -> dict:
    """Validate post-processing settings and fill defaults; unknown keys fail."""
    c = dict(DEFAULT_POSTPROCESS)
    unknown = set(cfg or {}) - set(c)
    if unknown:
        raise ValueError(f"Unknown postprocess settings: {sorted(unknown)}")
    c.update(cfg or {})
    if not 0 < float(threshold) < 1:
        raise ValueError("foreground threshold must lie strictly between 0 and 1")
    for key in ("min_area", "close_radius", "bridge_radius", "split_min_core_area"):
        value = c[key]
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 0:
            raise ValueError(f"postprocess.{key} must be a nonnegative integer")
        c[key] = int(value)
    if c["connectivity"] not in (1, 2):
        raise ValueError("postprocess.connectivity must be 1 (4-connected) or 2 (8-connected)")
    if not isinstance(c["fill_holes"], bool):
        raise ValueError("postprocess.fill_holes must be true or false")
    for key in ("seed_threshold", "split_threshold"):
        if c[key] is not None:
            c[key] = float(c[key])
            if not float(threshold) <= c[key] < 1:
                raise ValueError(f"postprocess.{key} must be null or in [threshold, 1)")
    if c["bridge_radius"] and c["split_threshold"] is not None:
        # Bridging would immediately re-join the touching parts produced by a split.
        raise ValueError("bridge_radius and split_threshold cannot be combined")
    return c


def is_baseline(cfg: Mapping | None) -> bool:
    """True when the settings reduce to plain threshold + connected components."""
    c = resolve_postprocess(cfg)
    return c == {**DEFAULT_POSTPROCESS, "connectivity": c["connectivity"]} and c["connectivity"] == 2


def _ellipse(radius: int) -> np.ndarray:
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))


def relabel_sequential(labels: np.ndarray) -> np.ndarray:
    """Map positive IDs to 1..N, preserving their order; 0 stays background."""
    labels = np.asarray(labels)
    ids = np.unique(labels)
    ids = ids[ids > 0]
    if ids.size == 0:
        return np.zeros(labels.shape, np.int32)
    lookup = np.zeros(int(ids.max()) + 1, np.int32)
    lookup[ids] = np.arange(1, ids.size + 1, dtype=np.int32)
    return lookup[labels]


def keep_seeded(labels: np.ndarray, probability: np.ndarray, seed_threshold: float) -> np.ndarray:
    """Hysteresis: remove regions whose maximum probability is not above the seed cutoff."""
    n = int(labels.max())
    if n == 0:
        return labels
    maxima = ndi.maximum(probability, labels, index=np.arange(1, n + 1))
    keep = np.zeros(n + 1, bool)
    keep[1:] = np.asarray(maxima) > seed_threshold
    return np.where(keep[labels], labels, 0)


def remove_small(labels: np.ndarray, min_area: int) -> np.ndarray:
    """Remove instances with fewer than ``min_area`` pixels."""
    if min_area <= 1 or not labels.any():
        return labels
    areas = np.bincount(labels.ravel())
    small = areas < min_area
    small[0] = False
    return np.where(small[labels], 0, labels)


def bridge_fragments(labels: np.ndarray, radius: int, connectivity: int = 2) -> np.ndarray:
    """Give regions separated by gaps of at most ``2 * radius`` pixels one ID; no pixels are added."""
    if radius <= 0 or not labels.any():
        return labels
    grown = cv2.dilate((labels > 0).astype(np.uint8), _ellipse(radius))
    groups = label_components(grown, connectivity)
    return np.where(labels > 0, groups, 0)


def split_by_cores(labels: np.ndarray, probability: np.ndarray, core_threshold: float,
                   min_core_area: int = 50, connectivity: int = 2) -> np.ndarray:
    """Watershed-split each region that contains at least two confident cores.

    Cores are connected pixels above ``core_threshold`` with at least
    ``min_core_area`` pixels. Each pixel of a split region joins the core it
    reaches first when flooding from high to low probability.
    """
    from skimage.segmentation import watershed

    cores = label_components(probability > core_threshold, connectivity)
    if cores.max() < 2:
        return labels
    core_areas = np.bincount(cores.ravel())
    small = core_areas < max(int(min_core_area), 1)
    small[0] = False
    cores = np.where(small[cores], 0, cores)
    out = labels.copy()
    next_id = int(labels.max()) + 1
    for region_id, box in enumerate(ndi.find_objects(labels), start=1):
        if box is None:
            continue
        region = labels[box] == region_id
        inside = np.unique(cores[box][region])
        inside = inside[inside > 0]
        if inside.size < 2:
            continue
        markers = np.where(region, cores[box], 0)
        flooded = watershed(-probability[box], markers=markers, mask=region,
                            connectivity=connectivity)
        view = out[box]
        for k, core_id in enumerate(inside):
            part = flooded == core_id
            if k:
                view[part] = next_id
                next_id += 1
    return out


def probability_to_instances_pp(probability: np.ndarray, threshold: float = 0.5,
                                cfg: Mapping | None = None) -> np.ndarray:
    """Convert one 2-D probability map to an int32 instance map (0 = background)."""
    c = resolve_postprocess(cfg, threshold)
    p = np.asarray(probability)
    if p.ndim != 2 or not np.isfinite(p).all() or np.any(p < 0) or np.any(p > 1):
        raise ValueError("probability must be a finite 2-D array in [0,1]")
    foreground = p > threshold
    if c["close_radius"]:
        closed = cv2.morphologyEx(foreground.astype(np.uint8), cv2.MORPH_CLOSE, _ellipse(c["close_radius"]))
        foreground |= closed.astype(bool)
    if c["fill_holes"]:
        foreground = ndi.binary_fill_holes(foreground)
    labels = label_components(foreground, c["connectivity"])
    if c["seed_threshold"] is not None:
        labels = keep_seeded(labels, p, c["seed_threshold"])
    if c["split_threshold"] is not None:
        labels = split_by_cores(labels, p, c["split_threshold"], c["split_min_core_area"], c["connectivity"])
    if c["bridge_radius"]:
        labels = bridge_fragments(labels, c["bridge_radius"], c["connectivity"])
    labels = remove_small(labels, c["min_area"])
    return relabel_sequential(labels)


def describe(threshold: float, cfg: Mapping | None = None) -> str:
    """Short human-readable label of the non-default settings."""
    c = resolve_postprocess(cfg, threshold)
    parts = [f"t={threshold:g}"]
    parts += [f"{k}={v}" for k, v in c.items() if v != DEFAULT_POSTPROCESS[k]]
    return " ".join(parts)
