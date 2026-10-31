"""Local evaluation metrics for solar filament instance segmentation (Person A).

Implements Panoptic Quality (Kirillov et al., CVPR 2019) plus the extra
quantities the Kaggle judges look at:

* PQ, SQ, RQ (IoU > 0.5 matching, so every GT filament has at most one match)
* per-filament Dice and IoU (for histograms)
* one-to-many  : a GT filament split across >= 2 predictions (fragmentation)
* many-to-one  : a prediction covering >= 2 GT filaments (merging)

Instance maps are 2-D integer arrays: 0 = background, 1..N = filament ids.

ASSUMPTIONS TO CONFIRM ON THE KAGGLE EVALUATION PAGE
----------------------------------------------------
1. The exact definition of one-to-many / many-to-one is not given in the
   assignment. We use the overlap rule documented in :func:`relations`
   (parameters ``tau`` and ``own``). Say this on the slide.
2. How Kaggle combines several annotation sets per image is unknown. We provide
   both per-image-mean PQ and dataset-level PQ (pooled TP/FP/FN), and the
   caller decides how to pool annotators (see ``evaluate_observation``).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from scipy import ndimage as ndi

IOU_THRESHOLD = 0.5


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def label_components(binary: np.ndarray, connectivity: int = 2) -> np.ndarray:
    """Label connected components of a binary mask.

    Args:
        binary: 2-D array, non-zero = foreground.
        connectivity: 1 = 4-connected, 2 = 8-connected.

    Returns:
        int32 instance map (0 background, 1..N components).
    """
    structure = ndi.generate_binary_structure(2, connectivity)
    lab, _ = ndi.label(binary > 0, structure=structure)
    return lab.astype(np.int32)


def _overlap_table(pred: np.ndarray, gt: np.ndarray):
    """Pixel overlap between every (gt id, pred id) pair, background excluded.

    Returns:
        gt_ids, pred_ids (unique non-zero ids), gt_area, pred_area (dicts id->px)
        and ``inter`` a dict {(gt_id, pred_id): intersection pixels}.
    """
    pred = np.asarray(pred)
    gt = np.asarray(gt)
    if pred.shape != gt.shape:
        raise ValueError(f"shape mismatch: pred {pred.shape} vs gt {gt.shape}")
    g = gt.ravel().astype(np.int64)
    p = pred.ravel().astype(np.int64)

    gt_ids, gt_counts = np.unique(g[g > 0], return_counts=True)
    pred_ids, pred_counts = np.unique(p[p > 0], return_counts=True)
    gt_area = dict(zip(gt_ids.tolist(), gt_counts.tolist()))
    pred_area = dict(zip(pred_ids.tolist(), pred_counts.tolist()))

    both = (g > 0) & (p > 0)
    inter: Dict[Tuple[int, int], int] = {}
    if both.any():
        base = int(p.max()) + 1
        code = g[both] * base + p[both]
        codes, counts = np.unique(code, return_counts=True)
        for c, n in zip(codes.tolist(), counts.tolist()):
            inter[(c // base, c % base)] = n
    return gt_area, pred_area, inter


def _safe_div(a: float, b: float, default: float = 0.0) -> float:
    return a / b if b > 0 else default


# --------------------------------------------------------------------------- #
# Panoptic quality
# --------------------------------------------------------------------------- #
def match_instances(pred: np.ndarray, gt: np.ndarray, iou_thr: float = IOU_THRESHOLD):
    """Match predicted to ground-truth instances (IoU > ``iou_thr``).

    With ``iou_thr >= 0.5`` a match is unique on both sides (Kirillov et al.).

    Returns:
        matches: list of (gt_id, pred_id, iou)
        unmatched_gt: list of gt ids (false negatives)
        unmatched_pred: list of pred ids (false positives)
    """
    gt_area, pred_area, inter = _overlap_table(pred, gt)
    matches = []
    matched_g, matched_p = set(), set()
    for (gid, pid), n in inter.items():
        union = gt_area[gid] + pred_area[pid] - n
        iou = n / union
        if iou > iou_thr:
            matches.append((gid, pid, iou))
            matched_g.add(gid)
            matched_p.add(pid)
    unmatched_gt = sorted(set(gt_area) - matched_g)
    unmatched_pred = sorted(set(pred_area) - matched_p)
    return matches, unmatched_gt, unmatched_pred


def pq_from_counts(sum_iou: float, tp: int, fp: int, fn: int) -> Dict[str, float]:
    """PQ = SQ * RQ from accumulated counts.

    PQ = sum(IoU of TP) / (TP + FP/2 + FN/2); SQ = sum(IoU)/TP; RQ = TP/(TP+FP/2+FN/2).
    With no TP, SQ is defined as 0. (If there is nothing at all, PQ = 0 here; the
    per-image wrapper handles the empty/empty case.)
    """
    denom = tp + 0.5 * fp + 0.5 * fn
    pq = _safe_div(sum_iou, denom)
    sq = _safe_div(sum_iou, tp)
    rq = _safe_div(tp, denom)
    return {"PQ": pq, "SQ": sq, "RQ": rq}


def panoptic_quality(
    pred: np.ndarray,
    gt: np.ndarray,
    iou_thr: float = IOU_THRESHOLD,
    empty_value: float = 1.0,
) -> Dict[str, float]:
    """Panoptic quality of one image.

    Args:
        pred, gt: instance maps (0 = background).
        iou_thr: matching threshold (0.5).
        empty_value: PQ/SQ/RQ returned when BOTH maps are empty (nothing to find,
            nothing predicted). 1.0 is the natural choice; Kaggle may differ.

    Returns:
        dict with PQ, SQ, RQ, tp, fp, fn, sum_iou.
    """
    matches, fn_ids, fp_ids = match_instances(pred, gt, iou_thr)
    tp, fp, fn = len(matches), len(fp_ids), len(fn_ids)
    sum_iou = float(sum(m[2] for m in matches))
    if tp + fp + fn == 0:
        return dict(PQ=empty_value, SQ=empty_value, RQ=empty_value,
                    tp=0, fp=0, fn=0, sum_iou=0.0)
    out = pq_from_counts(sum_iou, tp, fp, fn)
    out.update(tp=tp, fp=fp, fn=fn, sum_iou=sum_iou)
    return out


# --------------------------------------------------------------------------- #
# Per-filament Dice / IoU
# --------------------------------------------------------------------------- #
def per_filament_scores(pred: np.ndarray, gt: np.ndarray) -> Dict[str, np.ndarray]:
    """Dice and IoU for every GT filament against its best-overlapping prediction.

    A GT filament with no overlapping prediction scores 0. Use these arrays for
    histograms (not just means).

    Returns:
        dict with arrays ``gt_id``, ``iou``, ``dice``, ``gt_area``.
    """
    gt_area, pred_area, inter = _overlap_table(pred, gt)
    best_iou = {g: 0.0 for g in gt_area}
    best_dice = {g: 0.0 for g in gt_area}
    for (gid, pid), n in inter.items():
        iou = n / (gt_area[gid] + pred_area[pid] - n)
        if iou > best_iou[gid]:
            best_iou[gid] = iou
            best_dice[gid] = 2 * n / (gt_area[gid] + pred_area[pid])
    ids = sorted(gt_area)
    return dict(
        gt_id=np.array(ids, dtype=np.int64),
        iou=np.array([best_iou[i] for i in ids], dtype=np.float64),
        dice=np.array([best_dice[i] for i in ids], dtype=np.float64),
        gt_area=np.array([gt_area[i] for i in ids], dtype=np.int64),
    )


def semantic_scores(pred: np.ndarray, gt: np.ndarray) -> Dict[str, float]:
    """Pixel-level (foreground vs background) Dice and IoU of the whole image."""
    p, g = pred > 0, gt > 0
    inter = float(np.logical_and(p, g).sum())
    ps, gs = float(p.sum()), float(g.sum())
    union = ps + gs - inter
    if ps + gs == 0:
        return {"pixel_dice": 1.0, "pixel_iou": 1.0}
    return {"pixel_dice": 2 * inter / (ps + gs), "pixel_iou": inter / union}


# --------------------------------------------------------------------------- #
# One-to-many / many-to-one
# --------------------------------------------------------------------------- #
def relations(
    pred: np.ndarray, gt: np.ndarray, tau: float = 0.1, own: float = 0.5
) -> Dict[str, object]:
    """Count fragmentation (one-to-many) and merging (many-to-one).

    ASSUMPTION (confirm against the Kaggle Evaluation page):

    * one-to-many: a GT filament ``g`` is *fragmented* if at least two
      predictions ``p`` satisfy |g∩p|/|p| >= ``own`` (the prediction mostly lies
      inside g) and |g∩p|/|g| >= ``tau`` (it covers a meaningful part of g).
    * many-to-one: a prediction ``p`` is a *merge* if at least two GT filaments
      ``g`` satisfy |g∩p|/|g| >= ``own`` (g is mostly covered by p) and
      |g∩p|/|p| >= ``tau``.

    The ``own`` condition stops tiny noise blobs touching a big filament from
    counting as fragmentation.

    Returns:
        dict: one_to_many (count of fragmented GT filaments), many_to_one (count
        of merging predictions), plus the id lists ``fragmented_gt`` and
        ``merged_pred`` for plotting the error taxonomy.
    """
    gt_area, pred_area, inter = _overlap_table(pred, gt)
    parts_of_gt: Dict[int, List[int]] = {}
    gts_in_pred: Dict[int, List[int]] = {}
    for (gid, pid), n in inter.items():
        if n / pred_area[pid] >= own and n / gt_area[gid] >= tau:
            parts_of_gt.setdefault(gid, []).append(pid)
        if n / gt_area[gid] >= own and n / pred_area[pid] >= tau:
            gts_in_pred.setdefault(pid, []).append(gid)
    fragmented = sorted(g for g, ps in parts_of_gt.items() if len(ps) >= 2)
    merged = sorted(p for p, gs in gts_in_pred.items() if len(gs) >= 2)
    return dict(
        one_to_many=len(fragmented),
        many_to_one=len(merged),
        fragmented_gt=fragmented,
        merged_pred=merged,
    )


# --------------------------------------------------------------------------- #
# One-call evaluation + accumulation over a dataset
# --------------------------------------------------------------------------- #
def evaluate_pair(
    pred: np.ndarray,
    gt: np.ndarray,
    iou_thr: float = IOU_THRESHOLD,
    tau: float = 0.1,
    own: float = 0.5,
) -> Dict[str, object]:
    """All metrics for one (prediction, ground-truth annotation set) pair."""
    res: Dict[str, object] = dict(panoptic_quality(pred, gt, iou_thr))
    res.update(semantic_scores(pred, gt))
    rel = relations(pred, gt, tau, own)
    res.update(rel)
    res["filament"] = per_filament_scores(pred, gt)
    res["n_gt"] = int(len(np.unique(gt[gt > 0])))
    res["n_pred"] = int(len(np.unique(pred[pred > 0])))
    return res


@dataclass
class MetricAccumulator:
    """Accumulates :func:`evaluate_pair` results over many images.

    ``summary()`` reports both the *dataset-level* PQ (pooled TP/FP/FN, the
    Kirillov definition) and the *mean per-image* PQ with its std, because we do
    not know which one Kaggle uses.
    """

    per_image: List[Dict[str, float]] = field(default_factory=list)
    _dice: List[np.ndarray] = field(default_factory=list)
    _iou: List[np.ndarray] = field(default_factory=list)
    _area: List[np.ndarray] = field(default_factory=list)
    tp: int = 0
    fp: int = 0
    fn: int = 0
    sum_iou: float = 0.0
    one_to_many: int = 0
    many_to_one: int = 0

    def add(self, result: Dict[str, object], name: Optional[str] = None) -> None:
        """Add the output of :func:`evaluate_pair`."""
        self.tp += int(result["tp"]); self.fp += int(result["fp"]); self.fn += int(result["fn"])
        self.sum_iou += float(result["sum_iou"])
        self.one_to_many += int(result["one_to_many"])
        self.many_to_one += int(result["many_to_one"])
        f = result["filament"]
        self._dice.append(f["dice"]); self._iou.append(f["iou"]); self._area.append(f["gt_area"])
        row = {k: float(result[k]) for k in
               ("PQ", "SQ", "RQ", "tp", "fp", "fn", "pixel_dice", "pixel_iou",
                "one_to_many", "many_to_one", "n_gt", "n_pred")}
        row["name"] = name
        self.per_image.append(row)

    def add_pair(self, pred: np.ndarray, gt: np.ndarray, name: Optional[str] = None, **kw) -> None:
        """Convenience: evaluate and add in one call."""
        self.add(evaluate_pair(pred, gt, **kw), name)

    def filament_arrays(self) -> Dict[str, np.ndarray]:
        """Concatenated per-filament dice / iou / area for histograms."""
        cat = lambda xs: np.concatenate(xs) if xs else np.zeros(0)
        return dict(dice=cat(self._dice), iou=cat(self._iou), gt_area=cat(self._area))

    def summary(self) -> Dict[str, float]:
        """Dataset-level and mean-per-image scores in one flat dict."""
        ds = pq_from_counts(self.sum_iou, self.tp, self.fp, self.fn)
        out = {f"{k}_dataset": v for k, v in ds.items()}
        if self.per_image:
            for k in ("PQ", "SQ", "RQ", "pixel_dice", "pixel_iou"):
                vals = np.array([r[k] for r in self.per_image])
                out[f"{k}_mean"] = float(vals.mean())
                out[f"{k}_std"] = float(vals.std())
        fa = self.filament_arrays()
        out["filament_dice_mean"] = float(fa["dice"].mean()) if fa["dice"].size else 0.0
        out["filament_iou_mean"] = float(fa["iou"].mean()) if fa["iou"].size else 0.0
        out.update(tp=self.tp, fp=self.fp, fn=self.fn,
                   one_to_many=self.one_to_many, many_to_one=self.many_to_one,
                   n_images=len(self.per_image))
        return out


def evaluate_observation(
    pred: np.ndarray, gt_sets: Sequence[np.ndarray], reduce: str = "mean", **kw
) -> Dict[str, float]:
    """Score one prediction against ALL annotation sets of its observation.

    Kaggle's rule for several annotators is unconfirmed, so both are offered:

    * ``reduce="mean"``: average PQ/SQ/RQ over annotators (default).
    * ``reduce="max"``: best-matching annotator.
    """
    results = [panoptic_quality(pred, g, kw.get("iou_thr", IOU_THRESHOLD)) for g in gt_sets]
    keys = ("PQ", "SQ", "RQ")
    if reduce == "max":
        best = max(results, key=lambda r: r["PQ"])
        return {k: best[k] for k in keys}
    return {k: float(np.mean([r[k] for r in results])) for k in keys}


# --------------------------------------------------------------------------- #
# Inter-annotator agreement (rough ceiling for any model)
# --------------------------------------------------------------------------- #
def annotator_agreement(sets_by_obs: Dict[str, Sequence[np.ndarray]]) -> List[Dict[str, float]]:
    """PQ of annotator i vs annotator j for every pair within each observation.

    Args:
        sets_by_obs: observation id -> list of instance maps (one per annotator).
            Observations with a single annotation set are skipped.

    Returns:
        One row per ordered pair (i scored against j as ground truth): PQ, SQ, RQ,
        one_to_many, many_to_one, pixel_dice. Average these for the "human
        ceiling" line on the Dataset slide.
    """
    rows = []
    for obs, sets in sets_by_obs.items():
        for i in range(len(sets)):
            for j in range(len(sets)):
                if i == j:
                    continue
                r = evaluate_pair(sets[i], sets[j])
                rows.append(dict(obs=obs, a=i, b=j, PQ=r["PQ"], SQ=r["SQ"], RQ=r["RQ"],
                                 one_to_many=r["one_to_many"], many_to_one=r["many_to_one"],
                                 pixel_dice=r["pixel_dice"]))
    return rows