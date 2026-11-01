"""Figures for the updated final presentation (dark theme, no axes or titles).

Writes square PNG panels to delivery/.build/final_additions/:
  * best/worst validation cases of the final model (reference, before cleanup, after cleanup)
  * noisy / faint / disagreeing-annotator examples selected by eda.ipynb
All inputs are recorded artifacts; nothing is retrained or re-scored here.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter
from skimage.segmentation import find_boundaries

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "delivery" / ".build" / "final_additions"
FINAL_PREDICTIONS = ROOT / "runs" / "person_c" / "compression" / "bench_pruned30_fp16" / "validation" / "predictions"
FINAL_POSTPROCESS = {"bridge_radius": 8, "min_area": 320}
BEST, WORST = "20160928143554Bh", "20210222222850Bh"
SET_COLOURS = [(0x3B, 0xC5, 0xC7), (0xE8, 0x91, 0x6F), (0xF2, 0xF2, 0xF2)]


def _grey(img: np.ndarray) -> np.ndarray:
    scaled = np.clip(img / max(float(np.percentile(img, 99.5)), 1e-6), 0, 1)
    return np.repeat(scaled[..., None], 3, axis=2)


def _window(focus: np.ndarray, minimum=700, pad=200):
    ys, xs = np.nonzero(focus)
    if not len(ys):
        return slice(0, 2048), slice(0, 2048)
    size = int(np.clip(max(np.ptp(ys), np.ptp(xs)) + pad, minimum, 2048))
    cy, cx = (ys.min() + ys.max()) // 2, (xs.min() + xs.max()) // 2
    y0 = int(np.clip(cy - size // 2, 0, 2048 - size))
    x0 = int(np.clip(cx - size // 2, 0, 2048 - size))
    return slice(y0, y0 + size), slice(x0, x0 + size)


def _densest_window(mask: np.ndarray, size=900, step=64):
    best, where = -1, (0, 0)
    integral = np.pad(mask.astype(np.int64).cumsum(0).cumsum(1), ((1, 0), (1, 0)))
    for y in range(0, 2048 - size + 1, step):
        for x in range(0, 2048 - size + 1, step):
            total = integral[y + size, x + size] - integral[y, x + size] - integral[y + size, x] + integral[y, x]
            if total > best:
                best, where = total, (y, x)
    y, x = where
    return slice(y, y + size), slice(x, x + size)


def _save(rgb: np.ndarray, path: Path, size=640) -> None:
    Image.fromarray((np.clip(rgb, 0, 1) * 255).astype(np.uint8)).resize((size, size), Image.LANCZOS).save(path)


def _overlay_instances(grey: np.ndarray, labels: np.ndarray, palette: np.ndarray) -> np.ndarray:
    rgb = grey.copy()
    mask = labels > 0
    rgb[mask] = 0.35 * rgb[mask] + 0.65 * palette[labels[mask] % len(palette)]
    return rgb


def case_panels(cfg) -> dict:
    import data as D
    from postprocess import probability_to_instances_pp

    catalog = D.load_catalog(Path(cfg["data"]["cache_dir"]))
    palette = np.random.default_rng(0).uniform(0.25, 1.0, size=(4096, 3))
    info = {}
    for role, obs in (("best", BEST), ("worst", WORST)):
        group = catalog[catalog.obs == obs]
        image = D.load_image(Path(cfg["data"]["data_root"]) / "train" / "train_images" / group.file_name.iloc[0])
        reference = D.load_instance_map(group.set_id.iloc[0], Path(cfg["data"]["cache_dir"]))
        with np.load(FINAL_PREDICTIONS / f"{obs}.npz", allow_pickle=False) as archive:
            probability = archive["probability"].astype(np.float32)
            stored = archive["instances"].astype(np.int64)
        before = probability_to_instances_pp(probability, 0.5, None)
        after = probability_to_instances_pp(probability, 0.5, FINAL_POSTPROCESS)
        if not np.array_equal(after > 0, stored > 0):
            raise ValueError(f"{obs}: recomputed cleanup differs from the stored final-model instances")
        # The worst case is about what was missed, so zoom on its largest human-marked filament.
        if role == "worst":
            ids, areas = np.unique(reference[reference > 0], return_counts=True)
            focus, minimum = reference == ids[np.argmax(areas)], 420
        else:
            focus, minimum = (reference > 0) | (before > 0) | (after > 0), 700
        window = _window(focus, minimum=minimum)
        grey = _grey(image)[window]
        panels = {"reference": _overlay_instances(grey, reference[window], palette),
                  "before": _overlay_instances(grey, before[window], palette)}
        removed = (before[window] > 0) & (after[window] == 0)
        final = _overlay_instances(grey, after[window], palette)
        final[removed] = [1.0, 0.1, 0.1]
        panels["after"] = final
        for name, rgb in panels.items():
            _save(rgb, OUT / f"case_{role}_{name}.png")
        info[role] = dict(observation=obs, crop_pixels=int(window[0].stop - window[0].start),
                          reference_filaments_set1=int(len(np.unique(reference)) - 1),
                          predicted_before=int(before.max()), predicted_after=int(after.max()),
                          predicted_before_in_crop=int(len(np.unique(before[window])) - 1),
                          removed_pixels_in_crop=int(removed.sum()))
    return info


def noisy_panels(cfg) -> dict:
    import data as D

    summary = json.loads((ROOT / "results" / "eda_summary.json").read_text(encoding="utf-8-sig"))["noisy_samples"]
    chosen = {"noisy": summary["noisy"][0], "faint": summary["faint"][0],
              "disagree": summary["disagreeing annotators"][0]}
    catalog = D.load_catalog(Path(cfg["data"]["cache_dir"]))
    info = {}
    for role, obs in chosen.items():
        group = catalog[catalog.obs == obs]
        image = D.load_image(Path(cfg["data"]["data_root"]) / "train" / "train_images" / group.file_name.iloc[0])
        maps = [D.load_instance_map(s, Path(cfg["data"]["cache_dir"])) for s in group.set_id]
        window = _densest_window(np.any([m > 0 for m in maps], axis=0))
        rgb = _grey(image)[window]
        for k, labels in enumerate(maps):
            edge = find_boundaries(labels[window] > 0, mode="outer")
            edge = np.asarray(Image.fromarray(edge.astype(np.uint8) * 255).filter(ImageFilter.MaxFilter(3))) > 0
            rgb[edge] = np.array(SET_COLOURS[k % 3]) / 255
        _save(rgb, OUT / f"noisy_{role}.png")
        info[role] = dict(observation=obs, n_annotation_sets=len(maps))
    return info


def main() -> dict:
    from config import load_config

    OUT.mkdir(parents=True, exist_ok=True)
    cfg = load_config(ROOT / "configs" / "person_c" / "seed45_selected.yaml")
    info = dict(cases=case_panels(cfg), noisy=noisy_panels(cfg))
    (OUT / "figures.json").write_text(json.dumps(info, indent=2) + "\n", encoding="utf8")
    print(json.dumps(info, indent=2))
    return info


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(ROOT))
    main()
