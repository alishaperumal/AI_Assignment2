"""Tiny synthetic MAGFiLO-like dataset so the pipeline can be tested without Kaggle data.

Creates ``<root>/train/train_images/*.png``, a COCO json with 1-3 annotation sets per
observation (some polygons degenerate / overlapping), and ``<root>/test/test_images``.
Names follow YYYYMMDDHHMMSSII. Images are 8-bit with a bright disc on a dark sky.
"""
import json
from pathlib import Path

import cv2
import numpy as np


def make_synthetic(root, n_obs=60, size=256, n_test=12, seed=0):
    rng = np.random.default_rng(seed)
    root = Path(root)
    (root / "train" / "train_images").mkdir(parents=True, exist_ok=True)
    (root / "test" / "test_images").mkdir(parents=True, exist_ok=True)
    instruments = ["Bh", "Cy", "Mh"]
    images, anns = [], []
    set_id, ann_id = 1, 1
    yy, xx = np.mgrid[:size, :size]

    def disc_image():
        r = size * 0.45
        d = np.hypot(yy - size / 2, xx - size / 2)
        img = np.where(d < r, 0.6 - 0.2 * (d / r) ** 2, 0.05)
        return img + rng.normal(0, 0.02, img.shape)

    def poly(cx, cy, length, width, ang):
        t = np.linspace(-1, 1, 6)
        top = [(cx + length * s * np.cos(ang) - width * np.sin(ang),
                cy + length * s * np.sin(ang) + width * np.cos(ang)) for s in t]
        bot = [(cx + length * s * np.cos(ang) + width * np.sin(ang),
                cy + length * s * np.sin(ang) - width * np.cos(ang)) for s in t[::-1]]
        return [float(v) for p in top + bot for v in p]

    for i in range(n_obs):
        day = 1 + i * 2  # spread over ~4 months
        month, dom = 1 + (day - 1) // 28, 1 + (day - 1) % 28
        name = f"2015{month:02d}{dom:02d}{(i * 7) % 24:02d}0000{instruments[i % 3]}"
        img = disc_image()
        fils = []
        for _ in range(rng.integers(0, 4)):  # some observations have no filament
            fils.append((rng.uniform(.3, .7) * size, rng.uniform(.3, .7) * size,
                         rng.uniform(.05, .15) * size, rng.uniform(1, 3), rng.uniform(0, np.pi)))
        for f in fils:
            p = np.array(poly(*f)).reshape(-1, 2).astype(int)
            cv2.polylines(img, [p], True, 0.25, 1)
        cv2.imwrite(str(root / "train" / "train_images" / f"{name}.png"),
                    (np.clip(img, 0, 1) * 255).astype(np.uint8))
        for _ in range(int(rng.integers(1, 4))):  # 1-3 annotators
            images.append(dict(id=f"{i % 5 + 1:02d}{set_id % 3 + 1:02d}01-{name}", file_name=f"{name}.png", height=size, width=size))
            for f in fils:
                jitter = rng.normal(0, 1.0, 5) * [1, 1, 1, 0.1, 0.05]
                seg = [poly(*(np.array(f) + jitter))]
                anns.append(dict(id=ann_id, image_id=images[-1]["id"], category_id=1, iscrowd=0,
                                 segmentation=seg, area=1.0, bbox=[0, 0, 1, 1]))
                ann_id += 1
            if i % 10 == 0:  # inject a degenerate ring (2 points)
                anns.append(dict(id=ann_id, image_id=images[-1]["id"], category_id=1, iscrowd=0,
                                 segmentation=[[1.0, 1.0, 2.0, 2.0]], area=0, bbox=[0, 0, 0, 0]))
                ann_id += 1
            set_id += 1
    json.dump(dict(images=images, annotations=anns, categories=[dict(id=1, name="f")]),
              open(root / "train" / "MAGFiLO_1.0_Annotations_kaggle2026_train.json", "w"))
    for j in range(n_test):
        name = f"2015{6 + j // 6:02d}{1 + (j * 4) % 28:02d}120000{instruments[j % 3]}"
        cv2.imwrite(str(root / "test" / "test_images" / f"{name}.png"),
                    (np.clip(disc_image(), 0, 1) * 255).astype(np.uint8))
    return root