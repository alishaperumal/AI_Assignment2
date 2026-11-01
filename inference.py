"""Shared whole-image inference for validation and the baseline submission.

The input is already preprocessed/normalised. Outputs are probabilities at the
original image size. Tiles are blended BEFORE thresholding/instance labelling,
so tile borders cannot create independent instance IDs.
"""
from __future__ import annotations

from contextlib import nullcontext

import numpy as np
import torch
import torch.nn.functional as F

from metrics import label_components


def _positions(length: int, tile: int, stride: int) -> list[int]:
    if length <= tile:
        return [0]
    positions = list(range(0, length - tile + 1, stride))
    if positions[-1] != length - tile:
        positions.append(length - tile)
    return positions


def _forward_probability(model, image: torch.Tensor, amp: bool) -> torch.Tensor:
    """Pad to the encoder's stride, infer, and discard the padding."""
    h, w = image.shape[-2:]
    padded = F.pad(image, (0, (-w) % 32, 0, (-h) % 32), mode="replicate")
    context = torch.autocast("cuda", dtype=torch.float16) if amp and image.is_cuda else nullcontext()
    with context:
        logits = model(padded)
    if not isinstance(logits, torch.Tensor) or logits.shape != padded.shape:
        raise ValueError("binary segmentation model must return logits shaped [B,1,H,W]")
    if not torch.isfinite(logits).all():
        raise ValueError("model produced non-finite logits")
    return torch.sigmoid(logits.float())[..., :h, :w]


@torch.inference_mode()
def predict_probability(model, image: torch.Tensor, cfg: dict | None = None) -> torch.Tensor:
    """Predict [B,1,H,W] probabilities on the input device at its original size.

    Config: ``resize`` (optional square model input size), ``tile_size``
    (optional overlapping inference), ``overlap`` (pixels, default 128), and
    ``amp`` (CUDA mixed precision, default True). Model mode is restored.
    """
    c = {"resize": None, "tile_size": None, "overlap": 128, "amp": True, **(cfg or {})}
    if image.ndim != 4 or image.shape[1] != 1 or min(image.shape[-2:]) < 1:
        raise ValueError("image must have shape [B,1,H,W] with non-empty spatial dimensions")
    if not image.is_floating_point() or not torch.isfinite(image).all():
        raise ValueError("image must contain finite, normalised floating-point values")
    original_size = image.shape[-2:]
    if c["resize"] is not None:
        size = int(c["resize"])
        if size < 1:
            raise ValueError("resize must be positive")
        if size <= min(original_size):
            # Match the training dataset's area downsampling.
            image = F.interpolate(image, size=(size, size), mode="area")
        else:
            image = F.interpolate(image, size=(size, size), mode="bilinear", align_corners=False)
    was_training = model.training
    model.eval()
    try:
        if c["tile_size"] is None:
            probability = _forward_probability(model, image, bool(c["amp"]))
        else:
            tile, overlap = int(c["tile_size"]), int(c["overlap"])
            if tile <= 0 or not 0 <= overlap < tile:
                raise ValueError("tile_size must be positive and 0 <= overlap < tile_size")
            height, width = image.shape[-2:]
            probability = torch.zeros_like(image, dtype=torch.float32)
            weights = torch.zeros((1, 1, height, width), device=image.device, dtype=torch.float32)
            for y in _positions(height, tile, tile - overlap):
                for x in _positions(width, tile, tile - overlap):
                    patch = image[..., y:y + tile, x:x + tile]
                    ph, pw = patch.shape[-2:]
                    # Non-zero raised-cosine weights reduce seams and cover edges.
                    wy = torch.hann_window(ph + 2, periodic=False, device=image.device)[1:-1]
                    wx = torch.hann_window(pw + 2, periodic=False, device=image.device)[1:-1]
                    weight = (wy[:, None] * wx[None, :]).clamp_min(1e-4)[None, None]
                    pred = _forward_probability(model, patch, bool(c["amp"]))
                    probability[..., y:y + ph, x:x + pw] += pred * weight
                    weights[..., y:y + ph, x:x + pw] += weight
            probability /= weights
        if probability.shape[-2:] != original_size:
            probability = F.interpolate(probability, size=original_size, mode="bilinear", align_corners=False)
        return probability.clamp_(0, 1)
    finally:
        model.train(was_training)


def probability_to_instances(probability: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """Baseline conversion: threshold the complete image, then 8-connectivity."""
    p = np.asarray(probability)
    if p.ndim != 2 or not np.isfinite(p).all() or np.any(p < 0) or np.any(p > 1):
        raise ValueError("probability must be a finite 2-D array in [0,1]")
    if not 0 < threshold < 1:
        raise ValueError("foreground threshold must lie strictly between 0 and 1")
    return label_components(p > threshold, connectivity=2)
