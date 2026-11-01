"""Two transparent binary segmentation losses used in the controlled experiments.

Both combine BCE with a soft overlap loss and operate directly on logits. Dice
weights false positives and negatives equally. Tversky can put more weight on
missed foreground: alpha weights false positives, beta false negatives.
"""
from __future__ import annotations

from typing import Mapping

import torch
from torch import nn
from torch.nn import functional as F


class BinaryOverlapLoss(nn.Module):
    """BCE plus Dice or Tversky, averaged over images, including empty crops."""

    def __init__(self, name="bce_dice", bce_weight=1.0, dice_weight=1.0,
                 alpha=0.3, beta=0.7, smooth=1.0, pos_weight=None):
        super().__init__()
        if name not in {"bce_dice", "bce_tversky"}:
            raise ValueError(f"Unknown loss {name!r}")
        if bce_weight < 0 or dice_weight < 0 or bce_weight + dice_weight <= 0:
            raise ValueError("Loss weights must be nonnegative with at least one positive weight")
        if alpha < 0 or beta < 0 or alpha + beta <= 0 or smooth <= 0:
            raise ValueError("Tversky weights must be nonnegative and smooth must be positive")
        if pos_weight is not None and pos_weight <= 0:
            raise ValueError("BCE pos_weight must be positive")
        self.name = name
        self.bce_weight, self.dice_weight = float(bce_weight), float(dice_weight)
        self.alpha, self.beta, self.smooth = float(alpha), float(beta), float(smooth)
        self.register_buffer("pos_weight", None if pos_weight is None else torch.tensor(float(pos_weight)))

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if logits.shape != target.shape or logits.ndim != 4 or logits.shape[1] != 1:
            raise ValueError("Logits and targets must share shape (batch, 1, height, width)")
        # Float32 reductions avoid overflowing large mask sums during AMP.
        logits, target = logits.float(), target.float()
        bce = F.binary_cross_entropy_with_logits(logits, target, pos_weight=self.pos_weight)
        p, y = logits.sigmoid().flatten(1), target.flatten(1)
        intersection = (p * y).sum(1)
        if self.name == "bce_dice":
            score = (2 * intersection + self.smooth) / (p.sum(1) + y.sum(1) + self.smooth)
        else:
            false_positive = (p * (1 - y)).sum(1)
            false_negative = ((1 - p) * y).sum(1)
            score = (intersection + self.smooth) / (
                intersection + self.alpha * false_positive + self.beta * false_negative + self.smooth)
        return self.bce_weight * bce + self.dice_weight * (1 - score).mean()


def build_loss(cfg: Mapping | None = None) -> BinaryOverlapLoss:
    """Construct a loss from the experiment's ``loss`` settings.

    ``dice_weight`` is the overlap-term weight for both alternatives, so changing
    only ``name`` cleanly switches Dice to Tversky without changing BCE weight.
    """
    c = dict(name="bce_dice", bce_weight=1.0, dice_weight=1.0,
             alpha=0.3, beta=0.7, smooth=1.0, pos_weight=None)
    unknown = set(cfg or {}) - set(c)
    if unknown:
        raise ValueError(f"Unknown loss settings: {sorted(unknown)}")
    c.update(cfg or {})
    return BinaryOverlapLoss(**c)
