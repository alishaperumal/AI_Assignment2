"""Small model factory and portable checkpoint loading for Person B.

Models return raw logits with shape (batch, 1, height, width). Apply sigmoid
exactly once, in the prediction code. Input images are already normalized by
the data pipeline.
"""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Mapping

import torch
from torch import nn


def build_model(cfg: Mapping | None = None) -> nn.Module:
    """Build the baseline U-Net or its U-Net++ decoder comparison.

    Both models deliberately use the same encoder and one grayscale channel,
    making the architecture experiment easier to interpret. Setting
    ``encoder_weights=None`` avoids a download, useful for tests and reloads.
    """
    import segmentation_models_pytorch as smp

    c = dict(architecture="unet", encoder_name="resnet34",
             encoder_weights="imagenet", in_channels=1, classes=1)
    unknown = set(cfg or {}) - set(c)
    if unknown:
        raise ValueError(f"Unknown model settings: {sorted(unknown)}")
    c.update(cfg or {})
    architectures = {"unet": smp.Unet, "unetplusplus": smp.UnetPlusPlus}
    architecture = str(c.pop("architecture")).lower()
    if architecture not in architectures:
        raise ValueError(f"Unknown architecture {architecture!r}; choose {tuple(architectures)}")
    if c["in_channels"] != 1 or c["classes"] != 1:
        raise ValueError("This project uses one grayscale input and one foreground output")
    return architectures[architecture](**c, activation=None)


def model_size(model: nn.Module) -> dict:
    """Parameter counts and tensor storage, separate from training checkpoint size."""
    return {
        "parameters": sum(p.numel() for p in model.parameters()),
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "state_tensor_bytes": sum(t.numel() * t.element_size() for t in model.state_dict().values()),
    }


def load_checkpoint_model(checkpoint: str | Path, device="cpu"):
    """Load a trusted project checkpoint without downloading pretrained weights.

    Returns ``(model_in_eval_mode, saved_full_config, checkpoint_dictionary)``.
    Training checkpoints include Python/NumPy random states, so this explicitly
    permits pickle loading. Only use checkpoints produced by this project or
    another source you trust.
    """
    state = torch.load(Path(checkpoint), map_location="cpu", weights_only=False)
    if not isinstance(state, dict) or not {"config", "model_state_dict"} <= state.keys():
        raise ValueError("Expected a project checkpoint containing config and model_state_dict")
    cfg = deepcopy(state["config"])
    model_cfg = deepcopy(cfg["model"])
    model_cfg["encoder_weights"] = None
    model = build_model(model_cfg)
    model.load_state_dict(state["model_state_dict"], strict=True)
    model.to(device).eval()
    return model, cfg, state
