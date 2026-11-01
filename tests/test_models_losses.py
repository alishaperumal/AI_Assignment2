"""Checks for the chosen model interfaces and sparse-mask loss behavior."""
from copy import deepcopy

import pytest
import torch

from losses import build_loss
import models


@pytest.mark.parametrize("name", ["bce_dice", "bce_tversky"])
@pytest.mark.parametrize("foreground", [False, True])
def test_overlap_losses_have_finite_gradients_and_reward_correct_predictions(name, foreground):
    target = torch.zeros(2, 1, 16, 16)
    if foreground:
        target[:, :, 4:12, 7:9] = 1
    uncertain = torch.zeros_like(target, requires_grad=True)
    criterion = build_loss({"name": name})
    loss = criterion(uncertain, target)
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(uncertain.grad).all()
    confident = torch.where(target > 0, 8.0, -8.0)
    assert criterion(confident, target) < loss


def test_tversky_can_emphasize_missed_foreground():
    target = torch.zeros(1, 1, 4, 4)
    target[..., :2, :] = 1
    # Same number of errors in either case, but one misses positive pixels.
    missed = torch.where(target > 0, 8.0, -8.0)
    added = missed.clone()
    missed[..., 0, :] = -8.0
    added[..., 3, :] = 8.0
    criterion = build_loss({"name": "bce_tversky", "bce_weight": 0,
                            "alpha": 0.3, "beta": 0.7})
    assert criterion(missed, target) > criterion(added, target)


@pytest.mark.parametrize("architecture", ["unet", "unetplusplus"])
def test_model_returns_one_logit_per_input_pixel(architecture):
    pytest.importorskip("segmentation_models_pytorch")
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        model = models.build_model({"architecture": architecture, "encoder_name": "resnet18",
                                    "encoder_weights": None}).eval()
        with torch.no_grad():
            result = model(torch.zeros(1, 1, 64, 64))
        assert result.shape == (1, 1, 64, 64)
        assert torch.isfinite(result).all()
    finally:
        torch.set_num_threads(previous_threads)


def test_checkpoint_reload_uses_saved_config_without_pretrained_download(tmp_path, monkeypatch):
    configs_seen = []

    def factory(cfg):
        configs_seen.append(deepcopy(cfg))
        return torch.nn.Conv2d(1, 1, 1)

    monkeypatch.setattr(models, "build_model", factory)
    original = factory({}).eval()
    checkpoint = tmp_path / "model.pt"
    cfg = {"model": {"architecture": "unet", "encoder_weights": "imagenet"}}
    torch.save({"config": cfg, "model_state_dict": original.state_dict()}, checkpoint)
    restored, saved_cfg, _ = models.load_checkpoint_model(checkpoint)
    x = torch.randn(1, 1, 8, 8)
    assert torch.equal(original(x), restored(x))
    assert configs_seen[-1]["encoder_weights"] is None
    assert saved_cfg == cfg and not restored.training


def test_loss_rejects_shape_errors_and_unknown_settings():
    with pytest.raises(ValueError, match="shape"):
        build_loss()(torch.zeros(1, 1, 4, 4), torch.zeros(1, 4, 4))
    with pytest.raises(ValueError, match="Unknown"):
        build_loss({"dice_weigth": 1})
