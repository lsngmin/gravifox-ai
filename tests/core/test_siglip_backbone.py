from __future__ import annotations

import pathlib
import sys
from typing import Dict

import pytest
import torch

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.models.backbones.siglip import SiglipBackbone, SiglipBackboneConfig

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="SigLIP tests require CUDA and real checkpoints.")


@pytest.fixture(scope="module")
def device() -> torch.device:
    return torch.device("cuda")


def _capture_pixel_values(model: SiglipBackbone, store: Dict[str, torch.Tensor]) -> torch.utils.hooks.RemovableHandle:
    def _hook(_, __, kwargs):
        store["pixel_values"] = kwargs["pixel_values"].detach().cpu()

    return model.vision.register_forward_pre_hook(_hook, with_kwargs=True)


@pytest.mark.parametrize("freeze_encoder", [True, False])
def test_siglip_backbone_logits_forward(device: torch.device, freeze_encoder: bool) -> None:
    cfg = SiglipBackboneConfig(
        num_classes=4,
        freeze_vision_encoder=freeze_encoder,
        gradient_checkpointing=True,
    )
    model = SiglipBackbone(cfg).to(device).eval()

    captured: Dict[str, torch.Tensor] = {}
    handle = _capture_pixel_values(model, captured)

    dummy = torch.ones(2, 3, 384, 384, device=device)
    with torch.no_grad():
        logits = model(dummy)

    handle.remove()

    assert logits.shape == (2, cfg.num_classes)
    assert "pixel_values" in captured

    expected = torch.full((2, 3, 384, 384), (1.0 - 0.5) / 0.25)
    torch.testing.assert_close(captured["pixel_values"], expected, atol=1e-5, rtol=1e-5)

    if freeze_encoder:
        assert all(not p.requires_grad for p in model.vision.parameters())
    else:
        assert any(p.requires_grad for p in model.vision.parameters())

    assert bool(getattr(model.vision, "gradient_checkpointing", True))


def test_siglip_backbone_feature_output(device: torch.device) -> None:
    cfg = SiglipBackboneConfig(
        return_features=True,
        return_hidden_states=True,
        autocast_enabled=True,
    )
    model = SiglipBackbone(cfg).to(device).eval()

    dummy = torch.zeros(1, 3, 384, 384, device=device)
    with torch.no_grad():
        output = model(dummy)

    assert isinstance(output, SiglipBackbone.ForwardOutput)
    assert output.logits is None
    assert output.pooled.shape == (1, model.hidden_size)
    assert output.pooled.device.type == "cuda"

    assert output.hidden_states is not None
    assert len(output.hidden_states) > 0
    last_hidden = output.hidden_states[-1]
    assert last_hidden.shape[0] == 1
    assert last_hidden.shape[-1] == model.hidden_size
    assert last_hidden.device.type == "cuda"
