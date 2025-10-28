import types

import torch
import pytest

from core.models.backbones.siglip import SiglipBackbone, SiglipBackboneConfig


def _install_siglip_stubs(monkeypatch, *, hidden_size=32, pooler_output=True):
    """Install lightweight SigLIP stubs to avoid downloading real weights."""

    captured = {}

    class DummyVisionModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            vision_config = types.SimpleNamespace(hidden_size=hidden_size)
            self.config = types.SimpleNamespace(vision_config=vision_config)
            self.dummy_weight = torch.nn.Parameter(torch.ones(1))
            self.gradient_checkpointing_enabled = False

        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            return cls()

        def gradient_checkpointing_enable(self):
            self.gradient_checkpointing_enabled = True

        def parameters(self, recurse=True):
            return [self.dummy_weight]

        def forward(self, pixel_values, output_hidden_states=False):
            captured["pixel_values"] = pixel_values
            batch = pixel_values.shape[0]
            token_dim = 2
            hidden = torch.zeros(batch, token_dim, hidden_size, device=pixel_values.device)
            pooler = torch.ones(batch, hidden_size, device=pixel_values.device) if pooler_output else None
            hidden_states = (hidden,) if output_hidden_states else None
            return types.SimpleNamespace(
                pooler_output=pooler,
                last_hidden_state=hidden,
                hidden_states=hidden_states,
            )

        __call__ = forward

    class DummyProcessor:
        image_mean = [0.5, 0.5, 0.5]
        image_std = [0.25, 0.25, 0.25]

        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            return cls()

    target = "core.models.backbones.siglip"
    monkeypatch.setattr(f"{target}.SiglipVisionModel", DummyVisionModel, raising=False)
    monkeypatch.setattr(f"{target}.SiglipImageProcessor", DummyProcessor, raising=False)

    return captured, DummyVisionModel


@pytest.mark.parametrize("freeze_encoder", [True, False])
def test_siglip_backbone_logits_forward(monkeypatch, freeze_encoder):
    captured, DummyVisionModel = _install_siglip_stubs(monkeypatch, hidden_size=64, pooler_output=True)

    cfg = SiglipBackboneConfig(
        num_classes=4,
        freeze_vision_encoder=freeze_encoder,
        gradient_checkpointing=True,
    )
    model = SiglipBackbone(cfg)
    assert model.hidden_size == 64

    dummy = torch.ones(2, 3, 384, 384)
    logits = model(dummy)
    assert logits.shape == (2, 4)

    assert "pixel_values" in captured
    expected = torch.full_like(dummy, (1.0 - 0.5) / 0.25)
    torch.testing.assert_close(captured["pixel_values"], expected)

    if freeze_encoder:
        assert all(not p.requires_grad for p in model.vision.parameters())
    else:
        assert any(p.requires_grad for p in model.vision.parameters())
    assert model.vision.gradient_checkpointing_enabled is True


@pytest.mark.skipif(not torch.cuda.is_available(), reason="SigLIP autocast requires CUDA.")
def test_siglip_backbone_feature_output(monkeypatch):
    captured, DummyVisionModel = _install_siglip_stubs(monkeypatch, hidden_size=48, pooler_output=False)

    cfg = SiglipBackboneConfig(
        return_features=True,
        return_hidden_states=True,
        normalize_input=False,
        autocast_enabled=True,
    )
    model = SiglipBackbone(cfg)
    assert model.classifier is None

    dummy = torch.zeros(1, 3, 384, 384, device="cuda")
    output = model(dummy)

    assert isinstance(output, SiglipBackbone.ForwardOutput)
    assert output.logits is None
    assert output.pooled.shape == (1, 48)
    assert output.hidden_states is not None
    assert len(output.hidden_states) == 1
    assert output.hidden_states[0].shape == (1, 2, 48)

    captured_tensor = captured["pixel_values"]
    assert captured_tensor.device.type == "cuda"
    assert captured_tensor.shape == dummy.shape
