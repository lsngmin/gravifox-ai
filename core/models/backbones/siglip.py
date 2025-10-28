"""SigLIP vision backbone integration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn

from core.utils.logger import get_logger
from ..registry import register

logger = get_logger(__name__)

try:  # pragma: no cover - optional dependency
    from transformers import SiglipImageProcessor, SiglipVisionModel
except ImportError:  # pragma: no cover - handled at runtime
    SiglipImageProcessor = None  # type: ignore[assignment]
    SiglipVisionModel = None  # type: ignore[assignment]


@dataclass
class SiglipBackboneConfig:
    """Configuration container for the SigLIP vision backbone."""

    model_name: str = "google/siglip-large-patch16-384"
    num_classes: int = 2
    cache_dir: Optional[str] = None
    revision: Optional[str] = None
    classifier_dropout: float = 0.0
    return_features: bool = False
    normalize_input: bool = True
    freeze_vision_encoder: bool = False
    gradient_checkpointing: bool = False
    return_hidden_states: bool = False
    autocast_enabled: bool = False
    autocast_dtype: str = "float16"


class SiglipBackbone(nn.Module):
    """Wrapper around the Hugging Face SigLIP vision encoder.

    The module can either expose pooled feature vectors or attach a classification
    head, depending on ``return_features``. Inputs are expected to be batched image
    tensors shaped ``[B, 3, H, W]`` already resized to the resolution required by
    ``model_name`` (defaults to ``384``). When ``normalize_input`` is enabled, the
    backbone internally applies the mean/std from the official image processor so
    the upstream pipeline can pass images in the ``[0, 1]`` range without additional
    preprocessing.
    """

    def __init__(self, cfg: SiglipBackboneConfig):
        super().__init__()
        self.cfg = cfg

        if SiglipVisionModel is None:  # pragma: no cover - dependency guard
            raise ImportError(
                "transformers is required to use the SigLIP backbone. "
                "Please install transformers>=4.39."
            )

        model_kwargs = {}
        if cfg.cache_dir is not None:
            model_kwargs["cache_dir"] = cfg.cache_dir
        if cfg.revision is not None:
            model_kwargs["revision"] = cfg.revision

        logger.info("Loading SigLIP backbone: %s", cfg.model_name)
        self.vision = SiglipVisionModel.from_pretrained(cfg.model_name, **model_kwargs)

        if cfg.gradient_checkpointing:
            if hasattr(self.vision, "gradient_checkpointing_enable"):
                self.vision.gradient_checkpointing_enable()
                logger.info("Enabled gradient checkpointing for SigLIP vision encoder.")
            else:
                logger.warning("Gradient checkpointing is not supported by this transformers release.")

        vision_cfg = getattr(self.vision.config, "vision_config", self.vision.config)
        self.hidden_size = int(getattr(vision_cfg, "hidden_size"))
        self._autocast_dtype = self._resolve_autocast_dtype(cfg.autocast_dtype)

        self._image_mean: Optional[torch.Tensor]
        self._image_std: Optional[torch.Tensor]
        if cfg.normalize_input:
            if SiglipImageProcessor is None:  # pragma: no cover - dependency guard
                raise ImportError("SiglipImageProcessor requires transformers>=4.39.")
            processor = SiglipImageProcessor.from_pretrained(cfg.model_name, **model_kwargs)
            mean = torch.tensor(processor.image_mean, dtype=torch.float32).view(1, -1, 1, 1)
            std = torch.tensor(processor.image_std, dtype=torch.float32).view(1, -1, 1, 1)
            self.register_buffer("_image_mean", mean, persistent=False)
            self.register_buffer("_image_std", std, persistent=False)
            logger.debug(
                "SigLIP normalization configured - mean=%s std=%s",
                processor.image_mean,
                processor.image_std,
            )
        else:
            self._image_mean = None
            self._image_std = None

        if cfg.freeze_vision_encoder:
            for param in self.vision.parameters():
                param.requires_grad = False
            logger.info("SigLIP vision encoder parameters were frozen.")

        self.classifier: Optional[nn.Module]
        if cfg.return_features:
            self.classifier = None
        else:
            head: nn.Module
            if cfg.classifier_dropout > 0.0:
                head = nn.Sequential(
                    nn.Dropout(cfg.classifier_dropout),
                    nn.Linear(self.hidden_size, cfg.num_classes),
                )
            else:
                head = nn.Linear(self.hidden_size, cfg.num_classes)
            self.classifier = head
            self._init_classifier(self.classifier)
            if not any(p.requires_grad for p in self.classifier.parameters()):
                raise AssertionError("Classifier must have trainable params.")

    @staticmethod
    def _resolve_autocast_dtype(dtype_value: object) -> torch.dtype:
        if isinstance(dtype_value, torch.dtype):
            return dtype_value
        if isinstance(dtype_value, str):
            mapping = {
                "float16": torch.float16,
                "fp16": torch.float16,
                "half": torch.float16,
                "bfloat16": torch.bfloat16,
                "bf16": torch.bfloat16,
                "float32": torch.float32,
                "fp32": torch.float32,
            }
            key = dtype_value.lower()
            if key in mapping:
                return mapping[key]
        raise ValueError(f"Unsupported autocast dtype: {dtype_value}")

    @staticmethod
    def _init_classifier(module: nn.Module) -> None:
        for sub in module.modules():
            if isinstance(sub, nn.Linear):
                nn.init.xavier_uniform_(sub.weight)
                if sub.bias is not None:
                    nn.init.zeros_(sub.bias)

    @dataclass
    class ForwardOutput:
        logits: Optional[torch.Tensor]
        pooled: torch.Tensor
        hidden_states: Optional[Tuple[torch.Tensor, ...]] = None

    def forward(self, image_tensor: torch.Tensor) -> torch.Tensor | "SiglipBackbone.ForwardOutput":
        """Forward images through SigLIP and return features or logits."""

        if self.cfg.normalize_input:
            assert self._image_mean is not None and self._image_std is not None
            pixel_values = (image_tensor - self._image_mean.to(image_tensor.device)) / self._image_std.to(
                image_tensor.device
            )
        else:
            pixel_values = image_tensor

        device_type = pixel_values.device.type
        use_autocast = self.cfg.autocast_enabled and device_type == "cuda"
        autocast_scope = torch.amp.autocast if device_type != "cpu" else torch.amp.autocast
        with autocast_scope(device_type, dtype=self._autocast_dtype, enabled=use_autocast):
            outputs = self.vision(
                pixel_values=pixel_values,
                output_hidden_states=self.cfg.return_hidden_states,
            )

        pooled = getattr(outputs, "pooler_output", None)
        if pooled is None:
            pooled = outputs.last_hidden_state[:, 0]

        hidden_states = getattr(outputs, "hidden_states", None) if self.cfg.return_hidden_states else None

        if self.cfg.return_features:
            if self.cfg.return_hidden_states:
                return self.ForwardOutput(logits=None, pooled=pooled, hidden_states=hidden_states)  # type: ignore[return-value]
            return pooled

        logits = self.classifier(pooled)
        if self.cfg.return_hidden_states:
            return self.ForwardOutput(logits=logits, pooled=pooled, hidden_states=hidden_states)  # type: ignore[return-value]
        return logits


@register("siglip_large_patch16_384")
def build_siglip_large_patch16_384(**kwargs) -> SiglipBackbone:
    """Factory registered under ``siglip_large_patch16_384``."""

    cfg = SiglipBackboneConfig(**kwargs)
    return SiglipBackbone(cfg)
