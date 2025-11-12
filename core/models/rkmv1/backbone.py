"""SigLIP 백본 모듈."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import timm
except ImportError as exc:  # pragma: no cover
    timm = None  # type: ignore[assignment]
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None

from core.utils.logger import get_logger
from .config import BackboneConfig

LOGGER = get_logger(__name__)


class SiglipBackbone(nn.Module):
    """SigLIP-Large-Patch16-384 백본 래퍼."""

    def __init__(self, cfg: BackboneConfig):
        super().__init__()
        if timm is None:
            raise ImportError(
                "timm 패키지를 찾을 수 없습니다. SigLIP 백본을 로드하려면 timm>=0.9가 필요합니다."
            ) from _IMPORT_ERROR

        self.cfg = cfg
        self.target_size = cfg.target_size
        self.image_size = cfg.image_size
        self.siglip = timm.create_model(
            cfg.model_name,
            pretrained=cfg.pretrained,
            num_classes=0,
        )
        self.output_dim = getattr(self.siglip, "num_features", cfg.output_dim)
        if cfg.freeze_backbone:
            self.siglip.requires_grad_(False)
        LOGGER.info(
            "SigLIP 백본 초기화 - model=%s pretrained=%s frozen=%s",
            cfg.model_name,
            cfg.pretrained,
            cfg.freeze_backbone,
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """입력 이미지를 SigLIP 임베딩으로 변환한다."""

        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError(f"[B,3,H,W] 텐서가 필요합니다: {tuple(images.shape)}")
        resized = self._match_resolution(images)
        feats = self.siglip.forward_features(resized)
        if isinstance(feats, (tuple, list)):
            feats = feats[0]
        embedding = self.siglip.forward_head(feats, pre_logits=True)
        return embedding

    def _match_resolution(self, images: torch.Tensor) -> torch.Tensor:
        """SigLIP이 요구하는 384x384 해상도로 맞춘다."""

        if images.shape[-1] == self.target_size and images.shape[-2] == self.target_size:
            return images
        resized = F.interpolate(
            images,
            size=(self.target_size, self.target_size),
            mode="bicubic",
            align_corners=False,
        )
        return resized


__all__ = ["SiglipBackbone"]
