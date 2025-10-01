"""RGB 특징과 잔차 임베딩을 결합하는 Late Fusion 헤드."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from core.utils.logger import get_logger


logger = get_logger(__name__)


@dataclass
class FusionHeadConfig:
    """FusionClassifierHead 구성 값."""

    dim_rgb: int
    dim_residual: int
    dim_quality: int = 0
    num_classes: int = 2


class FusionClassifierHead(nn.Module):
    """여러 특징을 융합해 최종 분류 결과를 생성한다."""

    def __init__(self, cfg: FusionHeadConfig):
        super().__init__()
        self.cfg = cfg
        fused_dim = cfg.dim_rgb + cfg.dim_residual + cfg.dim_quality
        self.mlp = nn.Sequential(
            nn.LayerNorm(fused_dim),
            nn.Linear(fused_dim, fused_dim // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(fused_dim // 2, cfg.num_classes),
        )
        logger.debug(
            "Fusion 헤드 초기화 - dim_rgb=%d, dim_residual=%d, dim_quality=%d",
            cfg.dim_rgb,
            cfg.dim_residual,
            cfg.dim_quality,
        )

    def forward(self, rgb_features: torch.Tensor, residual_embedding: torch.Tensor, quality_features: torch.Tensor | None = None) -> torch.Tensor:
        """RGB/잔차/품질 특징을 결합하여 로짓을 반환한다."""

        parts = [rgb_features, residual_embedding]
        if quality_features is not None:
            parts.append(quality_features)
        fused = torch.cat(parts, dim=1)
        logits = self.mlp(fused)
        return logits

