"""RKMv1 분류 헤드."""

from __future__ import annotations

import torch.nn as nn

from .config import ClassifierConfig

class BinaryClassifierHead(nn.Module):
    """512 → 2 계층 MLP."""

    def __init__(self, cfg: ClassifierConfig):
        super().__init__()
        self.cfg = cfg
        self.net = nn.Sequential(
            nn.LayerNorm(cfg.input_dim),
            nn.Linear(cfg.input_dim, cfg.hidden_dim),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden_dim, cfg.num_classes),
        )

    def forward(self, fused: torch.Tensor) -> torch.Tensor:
        """512차원 임베딩을 2-클래스 로짓으로 변환한다."""

        return self.net(fused)


__all__ = ["BinaryClassifierHead"]
