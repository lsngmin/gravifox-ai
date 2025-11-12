"""Attention 기반 퓨전 계층."""

from __future__ import annotations

import torch
import torch.nn as nn

from core.utils.logger import get_logger
from .config import FusionConfig

LOGGER = get_logger(__name__)


class AttentionFusion(nn.Module):
    """Q=SigLIP, K/V=Residual 기반 어텐션 퓨전."""

    def __init__(self, cfg: FusionConfig):
        super().__init__()
        self.cfg = cfg
        self.query_proj = nn.Linear(cfg.siglip_dim, cfg.fused_dim)
        self.key_proj = nn.Linear(cfg.residual_dim, cfg.fused_dim)
        self.value_proj = nn.Linear(cfg.residual_dim, cfg.fused_dim)
        self.attn = nn.MultiheadAttention(
            cfg.fused_dim, cfg.num_heads, dropout=cfg.dropout, batch_first=True
        )
        self.output = nn.Sequential(
            nn.LayerNorm(cfg.fused_dim),
            nn.Linear(cfg.fused_dim, cfg.fused_dim),
            nn.GELU(),
        )

    def forward(
        self, siglip_embedding: torch.Tensor, residual_embedding: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """임베딩을 결합하여 512차원 특징을 생성한다."""

        q = self.query_proj(siglip_embedding).unsqueeze(1)
        k = self.key_proj(residual_embedding).unsqueeze(1)
        v = self.value_proj(residual_embedding).unsqueeze(1)
        fused, weights = self.attn(q, k, v, need_weights=True)
        fused = fused.squeeze(1)
        fused = self.output(fused)
        return fused, weights.squeeze(1)


__all__ = ["AttentionFusion"]
