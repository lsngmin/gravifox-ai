"""Laplacian + FFT 하이브리드 잔차 브랜치."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from core.utils.logger import get_logger
from .config import ResidualConfig

LOGGER = get_logger(__name__)


class LaplacianFilter(nn.Module):
    """라플라시안 고역 필터."""

    def __init__(self, channels: int = 3):
        super().__init__()
        kernel = torch.tensor(
            [[0.0, -1.0, 0.0], [-1.0, 4.0, -1.0], [0.0, -1.0, 0.0]], dtype=torch.float32
        )
        kernel = kernel.view(1, 1, 3, 3).repeat(channels, 1, 1, 1).contiguous()
        self.register_buffer("weight", kernel)
        self.groups = channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """고역 통과 필터를 적용한다."""

        return F.conv2d(x, self.weight, padding=1, groups=self.groups)


class ResidualHybridBranch(nn.Module):
    """라플라시안 + FFT 하이브리드 임베더."""

    def __init__(self, cfg: ResidualConfig):
        super().__init__()
        self.cfg = cfg
        self.laplacian = LaplacianFilter(cfg.input_channels)
        self.spatial = nn.Sequential(
            nn.Conv2d(cfg.input_channels, 64, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, 64),
            nn.GELU(),
            nn.Conv2d(64, 96, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(12, 96),
            nn.GELU(),
            nn.Conv2d(96, 128, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(16, 128),
            nn.GELU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.spatial_proj = nn.Linear(128, cfg.spatial_dim)

        self.spectral_pool = nn.AdaptiveAvgPool2d((16, 16))
        self.spectral_proj = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16 * 16 * cfg.input_channels, cfg.spectral_dim),
            nn.GELU(),
        )
        self.output = nn.Linear(cfg.spatial_dim + cfg.spectral_dim, cfg.embed_dim)
        self._shape_logged = False

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """입력을 잔차 임베딩으로 변환한다."""

        original_dtype = images.dtype
        residual_input = images if images.dtype == torch.float32 else images.float()
        residual = self.laplacian(residual_input)
        spatial_feat = self._spatial_features(residual)
        spectral_feat = self._spectral_features(residual)
        embedding = torch.cat([spatial_feat, spectral_feat], dim=1)
        embedding = torch.nan_to_num(embedding, nan=0.0, posinf=1.0e4, neginf=-1.0e4)
        embedding = self.output(embedding)
        embedding = torch.nan_to_num(embedding, nan=0.0, posinf=1.0e4, neginf=-1.0e4)
        if (
            not self._shape_logged
            and not torch.jit.is_tracing()
            and not torch.jit.is_scripting()
        ):
            LOGGER.debug(
                "ResidualHybrid 텐서 크기 - spatial=%s spectral=%s embed=%s",
                spatial_feat.shape,
                spectral_feat.shape,
                embedding.shape,
            )
            self._shape_logged = True
        return embedding.to(original_dtype)

    def _spatial_features(self, x: torch.Tensor) -> torch.Tensor:
        """CNN 기반 공간 특징을 계산한다."""

        feats = self.spatial(x).flatten(1)
        return self.spatial_proj(feats)

    def _spectral_features(self, x: torch.Tensor) -> torch.Tensor:
        """FFT 기반 스펙트럼 특징을 계산한다."""

        freq = torch.fft.rfft2(x, norm="ortho")
        magnitude = torch.abs(freq)
        magnitude = torch.nan_to_num(magnitude, nan=0.0, posinf=1.0e4, neginf=0.0)
        pooled = self.spectral_pool(magnitude)
        if self.cfg.fft_pool == "max":
            pooled, _ = torch.max(pooled, dim=1, keepdim=True)
        else:
            pooled = pooled.mean(dim=1, keepdim=True)
        pooled = pooled.repeat(1, self.cfg.input_channels, 1, 1)
        return self.spectral_proj(pooled)


__all__ = ["ResidualHybridBranch"]
