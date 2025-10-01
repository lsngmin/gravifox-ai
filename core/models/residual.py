"""SNS 재압축 환경에서 강건한 잔차 브랜치 추출기."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn

from core.utils.logger import get_logger


logger = get_logger(__name__)


class HighPassFilter(nn.Module):
    """라플라시안 기반 고역 필터.

    SNS에서 여러 번 리사이즈/압축된 이미지는 저주파 성분이 부드러워지고,
    합성된 흔적은 고주파 잔차에서 상대적으로 두드러진다. 따라서 학습 전에
    고역 통과 필터를 적용해 잡음을 강조한다.
    """

    def __init__(self, channels: int = 3):
        super().__init__()
        kernel = torch.tensor([
            [0.0, -1.0, 0.0],
            [-1.0, 4.0, -1.0],
            [0.0, -1.0, 0.0],
        ], dtype=torch.float32)
        weight = kernel.expand(channels, 1, 3, 3)
        self.register_buffer("weight", weight)
        self.groups = channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.conv2d(x, self.weight, groups=self.groups, padding=1)


@dataclass
class ResidualBranchConfig:
    """ResidualBranchExtractor 설정."""

    input_channels: int = 3
    embed_dim: int = 128


class ResidualBranchExtractor(nn.Module):
    """고역 정보를 활용해 잔차 임베딩을 추출한다."""

    def __init__(self, cfg: ResidualBranchConfig):
        super().__init__()
        self.cfg = cfg
        self.high_pass = HighPassFilter(cfg.input_channels)
        self.cnn = nn.Sequential(
            nn.Conv2d(cfg.input_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.proj = nn.Linear(64, cfg.embed_dim)
        self._shape_logged = False

    def forward(self, image_tensor: torch.Tensor) -> torch.Tensor:
        """입력 이미지를 잔차 임베딩으로 변환한다."""

        residual = self.high_pass(image_tensor)
        features = self.cnn(residual)
        if not self._shape_logged:
            logger.debug("Residual 브랜치 텐서 크기 - residual=%s, features=%s", residual.shape, features.shape)
            self._shape_logged = True
        flattened = features.flatten(1)
        embedding = self.proj(flattened)
        return embedding

