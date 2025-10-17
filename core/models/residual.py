"""SNS 재압축 환경에서 강건한 잔차 브랜치 추출기와 간단 CNN 모델.

이 파일은 두 가지 용도를 모두 지원한다.
1) ResidualBranchExtractor: ViT 결합용 잔차 임베딩 추출기(기존 파이프라인 호환)
2) ResidualCNN: 단독 학습/검증을 위한 경량 CNN 임베더(Residual 모델)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn

from core.utils.logger import get_logger
from .registry import register


logger = get_logger(__name__)


class HighPassFilter(nn.Module):
    """라플라시안 기반 고역 필터.

    SNS에서 여러 번 리사이즈/압축된 이미지는 저주파 성분이 부드러워지고,
    합성된 흔적은 고주파 잔차에서 상대적으로 두드러진다. 따라서 학습 전에
    고역 통과 필터를 적용해 잡음을 강조한다.
    """

    def __init__(self, channels: int = 3):
        super().__init__()
        base_kernel = torch.tensor([
            [0.0, -1.0, 0.0],
            [-1.0, 4.0, -1.0],
            [0.0, -1.0, 0.0],
        ], dtype=torch.float32)
        kernel = base_kernel.view(1, 1, 3, 3).repeat(channels, 1, 1, 1).contiguous()
        self.register_buffer("weight", kernel)
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
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.proj = nn.Linear(64, cfg.embed_dim)
        self._shape_logged = False

    def forward(self, image_tensor: torch.Tensor) -> torch.Tensor:
        """입력 이미지를 잔차 임베딩으로 변환한다."""

        residual = self.high_pass(image_tensor)
        features = self.cnn(residual)
        if (
            not self._shape_logged
            and not torch.jit.is_tracing()
            and not torch.jit.is_scripting()
        ):
            logger.debug(
                "Residual 브랜치 텐서 크기 - residual=%s, features=%s",
                residual.shape,
                features.shape,
            )
            self._shape_logged = True
        flattened = features.flatten(1)
        embedding = self.proj(flattened)
        return embedding


class ResidualCNN(nn.Module):
    """간단한 2-Conv CNN으로 임베딩을 생성하는 경량 모델.

    무엇을/왜:
        기본 RGB 이미지를 입력으로 받아 두 번의 3x3 Conv와 ReLU를 거친 뒤
        AdaptiveAvgPool로 공간 평균을 내어 고정 길이 임베딩을 만듭니다.
        Residual 단독 모델 실험을 빠르게 수행하기 위한 용도입니다.

    Args:
        in_channels: 입력 채널 수(기본 3)
        embed_dim: 출력 임베딩 차원

    Returns:
        forward(x): [B, embed_dim] 텐서
    """

    def __init__(self, in_channels: int = 3, embed_dim: int = 128):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, embed_dim, kernel_size=3, stride=1, padding=1),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)  # [B, embed_dim, H, W]
        x = self.pool(x).view(x.size(0), -1)  # [B, embed_dim]
        return x


@register("residual")
def build_vit_residual(**cfg) -> ResidualCNN:
    """ResidualCNN 빌더 함수.

    무엇을/왜:
        YAML 또는 Trainer에서 전달된 설정으로 ResidualCNN을 생성합니다.

    Args:
        cfg: in_channels, embed_dim 등을 포함하는 설정 딕셔너리

    Returns:
        ResidualCNN 인스턴스
    """

    in_channels = int(cfg.get("in_channels", 3))
    embed_dim = int(cfg.get("embed_dim", 128))
    logger.debug("ResidualCNN 빌드 - in_channels=%d, embed_dim=%d", in_channels, embed_dim)
    return ResidualCNN(in_channels=in_channels, embed_dim=embed_dim)
