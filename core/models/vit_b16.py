"""ViT-B/16 백본 모델 정의 모듈."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import timm
import torch
import torch.nn as nn

from core.utils.logger import get_logger
from .registry import register


logger = get_logger(__name__)


@dataclass
class ViTBackboneConfig:
    """ViT 백본 초기화에 필요한 하이퍼파라미터 묶음."""

    vit_name: str = "vit_base_patch16_224"
    num_classes: int = 2
    pretrained: bool = True
    drop_rate: float = 0.0
    drop_path_rate: float = 0.1
    return_features: bool = False


class ViTBackbone(nn.Module):
    """timm ViT-B/16 백본 래퍼.

    ViT는 전역 패치 토큰으로 이미지를 인코딩하기 때문에 SNS 재압축에 노출된
    이미지에서도 비교적 강건한 표현을 만든다. `return_features=True`로 설정하면
    분류기 대신 전역 풀링된 특징을 반환하여 다른 헤드와 결합할 수 있다.
    """

    def __init__(self, cfg: ViTBackboneConfig):
        super().__init__()
        self.cfg = cfg
        head_classes = cfg.num_classes if not cfg.return_features else 0
        self.backbone = timm.create_model(
            cfg.vit_name,
            pretrained=cfg.pretrained,
            num_classes=head_classes,
            drop_rate=cfg.drop_rate,
            drop_path_rate=cfg.drop_path_rate,
        )
        logger.debug("ViT 백본 초기화 - name=%s, return_features=%s", cfg.vit_name, cfg.return_features)

    def forward(self, image_tensor: torch.Tensor) -> torch.Tensor:
        """이미지를 입력받아 logits 또는 특징 텐서를 반환한다."""

        if self.cfg.return_features:
            features = self.backbone.forward_features(image_tensor)
            if isinstance(features, (tuple, list)):
                features = features[0]
            pooled = self.backbone.forward_head(features, pre_logits=True)
            return pooled
        return self.backbone(image_tensor)


@register("vit_b16")
def build_vit_b16(**kwargs) -> ViTBackbone:
    """레지스트리용 빌더 함수."""

    cfg = ViTBackboneConfig(**kwargs)
    return ViTBackbone(cfg)

