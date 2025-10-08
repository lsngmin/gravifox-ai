"""ViT + Residual 브랜치를 결합한 최종 분류 모델."""

from __future__ import annotations

from dataclasses import dataclass

import timm
import torch
import torch.nn as nn

from core.utils.logger import get_logger
from .fusion_head import FusionClassifierHead, FusionHeadConfig
from .residual import ResidualBranchConfig, ResidualBranchExtractor
from .registry import register


logger = get_logger(__name__)


@dataclass
class ViTResidualConfig:
    """ViTResidualFusionModel 설정."""

    vit_name: str = "vit_base_patch16_224"
    num_classes: int = 2
    pretrained: bool = True
    residual_embed_dim: int = 128
    model_version: str = "vit_residual_mvp_v1"


class ViTResidualFusionModel(nn.Module):
    """ViT 특징과 잔차 임베딩을 Late Fusion으로 결합한다."""

    def __init__(self, cfg: ViTResidualConfig):
        super().__init__()
        self.cfg = cfg
        self.model_version = cfg.model_version

        self.vit = timm.create_model(cfg.vit_name, pretrained=cfg.pretrained, num_classes=0)
        embed_dim = self.vit.num_features
        self.residual_branch = ResidualBranchExtractor(ResidualBranchConfig(embed_dim=cfg.residual_embed_dim))
        self.fusion_head = FusionClassifierHead(
            FusionHeadConfig(dim_rgb=embed_dim, dim_residual=cfg.residual_embed_dim, num_classes=cfg.num_classes)
        )
        self._shape_logged = False

    def forward(self, image_tensor: torch.Tensor) -> torch.Tensor:
        """이미지를 입력받아 2-클래스 로짓을 출력한다."""

        if image_tensor.ndim != 4 or image_tensor.shape[1] != 3:
            logger.error("입력 텐서가 [B,3,H,W] 형식이 아닙니다: %s", image_tensor.shape)  # TODO: 필요 없을 시 삭제 가능
            raise ValueError("Expected image tensor of shape [B,3,H,W]")

        vit_features = self.vit.forward_features(image_tensor)
        if isinstance(vit_features, (tuple, list)):
            vit_features = vit_features[0]
        vit_embedding = self.vit.forward_head(vit_features, pre_logits=True)

        residual_embedding = self.residual_branch(image_tensor)

        if not self._shape_logged:
            logger.debug(
                "Fusion 텐서 크기 - vit=%s, residual=%s",
                vit_embedding.shape,
                residual_embedding.shape,
            )
            self._shape_logged = True

        logits = self.fusion_head(vit_embedding, residual_embedding)
        return logits


@register("vit_residual")
def build_vit_residual(**kwargs) -> ViTResidualFusionModel:
    """레지스트리 빌더."""

    cfg = ViTResidualConfig(**kwargs)
    return ViTResidualFusionModel(cfg)


@register("vit_residual_fusion")
def build_vit_residual_fusion(**kwargs) -> ViTResidualFusionModel:
    """별칭 빌더 (Hydra config 호환)."""

    return build_vit_residual(**kwargs)
