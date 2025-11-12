"""RKMv1 전체 모델 정의."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
from torch.autograd import Function

from core.utils.logger import get_logger
from ..registry import register
from .backbone import SiglipBackbone
from .classifier import BinaryClassifierHead
from .config import RKMv1Config, load_rkmv1_config
from .fusion import AttentionFusion
from .losses import RKMv1Loss
from .residual import ResidualHybridBranch

LOGGER = get_logger(__name__)


class GradientReversalFunction(Function):
    """Gradient Reversal 구현."""

    @staticmethod
    def forward(ctx, input_tensor: torch.Tensor, lambd: float) -> torch.Tensor:  # type: ignore[override]
        ctx.lambd = lambd
        return input_tensor.view_as(input_tensor)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):  # type: ignore[override]
        return grad_output.neg() * ctx.lambd, None


class GradientReversal(nn.Module):
    """도메인 역전 층."""

    def __init__(self, lambd: float):
        super().__init__()
        self.lambd = lambd

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return GradientReversalFunction.apply(x, self.lambd)


class RKMv1Model(nn.Module):
    """SigLIP 백본 + 잔차 하이브리드 퓨전 모델."""

    def __init__(self, cfg: RKMv1Config):
        super().__init__()
        self.cfg = cfg
        self.model_version = cfg.model_version

        self.backbone = SiglipBackbone(cfg.backbone)
        self.residual = ResidualHybridBranch(cfg.residual)
        self.fusion = AttentionFusion(cfg.fusion)
        classifier_cfg = cfg.classifier
        classifier_cfg.input_dim = cfg.fusion.fused_dim
        self.classifier = BinaryClassifierHead(classifier_cfg)

        self.grl = GradientReversal(cfg.loss.domain_grl_lambda)
        self.domain_head = nn.Sequential(
            nn.LayerNorm(cfg.fusion.fused_dim),
            nn.Linear(cfg.fusion.fused_dim, cfg.fusion.fused_dim // 2),
            nn.GELU(),
            nn.Linear(cfg.fusion.fused_dim // 2, cfg.loss.domain_classes),
        )
        self.loss_fn = RKMv1Loss(cfg.loss)
        LOGGER.info("RKMv1 모델 초기화 - cfg=%s", asdict(cfg))

    def forward(
        self,
        images: torch.Tensor,
        *,
        domain_labels: Optional[torch.Tensor] = None,
        return_dict: bool = False,
    ) -> torch.Tensor | Dict[str, torch.Tensor]:
        """추론을 수행하고 필요 시 중간 특징을 함께 반환한다."""

        siglip_embedding = self.backbone(images)
        residual_embedding = self.residual(images)
        fused_embedding, attn_weights = self.fusion(
            siglip_embedding, residual_embedding
        )
        logits = self.classifier(fused_embedding)

        domain_logits = self.domain_head(self.grl(fused_embedding))

        if not return_dict:
            return logits

        return {
            "logits": logits,
            "siglip": siglip_embedding,
            "residual": residual_embedding,
            "fused": fused_embedding,
            "attention": attn_weights,
            "domain_logits": domain_logits,
            "domain_labels": domain_labels,
        }

    def compute_loss(
        self,
        images: torch.Tensor,
        targets: Optional[torch.Tensor],
        *,
        domain_labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """복합 손실을 계산한다."""

        outputs = self.forward(
            images, domain_labels=domain_labels, return_dict=True
        )
        loss_dict = self.loss_fn(
            outputs["logits"],
            targets,
            outputs["fused"],
            outputs["domain_logits"],
            domain_labels,
        )
        return loss_dict


@register("rkmv1")
def build_rkmv1(**kwargs: Any) -> RKMv1Model:
    """Hydra 레지스트리용 빌더."""

    config = load_rkmv1_config(kwargs)
    return RKMv1Model(config)


__all__ = ["RKMv1Model", "build_rkmv1"]
