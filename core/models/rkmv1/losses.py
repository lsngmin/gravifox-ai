"""RKMv1 복합 손실."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import LossConfig


class TripletSelector:
    """간단한 batch-hard triplet 선택기."""

    def __init__(self, margin: float):
        self.margin = margin

    def __call__(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """배치 내에서 hardest triplet을 찾아 손실을 계산한다."""

        if embeddings.size(0) < 2:
            return embeddings.new_tensor(0.0)
        work_embeddings = embeddings if embeddings.dtype == torch.float32 else embeddings.float()
        distance = torch.cdist(work_embeddings, work_embeddings, p=2)
        distance = torch.nan_to_num(distance, nan=0.0, posinf=1.0e6, neginf=0.0)
        loss = embeddings.new_tensor(0.0)
        triplet_count = 0
        unique_labels = torch.unique(labels)
        for label in unique_labels:
            mask_pos = labels == label
            mask_neg = labels != label
            if mask_pos.sum() < 2 or mask_neg.sum() == 0:
                continue
            pos_dist = distance[mask_pos][:, mask_pos]
            neg_dist = distance[mask_pos][:, mask_neg]
            hardest_pos = pos_dist.max(dim=1).values
            hardest_neg = neg_dist.min(dim=1).values
            batch_loss = F.relu(hardest_pos - hardest_neg + self.margin)
            loss = loss + batch_loss.sum()
            triplet_count += batch_loss.numel()
        if triplet_count == 0:
            return embeddings.new_tensor(0.0)
        return loss / triplet_count


class DomainAdversarialLoss(nn.Module):
    """도메인 분류 손실."""

    def __init__(self, num_domains: int):
        super().__init__()
        self.criterion = nn.CrossEntropyLoss()
        self.num_domains = num_domains

    def forward(
        self, logits: torch.Tensor, labels: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """도메인 분류 손실을 계산한다."""

        if labels is None:
            return logits.new_tensor(0.0)
        return self.criterion(logits, labels.long())


class RKMv1Loss(nn.Module):
    """BCE + Triplet + Domain-Adversarial 복합 손실."""

    def __init__(self, cfg: LossConfig):
        super().__init__()
        self.cfg = cfg
        self.bce = nn.BCEWithLogitsLoss()
        self.triplet = TripletSelector(cfg.triplet_margin)
        self.domain = DomainAdversarialLoss(cfg.domain_classes)

    def forward(
        self,
        logits: torch.Tensor,
        targets: Optional[torch.Tensor],
        embeddings: torch.Tensor,
        domain_logits: Optional[torch.Tensor] = None,
        domain_labels: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        """각 손실 항목과 총합을 반환한다."""

        loss_dict: dict[str, torch.Tensor] = {}
        total = logits.new_tensor(0.0)

        if targets is not None:
            labels_onehot = F.one_hot(
                targets.long(), num_classes=logits.shape[-1]
            ).float()
            bce_loss = self.bce(logits, labels_onehot)
            total = total + bce_loss * self.cfg.bce_weight
            loss_dict["bce"] = bce_loss
            triplet_loss = self.triplet(embeddings, targets.long())
            total = total + triplet_loss * self.cfg.triplet_weight
            loss_dict["triplet"] = triplet_loss

        if domain_logits is not None and domain_labels is not None:
            domain_loss = self.domain(domain_logits, domain_labels)
            total = total + domain_loss * self.cfg.domain_weight
            loss_dict["domain"] = domain_loss

        loss_dict["total"] = total
        return loss_dict


__all__ = ["RKMv1Loss"]
