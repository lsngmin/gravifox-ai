"""학습에 사용되는 기본 지표 구현 모듈."""

from __future__ import annotations

import torch


@torch.no_grad()
def top1_accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    """Top-1 정확도를 계산한다."""

    preds = logits.argmax(dim=1)
    correct = (preds == targets).sum().item()
    total = targets.numel()
    return float(correct) / float(max(total, 1))
