from __future__ import annotations

from typing import Iterable

import torch


def create_optimizer(params: Iterable, name: str = "adamw", lr: float = 3e-4, weight_decay: float = 0.05, betas=(0.9, 0.999)):
    """설정에 맞는 옵티마를 생성한다."""

    name = (name or "adamw").lower()
    if name == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay, betas=betas)
    if name == "sgd":
        return torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=weight_decay, nesterov=True)
    if name == "adam":
        return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay, betas=betas)
    raise KeyError(f"unknown optimizer: {name}")


def create_scheduler(*args, **kwargs):
    """엔진 내부에서 직접 스케줄러를 구성하므로 더미를 반환한다."""

    return None
