from __future__ import annotations

from typing import Iterable

import torch
from torch.optim.optimizer import Optimizer


class Lamb(Optimizer):
    """Layer-wise Adaptive Moments (LAMB) optimizer.

    무엇을/왜:
        LAMB는 대규모 배치 학습에서 안정적으로 수렴하도록 신뢰 비율(trust ratio)을 적용한
        Adam 계열 옵티마입니다. 파라미터 L2 노름과 업데이트 노름의 비율을 사용해 레이어별
        학습률을 조정합니다. 여기 구현은 AdamW 스타일의 decoupled weight decay를 사용합니다.

    참고: "Large Batch Optimization for Deep Learning: Training BERT in 76 minutes" (You et al.)

    Args:
        params: 파라미터 이터러블
        lr: 기본 학습률
        betas: (beta1, beta2)
        eps: 수치 안정성 상수
        weight_decay: decoupled weight decay 계수
    """

    def __init__(
        self,
        params: Iterable,
        lr: float = 1e-3,
        betas=(0.9, 0.999),
        eps: float = 1e-6,
        weight_decay: float = 0.0,
    ) -> None:
        if lr <= 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid eps: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta1: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta2: {betas[1]}")

        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):  # type: ignore[override]
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            wd = group["weight_decay"]
            lr = group["lr"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad

                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state["exp_avg_sq"] = torch.zeros_like(p, memory_format=torch.preserve_format)

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                state["step"] += 1

                # Adam moments
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                # Bias correction
                bias_c1 = 1 - beta1 ** state["step"]
                bias_c2 = 1 - beta2 ** state["step"]
                m_hat = exp_avg / bias_c1
                v_hat = exp_avg_sq / bias_c2

                adam_step = m_hat / (v_hat.sqrt() + eps)

                if wd != 0.0:
                    adam_step = adam_step + wd * p

                # Compute trust ratio
                w_norm = p.norm(p=2)
                s_norm = adam_step.norm(p=2)
                trust_ratio = torch.where(
                    (w_norm > 0) & (s_norm > 0),
                    w_norm / s_norm,
                    torch.ones((), device=p.device, dtype=p.dtype),
                )

                p.add_(adam_step, alpha=-lr * trust_ratio)

        return loss


def create_optimizer(
    params: Iterable,
    name: str = "adamw",
    lr: float = 3e-4,
    weight_decay: float = 0.05,
    betas=(0.9, 0.999),
):
    """설정에 맞는 옵티마를 생성한다.

    지원: adamw, adam, sgd, lamb
    """

    name = (name or "adamw").lower()
    if name == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay, betas=betas)
    if name == "sgd":
        return torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=weight_decay, nesterov=True)
    if name == "adam":
        return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay, betas=betas)
    if name == "lamb":
        return Lamb(params, lr=lr, weight_decay=weight_decay, betas=betas, eps=1e-6)
    raise KeyError(f"unknown optimizer: {name}")


def create_scheduler(*args, **kwargs):
    """엔진 내부에서 직접 스케줄러를 구성하므로 더미를 반환한다."""

    return None
