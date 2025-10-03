from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import math
import torch
import torch.nn as nn
from accelerate import Accelerator
from torch.utils.data import DataLoader

from core.utils.logger import get_logger
from .metrics import top1_accuracy
from .optim import create_optimizer
from ..utils.checkpoint import save_checkpoint


logger = get_logger(__name__)


@dataclass
class TrainCfg:
    """학습 루프 설정."""

    epochs: int = 10
    lr: float = 3e-4
    weight_decay: float = 0.05
    optimizer: str = "adamw"
    scheduler: str = "cosine"
    warmup_epochs: int = 0
    mixed_precision: Optional[str] = None
    grad_accum_steps: int = 1
    log_interval: int = 50
    save_interval: int = 1
    criterion: str = "ce"


class Trainer:
    """Accelerate 기반 학습 관리자."""

    def __init__(self, model: nn.Module, train_loader: DataLoader, val_loader: Optional[DataLoader], out_dir: str, cfg: TrainCfg):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.out_dir = Path(out_dir)
        self.cfg = cfg

        self.accel = Accelerator(mixed_precision=cfg.mixed_precision or "no", gradient_accumulation_steps=cfg.grad_accum_steps)
        # 멀티 GPU 활성 여부와 장치 정보를 짧게 로그(가벼운 확인용)
        try:
            gpu_cnt = torch.cuda.device_count()
        except Exception:
            gpu_cnt = 0
        world = getattr(self.accel.state, "num_processes", 1)
        device = str(getattr(self.accel, "device", "cpu"))
        per_proc_bs = getattr(train_loader, "batch_size", None)
        global_bs = (per_proc_bs * world) if isinstance(per_proc_bs, int) else None
        logger.info(
            "Accelerate 초기화 - processes=%d, device=%s, cuda_devices=%d, per_proc_bs=%s, global_bs=%s",
            world,
            device,
            gpu_cnt,
            str(per_proc_bs),
            str(global_bs),
        )
        self.criterion = self._build_criterion(cfg.criterion)
        self.optimizer = create_optimizer(self.model.parameters(), name=cfg.optimizer, lr=cfg.lr, weight_decay=cfg.weight_decay)
        self.scheduler = self._build_scheduler()

        if self.scheduler is not None:
            (self.model, self.optimizer, self.train_loader, self.val_loader, self.scheduler) = self.accel.prepare(
                self.model, self.optimizer, self.train_loader, self.val_loader, self.scheduler
            )
        else:
            (self.model, self.optimizer, self.train_loader, self.val_loader) = self.accel.prepare(
                self.model, self.optimizer, self.train_loader, self.val_loader
            )

        self.best_val_acc = 0.0

    def _build_criterion(self, name: str) -> nn.Module:
        """손실 함수를 구성한다."""

        if name.lower() == "focal":
            class FocalLoss(nn.Module):
                def __init__(self, gamma: float = 2.0):
                    super().__init__()
                    self.gamma = gamma

                def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
                    log_probs = torch.nn.functional.log_softmax(logits, dim=1)
                    probs = torch.exp(log_probs)
                    focal = (1 - probs) ** self.gamma
                    loss = torch.nn.functional.nll_loss(focal * log_probs, targets)
                    return loss

            return FocalLoss()
        return nn.CrossEntropyLoss()

    def _build_scheduler(self):
        """러닝레이트 스케줄러를 구성한다."""

        if self.cfg.scheduler.lower() != "cosine":
            return torch.optim.lr_scheduler.LambdaLR(self.optimizer, lambda _: 1.0)

        warmup = max(0, self.cfg.warmup_epochs)
        total = max(self.cfg.epochs, warmup + 1)

        def lr_lambda(epoch: int) -> float:
            if epoch < warmup:
                return float(epoch + 1) / float(max(1, warmup))
            progress = (epoch - warmup) / float(max(1, total - warmup))
            return 0.5 * (1.0 + math.cos(math.pi * progress))

        return torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lr_lambda)

    def _train_one_epoch(self, epoch: int) -> Dict[str, float]:
        """한 에폭 학습을 수행한다."""

        self.model.train()
        total_loss = 0.0
        total_acc = 0.0
        total_count = 0

        for step, (images, targets) in enumerate(self.train_loader, start=1):
            with self.accel.accumulate(self.model):
                with self.accel.autocast():
                    logits = self.model(images)
                    loss = self.criterion(logits, targets)
                self.accel.backward(loss)
                # Accelerate는 내부적으로 unscale → clip → rescale을 관리한다.
                # 누적 스텝(accumulation) 중간에는 sync_gradients=False이므로 clip을 건너뛴다.
                if self.accel.sync_gradients:
                    self.accel.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)

            bs = targets.size(0)
            total_loss += loss.detach().item() * bs
            total_acc += top1_accuracy(logits.detach(), targets) * bs
            total_count += bs

            if (step % max(1, self.cfg.log_interval)) == 0 and self.accel.is_main_process:
                avg_loss = total_loss / max(1, total_count)
                avg_acc = total_acc / max(1, total_count)
                logger.info("에폭 %d 스텝 %d - loss=%.4f acc=%.4f", epoch, step, avg_loss, avg_acc)

        return {"loss": total_loss / max(1, total_count), "acc": total_acc / max(1, total_count)}

    @torch.no_grad()
    def _validate(self) -> Dict[str, float]:
        """검증 데이터셋에서 지표를 계산한다."""

        if self.val_loader is None:
            return {"loss": 0.0, "acc": 0.0}
        self.model.eval()
        total_loss = 0.0
        total_acc = 0.0
        total_count = 0
        for images, targets in self.val_loader:
            with self.accel.autocast():
                logits = self.model(images)
                loss = self.criterion(logits, targets)
            bs = targets.size(0)
            total_loss += loss.detach().item() * bs
            total_acc += top1_accuracy(logits.detach(), targets) * bs
            total_count += bs
        return {"loss": total_loss / max(1, total_count), "acc": total_acc / max(1, total_count)}

    def fit(self) -> None:
        """전체 학습 과정을 실행한다."""

        self.out_dir.mkdir(parents=True, exist_ok=True)
        ckpt_dir = self.out_dir / "checkpoints"
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        for epoch in range(1, self.cfg.epochs + 1):
            train_metrics = self._train_one_epoch(epoch)
            val_metrics = self._validate()
            if self.scheduler is not None:
                self.scheduler.step()

            if self.accel.is_main_process:
                logger.info(
                    "에폭 %03d 완료 - train_loss=%.4f train_acc=%.4f val_loss=%.4f val_acc=%.4f",  # noqa: E501
                    epoch,
                    train_metrics["loss"],
                    train_metrics["acc"],
                    val_metrics["loss"],
                    val_metrics["acc"],
                )

                state = {
                    "epoch": epoch,
                    "model": self.accel.get_state_dict(self.model),
                    "optimizer": self.optimizer.state_dict(),
                    "train": train_metrics,
                    "val": val_metrics,
                }
                save_checkpoint(state, ckpt_dir, filename="last.pt")
                if val_metrics["acc"] >= self.best_val_acc:
                    self.best_val_acc = val_metrics["acc"]
                    path = save_checkpoint(state, ckpt_dir, filename="best.pt")
                    logger.info("새로운 best 체크포인트 저장: %s", path)
