from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import math
import torch
import torch.nn as nn
from accelerate import Accelerator
from torch.utils.data import DataLoader

from core.utils.logger import get_logger, get_train_logger
from core.utils.seed import set_seed as set_global_seed
from .experiment_manager import ExperimentManager, MonitorConfig
from .metrics import classification_metrics, top1_accuracy
from .optim import create_optimizer

@dataclass
class TrainCfg:
    """학습 루프 전역 설정."""

    epochs: int = 10
    lr: float = 3e-4
    weight_decay: float = 0.05
    optimizer: str = "adamw"
    scheduler: str = "cosine"  # "cosine" | "reduce_on_plateau"
    warmup_epochs: int = 0
    # ReduceLROnPlateau options
    sched_factor: float = 0.5
    sched_patience: int = 3
    sched_min_lr: float = 1e-6
    sched_monitor: str = "val_loss"  # val_loss | val_acc
    sched_mode: str = "min"  # min|max
    mixed_precision: Optional[str] = None
    grad_accum_steps: int = 1
    log_interval: int = 50
    criterion: str = "ce"
    label_smoothing: float = 0.0
    max_grad_norm: float = 1.0
    partial_epochs: Optional[int] = None
    full_epochs: Optional[int] = None
    partial_steps: Optional[int] = None
    full_steps: Optional[int] = None
    seed: Optional[int] = None
    # Early stopping
    early_stop: bool = False
    early_patience: int = 8
    early_monitor: str = "val_loss"
    early_mode: str = "min"
    # Checkpoint selection
    ckpt_monitor: str = "val_loss"
    ckpt_mode: str = "min"


class Trainer:
    """Accelerate 기반 학습 관리자."""

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader],
        out_dir: str,
        cfg: TrainCfg,
        experiment: Optional[ExperimentManager] = None,
        accelerator: Optional[Accelerator] = None,
    ) -> None:
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.cfg = cfg

        monitor = MonitorConfig(key=cfg.ckpt_monitor, mode=cfg.ckpt_mode)
        self.experiment = experiment or ExperimentManager(Path(out_dir), monitor=monitor)
        self.out_dir = Path(self.experiment.root)

        self.system_logger = get_logger(__name__)
        self.train_logger = get_train_logger()

        self.accel = accelerator or Accelerator(
            mixed_precision=cfg.mixed_precision or "no",
            gradient_accumulation_steps=cfg.grad_accum_steps,
        )
        try:
            gpu_cnt = torch.cuda.device_count()
        except Exception:
            gpu_cnt = 0
        world = getattr(self.accel.state, "num_processes", 1)
        device = str(getattr(self.accel, "device", "cpu"))
        per_proc_bs = getattr(train_loader, "batch_size", None)
        global_bs = (per_proc_bs * world) if isinstance(per_proc_bs, int) else None
        self.system_logger.info(
            "Accelerate 초기화 - processes=%d, device=%s, cuda_devices=%d, per_proc_bs=%s, global_bs=%s",
            world,
            device,
            gpu_cnt,
            str(per_proc_bs),
            str(global_bs),
        )

        self.criterion = self._build_criterion(cfg.criterion)
        self.optimizer = create_optimizer(
            self.model.parameters(),
            name=cfg.optimizer,
            lr=cfg.lr,
            weight_decay=cfg.weight_decay,
        )
        self._scheduler_needs_metric = False
        self.scheduler = self._build_scheduler()

        if self.scheduler is not None:
            (self.model, self.optimizer, self.train_loader, self.val_loader, self.scheduler) = self.accel.prepare(
                self.model, self.optimizer, self.train_loader, self.val_loader, self.scheduler
            )
        else:
            (self.model, self.optimizer, self.train_loader, self.val_loader) = self.accel.prepare(
                self.model, self.optimizer, self.train_loader, self.val_loader
            )

    # ------------------------------------------------------------------
    # Builders
    # ------------------------------------------------------------------
    def _build_criterion(self, name: str) -> nn.Module:
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
        smoothing = float(getattr(self.cfg, "label_smoothing", 0.0) or 0.0)
        try:
            return nn.CrossEntropyLoss(label_smoothing=smoothing)
        except TypeError:
            return nn.CrossEntropyLoss()

    def _build_scheduler(self):
        name = (self.cfg.scheduler or "").lower()
        if name in {"reduce_on_plateau", "plateau", "reducelronplateau"}:
            mode = (self.cfg.sched_mode or "min").lower()
            self._scheduler_needs_metric = True
            return torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode=mode,
                factor=self.cfg.sched_factor,
                patience=self.cfg.sched_patience,
                min_lr=self.cfg.sched_min_lr,
                verbose=True,
            )
        if name != "cosine":
            return torch.optim.lr_scheduler.LambdaLR(self.optimizer, lambda _: 1.0)

        warmup = max(0, self.cfg.warmup_epochs)
        total = max(self.cfg.epochs, warmup + 1)

        def lr_lambda(epoch: int) -> float:
            if epoch < warmup:
                return float(epoch + 1) / float(max(1, warmup))
            progress = (epoch - warmup) / float(max(1, total - warmup))
            return 0.5 * (1.0 + math.cos(math.pi * progress))

        return torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lr_lambda)

    # ------------------------------------------------------------------
    # Epoch helpers
    # ------------------------------------------------------------------
    def _train_one_epoch(self, epoch: int, steps_limit: int) -> Dict[str, float]:
        sampler = getattr(self.train_loader, "sampler", None)
        if hasattr(sampler, "set_epoch"):
            sampler.set_epoch(epoch)

        self.model.train()
        total_loss = 0.0
        total_acc = 0.0
        total_count = 0
        steps_processed = 0

        for step, (images, targets) in enumerate(self.train_loader, start=1):
            with self.accel.accumulate(self.model):
                with self.accel.autocast():
                    logits = self.model(images)
                    loss = self.criterion(logits, targets)
                self.accel.backward(loss)
                if self.accel.sync_gradients and self.cfg.max_grad_norm > 0:
                    self.accel.clip_grad_norm_(self.model.parameters(), max_norm=self.cfg.max_grad_norm)
                self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)

            bs = targets.size(0)
            total_loss += loss.detach().item() * bs
            total_acc += top1_accuracy(logits.detach(), targets) * bs
            total_count += bs

            steps_processed += 1

            if (step % max(1, self.cfg.log_interval)) == 0 and self.accel.is_main_process:
                avg_loss = total_loss / max(1, total_count)
                avg_acc = total_acc / max(1, total_count)
                self.train_logger.info(
                    "에폭 %d 스텝 %d/%d - loss=%.4f acc=%.4f",
                    epoch,
                    step,
                    steps_limit,
                    avg_loss,
                    avg_acc,
                )

            if steps_limit > 0 and steps_processed >= steps_limit:
                break

        return {"loss": total_loss / max(1, total_count), "acc": total_acc / max(1, total_count)}

    @torch.no_grad()
    def _validate(self) -> Dict[str, float]:
        if self.val_loader is None:
            return {"loss": 0.0, "acc": 0.0}
        self.model.eval()
        total_loss = 0.0
        total_acc = 0.0
        total_count = 0
        logits_collector = []
        targets_collector = []
        for images, targets in self.val_loader:
            with self.accel.autocast():
                logits = self.model(images)
                loss = self.criterion(logits, targets)
            bs = targets.size(0)
            total_loss += loss.detach().item() * bs
            total_acc += top1_accuracy(logits.detach(), targets) * bs
            total_count += bs

            logits_collector.append(self.accel.gather(logits.detach()))
            targets_collector.append(self.accel.gather(targets))

        metrics = {"loss": total_loss / max(1, total_count), "acc": total_acc / max(1, total_count)}
        if logits_collector:
            logits_cat = torch.cat(logits_collector)
            targets_cat = torch.cat(targets_collector)
            bundle = classification_metrics(logits_cat, targets_cat)
            bundle.loss = metrics["loss"]
            bundle.acc = metrics["acc"]
            metrics.update(bundle.as_dict())
        return metrics

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def fit(self) -> Dict[str, float]:
        self.out_dir.mkdir(parents=True, exist_ok=True)

        total_epochs = max(1, int(self.cfg.epochs))
        partial_epochs = self.cfg.partial_epochs if self.cfg.partial_epochs is not None else total_epochs
        full_epochs = self.cfg.full_epochs if self.cfg.full_epochs is not None else max(total_epochs - partial_epochs, 0)

        if partial_epochs < 0:
            partial_epochs = 0
        if full_epochs < 0:
            full_epochs = 0

        if partial_epochs + full_epochs < total_epochs:
            partial_epochs = total_epochs - full_epochs
        elif partial_epochs + full_epochs > total_epochs:
            if self.cfg.full_epochs is not None:
                partial_epochs = max(total_epochs - full_epochs, 0)
            else:
                full_epochs = max(total_epochs - partial_epochs, 0)

        max_steps_available = len(self.train_loader)
        partial_steps = self.cfg.partial_steps if (self.cfg.partial_steps or 0) > 0 else None
        full_steps = self.cfg.full_steps if (self.cfg.full_steps or 0) > 0 else None

        self.system_logger.info(
            "훈련 스케줄 설정 - total_epochs=%d, partial_epochs=%d, full_epochs=%d, partial_steps=%s, full_steps=%s",
            total_epochs,
            partial_epochs,
            full_epochs,
            str(partial_steps) if partial_steps is not None else "auto",
            str(full_steps) if full_steps is not None else "auto",
        )

        early_monitor = (self.cfg.early_monitor or self.cfg.ckpt_monitor or "val_loss").lower()
        early_mode = (self.cfg.early_mode or self.cfg.ckpt_mode or "min").lower()
        early_best: Optional[float] = None
        epochs_no_improve = 0
        last_val_metrics: Dict[str, float] = {}

        for epoch in range(1, total_epochs + 1):
            if self.cfg.seed is not None:
                set_global_seed(int(self.cfg.seed) + epoch)
            if epoch <= partial_epochs:
                steps_target = partial_steps
                phase = "partial"
            else:
                steps_target = full_steps
                phase = "full"

            if steps_target is None or steps_target <= 0:
                steps_limit = max_steps_available
            else:
                steps_limit = max(1, min(int(steps_target), max_steps_available))

            self.system_logger.info(
                "에폭 %d/%d (%s) - steps_limit=%d", epoch, total_epochs, phase, steps_limit
            )

            train_metrics = self._train_one_epoch(epoch, steps_limit)
            val_metrics = self._validate()
            last_val_metrics = val_metrics

            if self.scheduler is not None:
                if self._scheduler_needs_metric:
                    metric_key = (self.cfg.sched_monitor or "val_loss").lower()
                    metric = val_metrics["loss"] if metric_key == "val_loss" else val_metrics.get("acc", 0.0)
                    self.scheduler.step(metric)
                else:
                    self.scheduler.step()

            if not self.accel.is_main_process:
                continue

            self.train_logger.info(
                "에폭 %03d 완료 - train_loss=%.4f train_acc=%.4f val_loss=%.4f val_acc=%.4f",
                epoch,
                train_metrics.get("loss", 0.0),
                train_metrics.get("acc", 0.0),
                val_metrics.get("loss", 0.0),
                val_metrics.get("acc", 0.0),
            )

            self.experiment.log_metrics(epoch, "train", train_metrics)
            self.experiment.log_metrics(epoch, "val", val_metrics)

            state = {
                "epoch": epoch,
                "model": self.accel.get_state_dict(self.model),
                "optimizer": self.optimizer.state_dict(),
                "train": train_metrics,
                "val": val_metrics,
            }
            is_best = self.experiment.update_best(val_metrics)
            self.experiment.save_checkpoint(state, is_best=is_best)

            monitor_value = val_metrics.get(early_monitor)
            if monitor_value is not None:
                improved = False
                if early_best is None:
                    improved = True
                elif early_mode == "min":
                    improved = monitor_value < early_best
                else:
                    improved = monitor_value > early_best
                if improved:
                    early_best = monitor_value
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1

            if self.cfg.early_stop and epochs_no_improve >= max(1, self.cfg.early_patience):
                self.train_logger.info(
                    "Early stopping 발동 - patience=%d, monitor=%s, mode=%s",
                    self.cfg.early_patience,
                    self.cfg.early_monitor,
                    self.cfg.early_mode,
                )
                break

        return last_val_metrics
