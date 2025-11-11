"""간소화된 학습/검증 엔진."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
from accelerate import Accelerator
from accelerate.utils import tqdm
from torch.utils.data import DataLoader

from core.utils.logger import get_logger, get_train_logger

from .experiment_manager import ExperimentManager
from .metrics import classification_metrics
from .optim import create_optimizer

logger = get_logger(__name__)


@dataclass
class TrainCfg:
    """학습 하이퍼파라미터 데이터클래스."""

    epochs: int = 1
    lr: float = 3.0e-4
    weight_decay: float = 0.0
    optimizer: str = "adamw"
    scheduler: str = "none"
    warmup_epochs: Optional[int] = None
    sched_factor: Optional[float] = None
    sched_patience: Optional[int] = None
    sched_min_lr: Optional[float] = None
    sched_monitor: Optional[str] = None
    sched_mode: Optional[str] = None
    mixed_precision: Optional[str] = None
    grad_accum_steps: int = 1
    log_interval: int = 50
    criterion: str = "ce"
    label_smoothing: float = 0.0
    max_grad_norm: float = 0.0
    patch_chunk_size: int = 1
    partial_epochs: Optional[int] = None
    full_epochs: Optional[int] = None
    partial_steps: Optional[int] = None
    full_steps: Optional[int] = None
    max_train_steps: Optional[int] = None
    max_val_steps: Optional[int] = None
    seed: Optional[int] = None
    early_stop: bool = False
    early_patience: int = 5
    early_monitor: str = "val_loss"
    early_mode: str = "min"
    ckpt_monitor: str = "val_loss"
    ckpt_mode: str = "min"
    step_debug_logging: bool = False


class Trainer:
    """기본적인 이미지 분류 학습/검증 루프를 제공하는 엔진."""

    def __init__(
        self,
        *,
        model: nn.Module,
        train_loader: Optional[DataLoader],
        val_loader: Optional[DataLoader],
        cfg: TrainCfg,
        experiment: ExperimentManager,
        accelerator: Accelerator,
    ) -> None:
        """모델과 데이터로더, 설정을 받아 엔진을 초기화한다."""

        self.cfg = cfg
        self.accelerator = accelerator
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.experiment = experiment

        self.system_logger = get_logger(self.__class__.__name__)
        self.train_logger = get_train_logger()

        self.criterion = self._build_criterion()
        self.optimizer = self._build_optimizer() if self.train_loader is not None else None
        self.scheduler = self._build_scheduler() if self.optimizer is not None else None

        self.current_epoch: int = 0
        self._best_metric: Optional[float] = None
        self._patience_counter: int = 0

        self._prepare_components()

    # ------------------------------------------------------------------
    # 초기화 관련 유틸
    # ------------------------------------------------------------------
    def _build_criterion(self) -> nn.Module:
        """손실 함수를 생성한다."""

        name = (self.cfg.criterion or "ce").lower()
        if name in {"ce", "cross_entropy"}:
            return nn.CrossEntropyLoss(label_smoothing=float(self.cfg.label_smoothing))
        raise ValueError(f"지원하지 않는 손실 함수: {self.cfg.criterion}")

    def _build_optimizer(self) -> torch.optim.Optimizer:
        """설정에 맞춰 옵티마를 생성한다."""

        return create_optimizer(
            self.model.parameters(),
            name=self.cfg.optimizer,
            lr=self.cfg.lr,
            weight_decay=self.cfg.weight_decay,
        )

    def _build_scheduler(self) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
        """간단한 러닝레이트 스케줄러를 구성한다."""

        name = (self.cfg.scheduler or "none").lower()
        if name in {"none", "constant"}:
            return None
        if name == "cosine":
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,  # type: ignore[arg-type]
                T_max=max(1, self.cfg.epochs),
                eta_min=self.cfg.sched_min_lr or 0.0,
            )
        if name in {"step", "steplr"}:
            step_size = self.cfg.sched_patience or 1
            gamma = self.cfg.sched_factor or 0.1
            return torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=step_size, gamma=gamma)  # type: ignore[arg-type]
        logger.warning("알 수 없는 스케줄러 %s - 사용하지 않습니다.", self.cfg.scheduler)
        return None

    def _prepare_components(self) -> None:
        """Accelerate로 모델과 옵티마, 데이터로더를 준비한다."""

        components: list[Any] = [self.model]
        has_optimizer = self.optimizer is not None
        if has_optimizer:
            components.append(self.optimizer)
        if self.train_loader is not None:
            components.append(self.train_loader)
        if self.val_loader is not None:
            components.append(self.val_loader)

        prepared = self.accelerator.prepare(*components)

        idx = 0
        self.model = prepared[idx]
        idx += 1
        if has_optimizer:
            self.optimizer = prepared[idx]
            idx += 1
        if self.train_loader is not None:
            self.train_loader = prepared[idx]
            idx += 1
        if self.val_loader is not None:
            self.val_loader = prepared[idx]

    # ------------------------------------------------------------------
    # 학습 / 검증 루프
    # ------------------------------------------------------------------
    def fit(self) -> Dict[str, float]:
        """전체 에폭에 대해 학습을 수행하고 마지막 검증 지표를 반환한다."""

        if self.train_loader is None:
            raise RuntimeError("학습용 DataLoader가 필요합니다.")

        last_metrics: Dict[str, float] = {}
        for epoch in range(1, self.cfg.epochs + 1):
            self.current_epoch = epoch
            train_metrics = self._train_one_epoch(epoch)
            if train_metrics:
                self.experiment.log_metrics(epoch, "train", train_metrics)

            val_metrics = self._run_validation(epoch)
            last_metrics = val_metrics or train_metrics
            if val_metrics:
                self.experiment.log_metrics(epoch, "val", val_metrics)

            if self.accelerator.is_main_process:
                self._log_epoch_summary(epoch, train_metrics, val_metrics)

            is_best = self._update_early_stopping(val_metrics or train_metrics)
            self._save_checkpoint(epoch, is_best=is_best)

            if self.scheduler is not None:
                self.scheduler.step()

            if self.cfg.early_stop and self._should_stop():
                self.system_logger.info("Early stopping 조건이 충족되어 학습을 종료합니다.")
                break

            self.accelerator.wait_for_everyone()

        return last_metrics

    def validate_only(self) -> Dict[str, float]:
        """검증만 수행하고 결과를 반환한다."""

        metrics = self._run_validation(epoch=0)
        self.accelerator.wait_for_everyone()
        return metrics

    # ------------------------------------------------------------------
    # 세부 루프 구현
    # ------------------------------------------------------------------
    def _train_one_epoch(self, epoch: int) -> Dict[str, float]:
        """단일 에폭 학습을 수행한다."""

        if self.train_loader is None or self.optimizer is None:
            return {}

        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)

        accum_steps = max(1, self.cfg.grad_accum_steps)
        total_loss = 0.0
        total_correct = 0
        total_examples = 0
        step_since_update = 0

        total_steps: Optional[int] = self.cfg.max_train_steps
        if total_steps is None:
            try:
                total_steps = len(self.train_loader)
            except (TypeError, AttributeError):
                total_steps = None

        progress_bar = None
        if self.accelerator.is_main_process:
            progress_bar = tqdm(
                total=total_steps,
                desc=f"Epoch {epoch}",
                dynamic_ncols=True,
                leave=False,
            )

        for step, batch in enumerate(self.train_loader, start=1):
            images, targets = batch
            with self.accelerator.autocast():
                logits = self.model(images)
                loss_value = self.criterion(logits, targets)
            loss = loss_value / float(accum_steps)
            self.accelerator.backward(loss)

            step_since_update += 1
            if step_since_update == accum_steps:
                if self.cfg.max_grad_norm and self.cfg.max_grad_norm > 0.0:
                    self.accelerator.clip_grad_norm_(self.model.parameters(), self.cfg.max_grad_norm)
                self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)
                step_since_update = 0

            batch_size = targets.size(0)
            total_examples += batch_size
            total_loss += loss_value.detach().item() * batch_size

            preds = logits.detach().argmax(dim=1)
            total_correct += (preds == targets).sum().item()

            if progress_bar is not None:
                progress_bar.update(1)
                if self.cfg.log_interval and (step % self.cfg.log_interval == 0):
                    current_loss = total_loss / max(total_examples, 1)
                    current_acc = total_correct / max(total_examples, 1)
                    progress_bar.set_postfix(
                        loss=f"{current_loss:.4f}",
                        acc=f"{current_acc:.4f}",
                    )

            if self.cfg.max_train_steps and step >= self.cfg.max_train_steps:
                break

        if step_since_update > 0:
            if self.cfg.max_grad_norm and self.cfg.max_grad_norm > 0.0:
                self.accelerator.clip_grad_norm_(self.model.parameters(), self.cfg.max_grad_norm)
            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)

        loss_tensor = torch.tensor(total_loss, device=self.accelerator.device, dtype=torch.float32)
        count_tensor = torch.tensor(total_examples, device=self.accelerator.device, dtype=torch.float32)
        correct_tensor = torch.tensor(total_correct, device=self.accelerator.device, dtype=torch.float32)

        global_loss = self.accelerator.reduce(loss_tensor, reduction="sum").item()
        global_count = self.accelerator.reduce(count_tensor, reduction="sum").item()
        global_correct = self.accelerator.reduce(correct_tensor, reduction="sum").item()

        avg_loss = global_loss / max(global_count, 1.0)
        avg_acc = global_correct / max(global_count, 1.0)

        if progress_bar is not None:
            progress_bar.set_postfix(loss=f"{avg_loss:.4f}", acc=f"{avg_acc:.4f}")
            progress_bar.close()

        return {"loss": float(avg_loss), "acc": float(avg_acc)}

    def _run_validation(self, epoch: int) -> Dict[str, float]:
        """검증 루프를 실행해 지표를 계산한다."""

        if self.val_loader is None:
            return {}

        self.model.eval()
        total_loss = 0.0
        total_examples = 0
        all_logits: list[torch.Tensor] = []
        all_targets: list[torch.Tensor] = []

        with torch.no_grad():
            for step, (images, targets) in enumerate(self.val_loader, start=1):
                with self.accelerator.autocast():
                    logits = self.model(images)
                    loss_value = self.criterion(logits, targets)

                batch_size = targets.size(0)
                total_examples += batch_size
                total_loss += loss_value.detach().item() * batch_size

                gathered_logits = self.accelerator.gather(logits.detach())
                gathered_targets = self.accelerator.gather(targets.detach())
                all_logits.append(gathered_logits.cpu())
                all_targets.append(gathered_targets.cpu())

                if self.cfg.max_val_steps and step >= self.cfg.max_val_steps:
                    break

        loss_tensor = torch.tensor(total_loss, device=self.accelerator.device, dtype=torch.float32)
        count_tensor = torch.tensor(total_examples, device=self.accelerator.device, dtype=torch.float32)
        global_loss = self.accelerator.reduce(loss_tensor, reduction="sum").item()
        global_count = self.accelerator.reduce(count_tensor, reduction="sum").item()
        avg_loss = global_loss / max(global_count, 1.0)

        metrics: Dict[str, float] = {"loss": float(avg_loss)}
        if all_logits and all_targets and global_count > 0:
            logits_full = torch.cat(all_logits, dim=0)
            targets_full = torch.cat(all_targets, dim=0)
            bundle = classification_metrics(logits_full, targets_full)
            bundle_dict = bundle.as_dict()
            bundle_dict.pop("loss", None)
            metrics.update(bundle_dict)

        return metrics

    # ------------------------------------------------------------------
    # 부가 기능
    # ------------------------------------------------------------------
    def _update_early_stopping(self, metrics: Dict[str, float]) -> bool:
        """Early stopping 및 베스트 모델 판단을 갱신한다."""

        monitor_key = (self.cfg.early_monitor or "val_loss").lower()
        target_value = metrics.get(monitor_key)
        if target_value is None and monitor_key.startswith("val_"):
            target_value = metrics.get(monitor_key.replace("val_", "", 1))
        if target_value is None:
            target_value = metrics.get("loss")

        improved = False
        if target_value is not None:
            if self._best_metric is None:
                improved = True
                self._best_metric = float(target_value)
            elif self.cfg.early_mode == "min":
                improved = target_value < self._best_metric
            else:
                improved = target_value > self._best_metric

            if improved:
                self._best_metric = float(target_value)
                self._patience_counter = 0
            else:
                self._patience_counter += 1

        if metrics:
            try:
                improved |= self.experiment.update_best(metrics)
            except Exception as exc:  # pragma: no cover - 방어적 로깅
                logger.warning("베스트 지표 갱신 실패: %s", exc)

        return improved

    def _save_checkpoint(self, epoch: int, *, is_best: bool) -> None:
        """체크포인트를 저장한다."""

        if not self.accelerator.is_main_process:
            return

        state = {
            "epoch": epoch,
            "model": self.accelerator.get_state_dict(self.model),
            "optimizer": self.optimizer.state_dict() if self.optimizer is not None else None,
            "scheduler": self.scheduler.state_dict() if self.scheduler is not None else None,
        }
        self.experiment.save_checkpoint(state, is_best=is_best)

    def _should_stop(self) -> bool:
        """Early stopping 조건을 검사한다."""

        if not self.cfg.early_stop:
            return False
        return self._patience_counter >= max(1, self.cfg.early_patience)

    def _log_epoch_summary(
        self,
        epoch: int,
        train_metrics: Dict[str, float],
        val_metrics: Dict[str, float],
    ) -> None:
        """에폭별 학습/검증 요약을 단일 로그로 출력한다."""

        def _fmt(metrics: Dict[str, float], key: str) -> str:
            value = metrics.get(key)
            return f"{value:.4f}" if value is not None else "N/A"

        message = (
            "에폭 %d/%d 완료 | train_loss=%s train_acc=%s | val_loss=%s val_acc=%s"
            % (
                epoch,
                self.cfg.epochs,
                _fmt(train_metrics, "loss"),
                _fmt(train_metrics, "acc"),
                _fmt(val_metrics, "loss"),
                _fmt(val_metrics, "acc"),
            )
        )
        self.train_logger.info(message)
