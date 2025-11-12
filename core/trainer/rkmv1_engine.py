"""RKMv1 전용 학습 엔진.

목적:
    SigLIP 기반 RKMv1 모델은 BCE + Triplet + Domain-Adversarial 손실을
    동시에 최적화해야 하므로 기존 CrossEntropy 전용 Trainer와 별도
    루프를 사용한다. 이 모듈은 Accelerate 환경 설정, 체크포인트,
    메트릭 로깅 등의 공통 기능은 `Trainer`로부터 상속받고,
    배치 처리/손실 계산만 커스터마이징한다.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
from accelerate.utils import tqdm

from core.trainer.engine import Trainer
from core.trainer.metrics import classification_metrics
from core.utils.logger import get_logger

LOGGER = get_logger(__name__)


class RKMv1Trainer(Trainer):
    """RKMv1 모델 전용 학습기."""

    def _build_criterion(self) -> nn.Module:  # type: ignore[override]
        """BCE 기반 커스텀 손실을 사용하므로 placeholder를 반환한다."""

        return nn.Identity()

    def _extract_batch(
        self, batch: object
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """DataLoader 배치를 RKMv1 입력 형태로 변환한다."""

        images: Optional[torch.Tensor] = None
        targets: Optional[torch.Tensor] = None
        domain: Optional[torch.Tensor] = None

        if isinstance(batch, dict):
            images = batch.get("images") or batch.get("image")
            targets = batch.get("targets") or batch.get("labels")
            domain = batch.get("domain_labels") or batch.get("domains")
        elif isinstance(batch, (list, tuple)):
            if len(batch) >= 2:
                images = batch[0]
                targets = batch[1]
            if len(batch) >= 3:
                domain = batch[2]
        else:
            raise TypeError(f"지원되지 않는 배치 타입: {type(batch)}")

        if images is None or targets is None:
            raise ValueError("배치에서 이미지/타깃 텐서를 찾을 수 없습니다.")

        if domain is not None and not torch.is_tensor(domain):
            domain = torch.as_tensor(domain)
        return images, targets, domain

    def _train_one_epoch(self, epoch: int) -> Dict[str, float]:  # type: ignore[override]
        if self.train_loader is None or self.optimizer is None:
            return {}

        base_model = self.accelerator.unwrap_model(self.model)
        if not hasattr(base_model, "loss_fn"):
            raise AttributeError("RKMv1Trainer는 model.loss_fn 속성을 필요로 합니다.")
        loss_module: nn.Module = getattr(base_model, "loss_fn")

        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)

        accum_steps = max(1, self.cfg.grad_accum_steps)
        total_loss = 0.0
        total_correct = 0
        total_examples = 0
        step_since_update = 0
        comp_loss_sums: Dict[str, float] = defaultdict(float)
        skipped_batches = 0

        total_steps = self.cfg.max_train_steps or len(self.train_loader)
        progress_bar = None
        if self.accelerator.is_main_process:
            progress_bar = tqdm(
                total=total_steps,
                desc=f"Epoch {epoch}",
                dynamic_ncols=True,
                leave=False,
            )

        for step, batch in enumerate(self.train_loader, start=1):
            images, targets, domain = self._extract_batch(batch)
            with self.accelerator.autocast():
                outputs = self.model(  # type: ignore[misc]
                    images, domain_labels=domain, return_dict=True
                )
                loss_dict = loss_module(
                    outputs["logits"],
                    targets,
                    outputs["fused"],
                    outputs["domain_logits"],
                    domain,
                )
                loss_value = loss_dict["total"]

            if not torch.isfinite(loss_value):
                skipped_batches += 1
                self.optimizer.zero_grad(set_to_none=True)
                if progress_bar is not None:
                    progress_bar.update(1)
                    progress_bar.set_postfix(loss="nan", acc="nan")
                LOGGER.warning(
                    "Epoch %d step %d에서 비유한 손실을 감지하여 배치를 건너뜁니다.",
                    epoch,
                    step,
                )
                continue

            loss = loss_value / float(accum_steps)
            self.accelerator.backward(loss)
            step_since_update += 1

            if step_since_update == accum_steps:
                if self.cfg.max_grad_norm and self.cfg.max_grad_norm > 0.0:
                    self.accelerator.clip_grad_norm_(
                        self.model.parameters(), self.cfg.max_grad_norm
                    )
                self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)
                step_since_update = 0

            batch_size = targets.size(0)
            total_examples += batch_size
            total_loss += loss_value.detach().item() * batch_size

            preds = outputs["logits"].detach().argmax(dim=1)
            total_correct += (preds == targets).sum().item()

            for name, value in loss_dict.items():
                if not torch.is_tensor(value) or name == "total":
                    continue
                comp_loss_sums[name] += value.detach().item() * batch_size

            if progress_bar is not None:
                progress_bar.update(1)
                if self.cfg.log_interval and (step % self.cfg.log_interval == 0):
                    current_loss = total_loss / max(total_examples, 1)
                    current_acc = total_correct / max(total_examples, 1)
                    progress_bar.set_postfix(
                        loss=f"{current_loss:.4f}", acc=f"{current_acc:.4f}"
                    )

            if self.cfg.max_train_steps and step >= self.cfg.max_train_steps:
                break

        if step_since_update > 0:
            if self.cfg.max_grad_norm and self.cfg.max_grad_norm > 0.0:
                self.accelerator.clip_grad_norm_(
                    self.model.parameters(), self.cfg.max_grad_norm
                )
            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)

        avg_loss, avg_acc = self._finalize_train_metrics(
            total_loss, total_examples, total_correct
        )
        if skipped_batches > 0 and self.accelerator.is_main_process:
            LOGGER.warning(
                "Epoch %d에서 총 %d개 배치를 NaN/Inf 손실로 건너뛰었습니다.",
                epoch,
                skipped_batches,
            )
        if progress_bar is not None:
            progress_bar.set_postfix(loss=f"{avg_loss:.4f}", acc=f"{avg_acc:.4f}")
            progress_bar.close()

        metrics = {"loss": float(avg_loss), "acc": float(avg_acc)}
        metrics.update(self._normalize_component_losses(comp_loss_sums, total_examples))
        return metrics

    def _finalize_train_metrics(
        self, loss_sum: float, sample_count: int, correct_sum: int
    ) -> Tuple[float, float]:
        loss_tensor = torch.tensor(
            loss_sum, device=self.accelerator.device, dtype=torch.float32
        )
        count_tensor = torch.tensor(
            sample_count, device=self.accelerator.device, dtype=torch.float32
        )
        correct_tensor = torch.tensor(
            correct_sum, device=self.accelerator.device, dtype=torch.float32
        )

        global_loss = self.accelerator.reduce(loss_tensor, reduction="sum").item()
        global_count = self.accelerator.reduce(count_tensor, reduction="sum").item()
        global_correct = self.accelerator.reduce(correct_tensor, reduction="sum").item()
        avg_loss = global_loss / max(global_count, 1.0)
        avg_acc = global_correct / max(global_count, 1.0)
        return avg_loss, avg_acc

    def _normalize_component_losses(
        self, comp_loss_sums: Dict[str, float], total_examples: int
    ) -> Dict[str, float]:
        if not comp_loss_sums or total_examples <= 0:
            return {}
        total_tensor = torch.tensor(
            total_examples, device=self.accelerator.device, dtype=torch.float32
        )
        global_total = self.accelerator.reduce(total_tensor, reduction="sum").item()
        if global_total <= 0:
            return {}
        metrics: Dict[str, float] = {}
        for name, value in comp_loss_sums.items():
            tensor = torch.tensor(
                value, device=self.accelerator.device, dtype=torch.float32
            )
            global_value = self.accelerator.reduce(tensor, reduction="sum").item()
            metrics[f"loss_{name}"] = global_value / global_total
        return metrics

    def _run_validation(self, epoch: int) -> Dict[str, float]:  # type: ignore[override]
        if self.val_loader is None:
            return {}

        base_model = self.accelerator.unwrap_model(self.model)
        if not hasattr(base_model, "loss_fn"):
            raise AttributeError("RKMv1Trainer는 model.loss_fn 속성을 필요로 합니다.")
        loss_module: nn.Module = getattr(base_model, "loss_fn")

        self.model.eval()
        total_loss = 0.0
        total_examples = 0
        comp_loss_sums: Dict[str, float] = defaultdict(float)
        all_logits: list[torch.Tensor] = []
        all_targets: list[torch.Tensor] = []

        with torch.no_grad():
            for step, batch in enumerate(self.val_loader, start=1):
                images, targets, domain = self._extract_batch(batch)
                with self.accelerator.autocast():
                    outputs = self.model(  # type: ignore[misc]
                        images, domain_labels=domain, return_dict=True
                    )
                    loss_dict = loss_module(
                        outputs["logits"],
                        targets,
                        outputs["fused"],
                        outputs["domain_logits"],
                        domain,
                    )
                    loss_value = loss_dict["total"]

                batch_size = targets.size(0)
                total_examples += batch_size
                total_loss += loss_value.detach().item() * batch_size

                gathered_logits = self.accelerator.gather(outputs["logits"].detach())
                gathered_targets = self.accelerator.gather(targets.detach())
                all_logits.append(gathered_logits.cpu())
                all_targets.append(gathered_targets.cpu())

                for name, value in loss_dict.items():
                    if not torch.is_tensor(value) or name == "total":
                        continue
                    comp_loss_sums[name] += value.detach().item() * batch_size

                if self.cfg.max_val_steps and step >= self.cfg.max_val_steps:
                    break

        loss_tensor = torch.tensor(
            total_loss, device=self.accelerator.device, dtype=torch.float32
        )
        count_tensor = torch.tensor(
            total_examples, device=self.accelerator.device, dtype=torch.float32
        )
        global_loss = self.accelerator.reduce(loss_tensor, reduction="sum").item()
        global_count = self.accelerator.reduce(count_tensor, reduction="sum").item()
        avg_loss = global_loss / max(global_count, 1.0)

        metrics: Dict[str, float] = {"loss": float(avg_loss)}
        metrics.update(self._normalize_component_losses(comp_loss_sums, total_examples))

        if all_logits and all_targets and global_count > 0:
            logits_full = torch.cat(all_logits, dim=0)
            targets_full = torch.cat(all_targets, dim=0)
            bundle = classification_metrics(logits_full, targets_full)
            bundle_dict = bundle.as_dict()
            bundle_dict.pop("loss", None)
            metrics.update(bundle_dict)

        return metrics


__all__ = ["RKMv1Trainer"]
