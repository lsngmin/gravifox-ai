from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Optional, Sequence, Tuple, List, Any
import gc

import datetime
import json
import math
import os
import logging
import torch
import torch.nn as nn
import torch.distributed as dist
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs
from torch.utils.data import DataLoader
from PIL import Image
from torchvision import transforms as vision_transforms
from tqdm.auto import tqdm

from core.utils.logger import get_logger, get_train_logger
from core.utils.seed import set_seed as set_global_seed
from .experiment_manager import ExperimentManager, MonitorConfig
from .metrics import classification_metrics, top1_accuracy
from .optim import create_optimizer


_STEP_DEBUG_ENV = os.environ.get("TVB_TRAIN_STEP_DEBUG", "").strip().lower() in {"1", "true", "yes"}
_STEP_DEBUG_ENABLED = _STEP_DEBUG_ENV


def _log_rank(message: str) -> None:
    """분산 환경에서 rank별 디버그 로그를 출력한다."""

    if not _STEP_DEBUG_ENABLED:
        return

    if dist.is_available() and dist.is_initialized():
        try:
            rank = dist.get_rank()
        except Exception:
            rank = -1
    else:
        rank = 0
    timestamp = datetime.datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] rank={rank} | {message}", flush=True)


def _get_env_limit(name: str) -> Optional[int]:
    """환경 변수에서 양의 정수 제한 값을 읽어 반환한다."""

    value = os.environ.get(name, "").strip()
    if not value:
        return None
    try:
        parsed = int(value)
    except ValueError:
        return None
    return parsed if parsed > 0 else None


def _log_epoch_event(logger: logging.Logger, epoch: int, split: str, metrics: Dict[str, Any]) -> None:
    """JSONL 이벤트 로그 파일에 에폭 단위 지표를 기록한다."""

    payload: Dict[str, Any] = {
        "event": "epoch_metrics",
        "epoch": int(epoch),
        "split": split,
        "loss": float(metrics.get("loss", 0.0) or 0.0),
        "acc": float(metrics.get("acc", 0.0) or 0.0),
    }
    if "auc" in metrics:
        try:
            payload["auc"] = float(metrics.get("auc", 0.0) or 0.0)
        except (TypeError, ValueError):
            pass
    if "ece" in metrics:
        try:
            payload["ece"] = float(metrics.get("ece", 0.0) or 0.0)
        except (TypeError, ValueError):
            pass
    logger.info(json.dumps(payload, ensure_ascii=False))

@dataclass
class TrainCfg:
    epochs: int
    lr: float
    weight_decay: float
    optimizer: str
    scheduler: str
    warmup_epochs: Optional[int]
    sched_factor: Optional[float]
    sched_patience: Optional[int]
    sched_min_lr: Optional[float]
    sched_monitor: Optional[str]
    sched_mode: Optional[str]
    mixed_precision: Optional[str]
    grad_accum_steps: int
    log_interval: int
    criterion: str
    label_smoothing: float
    max_grad_norm: float
    patch_chunk_size: int
    partial_epochs: Optional[int]
    full_epochs: Optional[int]
    partial_steps: Optional[int]
    full_steps: Optional[int]
    seed: Optional[int]
    early_stop: bool
    early_patience: int
    early_monitor: str
    early_mode: str
    ckpt_monitor: str
    ckpt_mode: str
    step_debug_logging: bool

    def __post_init__(self) -> None:
        required = {
            "epochs": self.epochs,
            "lr": self.lr,
            "weight_decay": self.weight_decay,
            "optimizer": self.optimizer,
            "scheduler": self.scheduler,
            "grad_accum_steps": self.grad_accum_steps,
            "log_interval": self.log_interval,
            "criterion": self.criterion,
            "max_grad_norm": self.max_grad_norm,
            "patch_chunk_size": self.patch_chunk_size,
            "early_patience": self.early_patience,
            "early_monitor": self.early_monitor,
            "early_mode": self.early_mode,
            "ckpt_monitor": self.ckpt_monitor,
            "ckpt_mode": self.ckpt_mode,
        }
        missing = [key for key, value in required.items() if value is None]
        if missing:
            raise ValueError(f"Missing required trainer fields: {', '.join(missing)}")


class Trainer:
    """Accelerate 기반 학습 관리자."""

    @staticmethod
    def _loader_summary(loader: Optional[DataLoader]) -> str:
        if loader is None:
            return "<none>"
        prefetch = getattr(loader, "prefetch_factor", None)
        try:
            batch_size = loader.batch_size
        except AttributeError:
            batch_size = getattr(loader, "batch_sampler", None)
        return (
            "batch_size={} num_workers={} prefetch_factor={} pin_memory={} persistent_workers={}".format(
                getattr(loader, "batch_size", batch_size),
                getattr(loader, "num_workers", "n/a"),
                prefetch,
                getattr(loader, "pin_memory", "n/a"),
                getattr(loader, "persistent_workers", "n/a"),
            )
        )

    def _prepare_train_loader(self, loader: DataLoader) -> DataLoader:
        prepared = self.accel.prepare_data_loader(loader)
        if self.accel.is_main_process:
            self.system_logger.info(
                "DataLoader 설정(train/after prepare) - %s",
                self._loader_summary(prepared),
            )
        return prepared

    def _prepare_val_loader(self, loader: DataLoader) -> DataLoader:
        prepared = self.accel.prepare_data_loader(loader)
        if self.accel.is_main_process:
            self.system_logger.info(
                "DataLoader 설정(val/after prepare) - %s",
                self._loader_summary(prepared),
            )
        return prepared

    def _release_train_loader(self) -> None:
        if self.train_loader is None:
            return
        try:
            self.train_loader = None
        finally:
            gc.collect()

    def _release_val_loader(self, loader: Optional[DataLoader]) -> None:
        if loader is None:
            return
        try:
            pass
        finally:
            gc.collect()

    def _build_val_loader(self) -> Optional[DataLoader]:
        if self.val_loader_factory is None:
            return None
        loader = self.val_loader_factory()
        if loader is None:
            return None
        return self._prepare_val_loader(loader)

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        train_loader_factory: Callable[[], DataLoader],
        val_loader_factory: Optional[Callable[[], Optional[DataLoader]]],
        out_dir: str,
        cfg: TrainCfg,
        experiment: Optional[ExperimentManager] = None,
        accelerator: Optional[Accelerator] = None,
        inference_cfg: Optional[Dict[str, object]] = None,
        inference_transform: Optional[torch.nn.Module] = None,
        train_augment: Optional[Callable[[Image.Image], Image.Image]] = None,
    ) -> None:
        self.model = model
        self.train_loader_factory = train_loader_factory
        self.val_loader_factory = val_loader_factory
        self.train_loader = train_loader
        self.cfg = cfg
        self.inference_cfg = inference_cfg or {}
        self.inference_transform = inference_transform
        self.train_augment = train_augment
        self._to_pil = vision_transforms.ToPILImage()
        self.system_logger = get_logger(__name__)
        self.train_logger = get_train_logger()
        ddp_kwargs = DistributedDataParallelKwargs(
            broadcast_buffers=False,
            find_unused_parameters=True,
            gradient_as_bucket_view=True,
            static_graph=False,
        )

        self.accel = accelerator or Accelerator(
            mixed_precision=cfg.mixed_precision or "no",
            gradient_accumulation_steps=cfg.grad_accum_steps,
            kwargs_handlers=[ddp_kwargs],
        )
        try:
            setattr(self.accel, "even_batches", False)
        except Exception:
            pass

        global _STEP_DEBUG_ENABLED

        _STEP_DEBUG_ENABLED = bool(cfg.step_debug_logging) or _STEP_DEBUG_ENV
        self.patch_chunk_size = max(1, int(cfg.patch_chunk_size or 8))

        if self.inference_transform is None:
            raise ValueError("Trainer requires inference_transform for training pipeline.")

        monitor = MonitorConfig(key=cfg.ckpt_monitor, mode=cfg.ckpt_mode)
        self.experiment = experiment or ExperimentManager(Path(out_dir), monitor=monitor)
        self.out_dir = Path(self.experiment.root)

        if self.accel.is_main_process:
            self.system_logger.info("검증 모드: standard (서비스 파이프라인 비활성화)")

        if self.accel.is_main_process:
            self.system_logger.info(
                "DataLoader 설정(train/before prepare) - %s",
                self._loader_summary(train_loader),
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

        prepare_items: List[Any] = [self.model, self.optimizer]
        has_scheduler = self.scheduler is not None
        if has_scheduler:
            prepare_items.append(self.scheduler)
        prepared = self.accel.prepare(*prepare_items)
        self.model = prepared[0]
        self.optimizer = prepared[1]
        if has_scheduler:
            self.scheduler = prepared[2]
        self.train_loader = self._prepare_train_loader(train_loader)

        wrapper_chain: List[str] = []
        cursor = self.model
        visited: set[int] = set()
        while id(cursor) not in visited:
            visited.add(id(cursor))
            wrapper_chain.append(type(cursor).__name__)
            next_cursor = getattr(cursor, "module", None)
            if next_cursor is None or next_cursor is cursor:
                break
            cursor = next_cursor

        broadcast_value: Optional[object] = None
        find_unused_value: Optional[object] = None
        cursor = self.model
        while cursor is not None:
            if broadcast_value is None and hasattr(cursor, "broadcast_buffers"):
                broadcast_value = getattr(cursor, "broadcast_buffers")
                setattr(cursor, "broadcast_buffers", False)
            if find_unused_value is None and hasattr(cursor, "find_unused_parameters"):
                find_unused_value = getattr(cursor, "find_unused_parameters")
                setattr(cursor, "find_unused_parameters", True)
            next_cursor = getattr(cursor, "module", None)
            if next_cursor is None or next_cursor is cursor:
                break
            cursor = next_cursor

        self.system_logger.info("DDP wrapper chain: %s", " -> ".join(wrapper_chain))
        self.system_logger.info(
            "DDP 설정 - broadcast_buffers(before)=%s, find_unused_parameters(before)=%s",
            str(broadcast_value),
            str(find_unused_value),
        )

        final_bcast: Optional[object] = None
        final_find: Optional[object] = None
        cursor = self.model
        visited.clear()
        while id(cursor) not in visited:
            visited.add(id(cursor))
            if hasattr(cursor, "broadcast_buffers"):
                final_bcast = getattr(cursor, "broadcast_buffers")
            if hasattr(cursor, "find_unused_parameters"):
                final_find = getattr(cursor, "find_unused_parameters")
            next_cursor = getattr(cursor, "module", None)
            if next_cursor is None or next_cursor is cursor:
                break
            cursor = next_cursor

        self.system_logger.info(
            "DDP 설정 적용 후 - broadcast_buffers=%s, find_unused_parameters=%s",
            str(final_bcast),
            str(final_find),
        )

        if self.accel.is_main_process:
            self.system_logger.info(
                "DataLoader 설정(train/after prepare) - %s",
                self._loader_summary(self.train_loader),
            )


    def _forward_patch_tensor(self, patches: torch.Tensor) -> torch.Tensor:
        """멀티패치 텐서를 chunk 단위로 모델에 통과시킨다."""

        chunk_size = max(1, int(self.patch_chunk_size))
        if patches.size(0) <= chunk_size:
            batch = patches.to(self.accel.device, non_blocking=True)
            with self.accel.autocast():
                return self.model(batch)

        logits_list: List[torch.Tensor] = []
        stream = torch.cuda.Stream(device=self.accel.device) if self.accel.device.type == "cuda" else None
        events: List[torch.cuda.Event] = []
        for chunk in patches.split(chunk_size):
            if chunk.numel() == 0:
                continue
            batch = chunk.to(self.accel.device, non_blocking=True)
            if stream is not None:
                with torch.cuda.stream(stream):
                    with self.accel.autocast():
                        logits = self.model(batch)
                done = torch.cuda.Event()
                done.record(stream)
                events.append(done)
                logits_list.append(logits)
            else:
                with self.accel.autocast():
                    logits_list.append(self.model(batch))
        if stream is not None:
            for event in events:
                event.synchronize()
        if not logits_list:
            batch = patches.to(self.accel.device, non_blocking=True)
            with self.accel.autocast():
                return self.model(batch)
        return torch.cat(logits_list, dim=0)

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

        coverage_sum_y = 0.0
        coverage_sum_y_sq = 0.0
        coverage_count = 0
        complexity_sum = 0.0
        complexity_max = 0.0
        quadrant_counts = {"top": 0, "bottom": 0, "left": 0, "right": 0}

        def _update_patch_stats(meta: Any) -> None:
            nonlocal coverage_sum_y, coverage_sum_y_sq, coverage_count, complexity_sum, complexity_max, quadrant_counts
            if not isinstance(meta, dict):
                return
            cx = float(meta.get("center_x", 0.5))
            cy = float(meta.get("center_y", 0.5))
            complexity = float(meta.get("complexity", 0.0))
            coverage_sum_y += cy
            coverage_sum_y_sq += cy * cy
            coverage_count += 1
            complexity_sum += complexity
            if complexity > complexity_max:
                complexity_max = complexity
            if cy < 0.5:
                quadrant_counts["top"] += 1
            else:
                quadrant_counts["bottom"] += 1
            if cx < 0.5:
                quadrant_counts["left"] += 1
            else:
                quadrant_counts["right"] += 1

        actual_train_steps = len(self.train_loader)
        if steps_limit > 0:
            total_steps = min(steps_limit, actual_train_steps)
        else:
            total_steps = actual_train_steps
        pbar = None
        if self.accel.is_main_process:
            pbar = tqdm(
                total=total_steps,
                desc=f"Epoch {epoch}",
                leave=True,
                dynamic_ncols=True,
            )

        for step, batch in enumerate(self.train_loader, start=1):
            _log_rank(f"epoch={epoch} step={step} start")
            if not batch:
                continue

            batch_loss_total = 0.0
            batch_correct = 0.0
            batch_samples = 0

            with self.accel.accumulate(self.model):
                def _normalize_target(value: Any) -> int:
                    if isinstance(value, (list, tuple)) and value:
                        return _normalize_target(value[0])
                    if isinstance(value, torch.Tensor):
                        if value.ndim == 0:
                            return int(value.item())
                        return int(value.view(-1)[0].item())
                    return int(value)

                tensors_for_batch: List[torch.Tensor] = []
                targets_list: List[int] = []

                iterable_batch = batch if isinstance(batch, (list, tuple)) else [batch]

                for sample in iterable_batch:
                    if sample is None:
                        continue

                    primary = sample
                    target_idx = 0
                    sample_metadata: Optional[Sequence[Any]] = None
                    if isinstance(sample, (tuple, list)) and len(sample) > 0:
                        primary = sample[0]
                        if len(sample) > 1:
                            candidate_meta = sample[1]
                            if isinstance(candidate_meta, (list, tuple)):
                                sample_metadata = candidate_meta
                            try:
                                target_idx = _normalize_target(sample[-1])
                            except (TypeError, ValueError):
                                continue

                    tensor = None
                    if isinstance(primary, torch.Tensor):
                        if primary.dim() == 4:
                            for patch_idx, patch in enumerate(primary):
                                if not isinstance(patch, torch.Tensor) or patch.dim() != 3:
                                    continue
                                tensors_for_batch.append(patch.unsqueeze(0))
                                targets_list.append(target_idx)
                                if (
                                    sample_metadata is not None
                                    and 0 <= patch_idx < len(sample_metadata)
                                ):
                                    _update_patch_stats(sample_metadata[patch_idx])
                            continue
                        elif primary.dim() == 3:
                            tensor = primary
                        else:
                            continue
                    else:
                        image = primary
                        if isinstance(image, torch.Tensor):
                            if image.dim() == 3:
                                image = self._to_pil(image.cpu())
                            elif image.dim() == 4:
                                image = self._to_pil(image[0].cpu())
                            else:
                                continue
                        if not isinstance(image, Image.Image):
                            continue

                        if self.train_augment is not None:
                            augmented = self.train_augment(image)
                            if isinstance(augmented, Image.Image):
                                image = augmented
                            elif isinstance(augmented, torch.Tensor) and augmented.dim() == 3:
                                image = self._to_pil(augmented.cpu())
                        tensor = self.inference_transform(image)

                    if not isinstance(tensor, torch.Tensor) or tensor.dim() != 3:
                        continue

                    tensors_for_batch.append(tensor.unsqueeze(0))
                    targets_list.append(target_idx)
                    if sample_metadata is not None and len(sample_metadata) > 0:
                        _update_patch_stats(sample_metadata[0])

                if not tensors_for_batch:
                    continue

                batch_tensor = torch.cat(tensors_for_batch, dim=0)
                _log_rank(f"epoch={epoch} step={step} after_batch_tensor")
                targets_tensor = torch.tensor(
                    targets_list, device=self.accel.device, dtype=torch.long
                )

                _log_rank(f"epoch={epoch} step={step} before_forward")
                logits = self._forward_patch_tensor(batch_tensor)
                _log_rank(f"epoch={epoch} step={step} after_forward")
                with self.accel.autocast():
                    loss = self.criterion(logits, targets_tensor)
                _log_rank(f"epoch={epoch} step={step} after_loss")

                self.accel.backward(loss)
                _log_rank(f"epoch={epoch} step={step} after_backward")
                try:
                    self.accel.wait_for_everyone()
                except Exception:
                    _log_rank(f"epoch={epoch} step={step} barrier_failed")
                    raise

                loss_value = float(loss.detach().item())
                current_samples = targets_tensor.size(0)
                probs = torch.softmax(logits.detach(), dim=1)
                preds = torch.argmax(probs, dim=1)

                batch_loss_total += loss_value * current_samples
                batch_samples += current_samples
                batch_correct += float((preds == targets_tensor).sum().item())

                if self.accel.sync_gradients and self.cfg.max_grad_norm > 0:
                    self.accel.clip_grad_norm_(self.model.parameters(), max_norm=self.cfg.max_grad_norm)
                self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)

            if batch_samples == 0:
                continue

            total_loss += batch_loss_total
            total_count += batch_samples
            total_acc += batch_correct

            if pbar is not None:
                avg_loss = total_loss / max(1, total_count)
                avg_acc = total_acc / max(1, total_count)
                pbar.set_postfix(loss=f"{avg_loss:.4f}", acc=f"{avg_acc:.4f}")
                pbar.update(1)

            steps_processed += 1

            if steps_limit > 0 and steps_processed >= steps_limit:
                break

        if self.accel.is_main_process and coverage_count > 0:
            mean_y = coverage_sum_y / coverage_count
            variance_y = max(0.0, (coverage_sum_y_sq / coverage_count) - (mean_y * mean_y))
            std_y = math.sqrt(variance_y)
            complexity_mean = complexity_sum / coverage_count if coverage_count > 0 else 0.0
            payload = {
                "event": "patch_coverage",
                "epoch": int(epoch),
                "patch_count": int(coverage_count),
                "patch_center_y_mean": float(mean_y),
                "patch_center_y_std": float(std_y),
                "quadrant_counts": {
                    "top": int(quadrant_counts["top"]),
                    "bottom": int(quadrant_counts["bottom"]),
                    "left": int(quadrant_counts["left"]),
                    "right": int(quadrant_counts["right"]),
                },
                "complexity_mean": float(complexity_mean),
                "complexity_max": float(complexity_max),
            }
            self.train_logger.info(json.dumps(payload, ensure_ascii=False))

        if pbar is not None:
            pbar.close()

        return {"loss": total_loss / max(1, total_count), "acc": total_acc / max(1, total_count)}

    @torch.no_grad()
    def _validate_standard(self) -> Dict[str, float]:
        val_loader = self._build_val_loader()
        if val_loader is None:
            return {"loss": 0.0, "acc": 0.0}
        self.model.eval()
        total_loss = 0.0
        total_acc = 0.0
        total_count = 0.0
        logits_collector: List[torch.Tensor] = []
        targets_collector: List[torch.Tensor] = []
        env_val_limit = _get_env_limit("TVB_VAL_MAX_STEPS")
        steps_limit = env_val_limit or 0
        try:
            total_batches = len(val_loader)
        except TypeError:
            total_batches = 0
        total_steps = min(total_batches, steps_limit) if steps_limit > 0 else total_batches
        if env_val_limit is not None and self.system_logger is not None and self.accel.is_main_process:
            self.system_logger.info(
                "검증 배치를 최대 %d회로 제한합니다 (전체=%d)",
                total_steps,
                total_batches,
            )

        pbar = None
        if self.accel.is_main_process:
            pbar = tqdm(
                total=total_steps,
                desc="Validation",
                leave=True,
                dynamic_ncols=True,
            )

        steps_processed = 0
        try:
            for batch in val_loader:
                if steps_limit > 0 and steps_processed >= steps_limit:
                    break
                if batch is None:
                    continue
                if isinstance(batch, dict):
                    images = batch.get("images") or batch.get("image")
                    targets = batch.get("targets") or batch.get("target")
                    if images is None or targets is None:
                        continue
                elif isinstance(batch, (list, tuple)) and len(batch) >= 2:
                    images, targets = batch[0], batch[1]
                else:
                    continue

                if isinstance(images, (list, tuple)):
                    tensor_list = [img for img in images if isinstance(img, torch.Tensor)]
                    if not tensor_list:
                        continue
                    images = torch.stack(tensor_list, dim=0)
                if not isinstance(images, torch.Tensor):
                    continue

                if isinstance(targets, (list, tuple)):
                    targets = torch.tensor(targets, dtype=torch.long)
                elif not isinstance(targets, torch.Tensor):
                    targets = torch.tensor(targets, dtype=torch.long)

                images = images.to(self.accel.device, non_blocking=True)
                targets = targets.to(self.accel.device, non_blocking=True)

                if images.ndim == 3:
                    images = images.unsqueeze(0)
                if images.ndim != 4:
                    continue

                with self.accel.autocast():
                    logits = self.model(images)
                    loss_tensor = torch.nn.functional.cross_entropy(logits, targets)

                batch_loss = float(loss_tensor.detach().item())
                preds = torch.argmax(logits, dim=1)
                correct = float((preds == targets).sum().item())
                batch_size = float(targets.size(0))

                total_loss += batch_loss * batch_size
                total_acc += correct
                total_count += batch_size

                logits_collector.append(logits.detach().cpu())
                targets_collector.append(targets.detach().cpu())

                if pbar is not None:
                    avg_loss = total_loss / max(1.0, total_count)
                    avg_acc = total_acc / max(1.0, total_count)
                    pbar.set_postfix(loss=f"{avg_loss:.4f}", acc=f"{avg_acc:.4f}")
                    pbar.update(1)
                steps_processed += 1

        finally:
            if pbar is not None:
                pbar.close()
            self._release_val_loader(val_loader)

        summary_tensor = torch.tensor(
            [total_loss, total_acc, total_count],
            device=self.accel.device,
            dtype=torch.float64,
        )
        summary_tensor = self.accel.reduce(summary_tensor, reduction="sum")
        total_loss = float(summary_tensor[0].item())
        total_acc = float(summary_tensor[1].item())
        total_count = float(summary_tensor[2].item())

        metrics = {"loss": total_loss / max(1.0, total_count), "acc": total_acc / max(1.0, total_count)}
        logits_cat = torch.cat(logits_collector) if logits_collector else torch.empty(0, 0)
        targets_cat = torch.cat(targets_collector).long() if targets_collector else torch.empty(0, dtype=torch.long)
        if logits_cat.dim() == 2 and logits_cat.numel() > 0:
            num_classes = logits_cat.size(1)
        else:
            num_classes = int(getattr(self.model, "num_classes", 2) or 2)
            logits_cat = logits_cat.view(-1, num_classes)

        logits_device = logits_cat.to(self.accel.device, dtype=torch.float32)
        targets_device = targets_cat.to(self.accel.device, dtype=torch.long)

        padded_logits = self.accel.pad_across_processes(
            logits_device,
            dim=0,
            pad_index=0.0,
        )
        padded_targets = self.accel.pad_across_processes(
            targets_device,
            dim=0,
            pad_index=-1,
        )

        local_len_tensor = torch.tensor([logits_device.shape[0]], device=self.accel.device, dtype=torch.long)
        gathered_lengths = self.accel.gather(local_len_tensor)
        gathered_logits = self.accel.gather(padded_logits)
        gathered_targets = self.accel.gather(padded_targets)

        all_lengths: list[int] = []
        if gathered_lengths is not None:
            all_lengths = [int(val) for val in gathered_lengths.long().cpu().tolist()]
        else:
            all_lengths = [int(local_len_tensor.item())]

        merged_logits: list[torch.Tensor] = []
        merged_targets: list[torch.Tensor] = []
        if gathered_logits is not None and gathered_targets is not None:
            max_len = padded_logits.shape[0]
            gathered_logits_cpu = gathered_logits.detach().cpu()
            gathered_targets_cpu = gathered_targets.detach().cpu()
            for idx, length in enumerate(all_lengths):
                if length <= 0:
                    continue
                start = idx * max_len
                end = start + max_len
                logits_slice = gathered_logits_cpu[start:end][:length]
                targets_slice = gathered_targets_cpu[start:end][:length]
                if logits_slice.numel() == 0 or targets_slice.numel() == 0:
                    continue
                valid_mask = targets_slice >= 0
                if not torch.any(valid_mask):
                    continue
                merged_logits.append(logits_slice[valid_mask])
                merged_targets.append(targets_slice[valid_mask])

        if merged_logits and merged_targets:
            logits_global = torch.cat(merged_logits, dim=0)
            targets_global = torch.cat(merged_targets, dim=0).long()
            if logits_global.numel() > 0 and targets_global.numel() > 0:
                bundle = classification_metrics(logits_global, targets_global)
                bundle.loss = metrics["loss"]
                bundle.acc = metrics["acc"]
                metrics.update(bundle.as_dict())
        metrics.setdefault("val_loss", metrics["loss"])
        metrics.setdefault("val_acc", metrics["acc"])
        return metrics

    @torch.no_grad()
    def _validate(self) -> Dict[str, float]:
        return self._validate_standard()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def validate_only(self) -> Dict[str, float]:
        """훈련 없이 검증 파이프라인만 실행한다."""

        self._release_train_loader()
        metrics = self._validate()
        if self.accel.is_main_process:
            self.train_logger.info(
                (
                    "검증 전용 실행 - val_loss=%.4f val_acc=%.4f "
                    "val_f1=%.4f val_auc=%.4f val_ece=%.4f val_tpr@1%%=%.4f"
                ),
                metrics.get("val_loss", metrics.get("loss", 0.0)),
                metrics.get("val_acc", metrics.get("acc", 0.0)),
                metrics.get("f1", 0.0),
                metrics.get("auc", 0.0),
                metrics.get("ece", 0.0),
                metrics.get("tpr@fpr=1%", 0.0),
            )
        return metrics

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
            if self.train_loader is None:
                self.train_loader = self._prepare_train_loader(self.train_loader_factory())

            max_steps_available = len(self.train_loader)
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

            env_train_limit = _get_env_limit("TVB_TRAIN_MAX_STEPS")
            if env_train_limit is not None:
                steps_limit = max(1, min(env_train_limit, steps_limit))
                if self.accel.is_main_process and epoch == 1:
                    self.system_logger.info(
                        "환경 변수 TVB_TRAIN_MAX_STEPS=%d 적용 - 실제 학습 스텝 %d",
                        env_train_limit,
                        steps_limit,
                    )

            train_metrics = self._train_one_epoch(epoch, steps_limit)
            self._release_train_loader()
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
                (
                    "에폭 %03d 완료 - train_loss=%.4f train_acc=%.4f "
                    "val_loss=%.4f val_acc=%.4f val_f1=%.4f val_auc=%.4f "
                    "val_ece=%.4f val_tpr@1%%=%.4f"
                ),
                epoch,
                train_metrics.get("loss", 0.0),
                train_metrics.get("acc", 0.0),
                val_metrics.get("loss", 0.0),
                val_metrics.get("acc", 0.0),
                val_metrics.get("f1", 0.0),
                val_metrics.get("auc", 0.0),
                val_metrics.get("ece", 0.0),
                val_metrics.get("tpr@fpr=1%", 0.0),
            )

            _log_epoch_event(self.train_logger, epoch, "train", train_metrics)
            _log_epoch_event(self.train_logger, epoch, "val", val_metrics)
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

        try:
            self.accel.wait_for_everyone()
        except Exception:
            pass
        try:
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                torch.distributed.destroy_process_group()
        except Exception:
            pass

        return last_val_metrics
