from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Optional, Sequence, Tuple, List, Any

import math
import os
import torch
import torch.nn as nn
from accelerate import Accelerator
from torch.utils.data import DataLoader
from PIL import Image
from torchvision import transforms as vision_transforms
from tqdm.auto import tqdm

from core.utils.logger import get_logger, get_train_logger
from core.utils.seed import set_seed as set_global_seed
from core.models.multipatch import (
    aggregate_scores,
    compute_patch_weights,
    estimate_priority_regions,
    generate_patches,
)
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
    patch_chunk_size: int = 8
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

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader],
        out_dir: str,
        cfg: TrainCfg,
        experiment: Optional[ExperimentManager] = None,
        accelerator: Optional[Accelerator] = None,
        inference_cfg: Optional[Dict[str, object]] = None,
        inference_transform: Optional[torch.nn.Module] = None,
        train_augment: Optional[Callable[[Image.Image], Image.Image]] = None,
    ) -> None:
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.cfg = cfg
        self.inference_cfg = inference_cfg or {}
        self.inference_transform = inference_transform
        self.train_augment = train_augment
        self._to_pil = vision_transforms.ToPILImage()
        self._init_inference_settings()
        self.patch_chunk_size = max(1, int(cfg.patch_chunk_size or 8))

        monitor = MonitorConfig(key=cfg.ckpt_monitor, mode=cfg.ckpt_mode)
        self.experiment = experiment or ExperimentManager(Path(out_dir), monitor=monitor)
        self.out_dir = Path(self.experiment.root)

        self.system_logger = get_logger(__name__)
        self.train_logger = get_train_logger()

        main_process = str(os.environ.get("LOCAL_RANK", "0")) == "0"
        if main_process:
            self.system_logger.info(
                "DataLoader 설정(train/before prepare) - %s",
                self._loader_summary(train_loader),
            )
            self.system_logger.info(
                "DataLoader 설정(val/before prepare) - %s",
                self._loader_summary(val_loader),
            )

        self.accel = accelerator or Accelerator(
            mixed_precision=cfg.mixed_precision or "no",
            gradient_accumulation_steps=cfg.grad_accum_steps,
        )
        try:
            setattr(self.accel, "even_batches", False)
        except Exception:
            pass
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

        if self.accel.is_main_process:
            self.system_logger.info(
                "DataLoader 설정(train/after prepare) - %s",
                self._loader_summary(self.train_loader),
            )
            self.system_logger.info(
                "DataLoader 설정(val/after prepare) - %s",
                self._loader_summary(self.val_loader),
            )

    def _init_inference_settings(self) -> None:
        cfg = self.inference_cfg or {}
        raw_scales = cfg.get("multiscale") or cfg.get("scales") or (224,)
        if isinstance(raw_scales, (int, float)):
            scales = (int(raw_scales),)
        elif isinstance(raw_scales, (list, tuple)):
            scales = tuple(int(float(s)) for s in raw_scales if s is not None)
            if not scales:
                scales = (224,)
        else:
            scales = (224,)

        raw_cells = (
            cfg.get("cell_sizes")
            or cfg.get("min_cell_sizes")
            or cfg.get("min_cell_size")
            or cfg.get("cell_size")
        )
        if isinstance(raw_cells, (int, float)):
            cell_sizes = tuple([int(raw_cells)] * len(scales))
        elif isinstance(raw_cells, (list, tuple)):
            normalized = [int(float(c)) for c in raw_cells if c is not None]
            if not normalized:
                cell_sizes = scales
            elif len(normalized) == 1 and len(scales) > 1:
                cell_sizes = tuple([normalized[0]] * len(scales))
            elif len(normalized) == len(scales):
                cell_sizes = tuple(normalized)
            else:
                cell_sizes = scales
        else:
            cell_sizes = scales

        try:
            n_patches = int(cfg.get("n_patches") or cfg.get("patches") or 0)
        except (TypeError, ValueError):
            n_patches = 0

        aggregate = str(cfg.get("aggregate") or "mean").lower()
        if aggregate not in {"mean", "max", "quality_weighted"}:
            aggregate = "mean"

        overlap = cfg.get("patch_overlap") or cfg.get("overlap")
        jitter = cfg.get("patch_jitter") or cfg.get("jitter")
        max_patches = cfg.get("max_patches") or cfg.get("patch_limit")

        if self.inference_transform is None:
            raise ValueError(
                "inference_transform must be provided to Trainer to match service inference pipeline."
            )

        self.inference_scales: Tuple[int, ...] = tuple(scales)
        self.inference_cell_sizes: Tuple[int, ...] = tuple(cell_sizes)
        self.inference_n_patches: int = max(0, n_patches)
        self.inference_aggregate: str = aggregate
        self.inference_overlap = overlap
        self.inference_jitter = jitter
        try:
            mp_value = int(max_patches) if max_patches not in (None, "", False) else None
        except (TypeError, ValueError):
            mp_value = None
        if mp_value is not None and mp_value <= 0:
            mp_value = None
        self.inference_max_patches = mp_value

    def _evaluate_validation_image(self, image: Image.Image) -> Tuple[torch.Tensor, torch.Tensor]:
        priority_regions = estimate_priority_regions(image)
        patch_samples = generate_patches(
            image,
            sizes=self.inference_scales,
            n_patches=self.inference_n_patches,
            min_cell_size=self.inference_cell_sizes,
            overlap=self.inference_overlap,
            jitter=self.inference_jitter,
            max_patches=self.inference_max_patches,
            priority_regions=priority_regions,
        )
        if not patch_samples:
            base_probs = torch.tensor([0.5, 0.5], device=self.accel.device, dtype=torch.float32)
            logits_tensor = torch.log(torch.clamp(base_probs, min=1e-8))
            return base_probs, logits_tensor

        tensors = []
        for sample in patch_samples:
            patch_img = sample.image if sample.image.mode == "RGB" else sample.image.convert("RGB")
            tensor = self.inference_transform(patch_img).unsqueeze(0)
            tensors.append(tensor)

        batch_tensor = torch.cat(tensors, dim=0).to(self.accel.device)
        with self.accel.autocast():
            logits = self.model(batch_tensor)
            probs = torch.softmax(logits, dim=1)

        probs_cpu = probs.detach().cpu()
        patch_scores = [
            {"real": float(prob[0]), "ai": float(prob[1])}
            for prob in probs_cpu
        ]
        weights = compute_patch_weights(patch_samples)
        aggregated = aggregate_scores(
            patch_scores,
            method=self.inference_aggregate,
            weights=weights if weights else None,
        )
        probs_tensor = torch.tensor(
            [aggregated["real"], aggregated["ai"]],
            device=self.accel.device,
            dtype=torch.float32,
        )
        probs_tensor = torch.clamp(probs_tensor, min=1e-8)
        logits_tensor = torch.log(probs_tensor)
        return probs_tensor, logits_tensor

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

        total_steps = steps_limit if steps_limit > 0 else len(self.train_loader)
        pbar = None
        if self.accel.is_main_process:
            pbar = tqdm(
                total=total_steps,
                desc=f"Epoch {epoch}",
                leave=False,
                dynamic_ncols=True,
            )

        for step, batch in enumerate(self.train_loader, start=1):
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
                    if isinstance(sample, (tuple, list)) and len(sample) > 0:
                        primary = sample[0]
                        if len(sample) > 1:
                            target_idx = _normalize_target(sample[1])

                    if isinstance(primary, torch.Tensor):
                        if primary.dim() == 4:
                            tensor = primary[0]
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

                if not tensors_for_batch:
                    continue

                batch_tensor = torch.cat(tensors_for_batch, dim=0).to(
                    self.accel.device, non_blocking=True
                )
                targets_tensor = torch.tensor(
                    targets_list, device=self.accel.device, dtype=torch.long
                )

                with self.accel.autocast():
                    logits = self.model(batch_tensor)
                    loss = self.criterion(logits, targets_tensor)

                self.accel.backward(loss)

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

            if (
                (step % max(1, self.cfg.log_interval)) == 0
                and self.accel.is_main_process
                and total_count > 0
            ):
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

        if pbar is not None:
            pbar.close()

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
        eps = 1e-8
        total_steps = len(self.val_loader)
        pbar = None
        if self.accel.is_main_process:
            pbar = tqdm(
                total=total_steps,
                desc="Validation",
                leave=False,
                dynamic_ncols=True,
            )
        for batch in self.val_loader:
            if not batch:
                continue
            for image, target in batch:
                if isinstance(image, torch.Tensor):
                    image = self._to_pil(image.cpu())
                elif not isinstance(image, Image.Image):
                    raise TypeError(f"Unsupported validation sample type: {type(image)}")
                probs_tensor, logits_tensor = self._evaluate_validation_image(image)
                if isinstance(target, torch.Tensor):
                    target_idx = int(target.item())
                else:
                    target_idx = int(target)

                loss = -torch.log(torch.clamp(probs_tensor[target_idx], min=eps))
                total_loss += float(loss.item())
                pred_idx = int(torch.argmax(probs_tensor).item())
                if pred_idx == target_idx:
                    total_acc += 1.0
                total_count += 1

                logits_collector.append(self.accel.gather(logits_tensor.unsqueeze(0)))
                target_tensor = torch.tensor([target_idx], device=self.accel.device, dtype=torch.long)
                targets_collector.append(self.accel.gather(target_tensor))

            if pbar is not None:
                avg_loss = total_loss / max(1, total_count)
                avg_acc = total_acc / max(1, total_count)
                pbar.set_postfix(loss=f"{avg_loss:.4f}", acc=f"{avg_acc:.4f}")
                pbar.update(1)

        if pbar is not None:
            pbar.close()

        metrics = {"loss": total_loss / max(1, total_count), "acc": total_acc / max(1, total_count)}
        if logits_collector:
            logits_cat = torch.cat(logits_collector)
            targets_cat = torch.cat(targets_collector)
            bundle = classification_metrics(logits_cat, targets_cat)
            bundle.loss = metrics["loss"]
            bundle.acc = metrics["acc"]
            metrics.update(bundle.as_dict())
        metrics.setdefault("val_loss", metrics["loss"])
        metrics.setdefault("val_acc", metrics["acc"])
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
