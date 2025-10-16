from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import hydra
import torch
import torch.distributed as dist
from accelerate import Accelerator
from omegaconf import DictConfig, OmegaConf

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.datasets import build_dataloaders
from core.models.registry import get_model
from core.trainer.engine import TrainCfg, Trainer
from core.trainer.experiment_manager import ExperimentManager, MonitorConfig
from core.utils.logger import get_logger, setup_experiment_loggers
from core.utils.seed import set_seed

logger = get_logger(__name__)


def _to_dict(cfg: Any) -> Dict[str, Any]:
    if isinstance(cfg, DictConfig):
        return OmegaConf.to_container(cfg, resolve=True)  # type: ignore[return-value]
    return cfg


def _build_train_cfg(cfg: DictConfig) -> TrainCfg:
    optimizer_cfg = cfg.optimizer
    scheduler_cfg = cfg.scheduler
    trainer_cfg = cfg.trainer
    early_cfg = trainer_cfg.get("early_stopping", {})
    monitor_cfg = trainer_cfg.get("monitor", {})

    return TrainCfg(
        epochs=trainer_cfg.get("epochs", 10),
        lr=optimizer_cfg.get("lr", 3.0e-4),
        weight_decay=optimizer_cfg.get("weight_decay", 0.05),
        optimizer=optimizer_cfg.get("name", "adamw"),
        scheduler=scheduler_cfg.get("name", "cosine"),
        warmup_epochs=scheduler_cfg.get("warmup_epochs", 0),
        sched_factor=scheduler_cfg.get("factor", 0.5),
        sched_patience=scheduler_cfg.get("patience", 3),
        sched_min_lr=scheduler_cfg.get("min_lr", 1.0e-6),
        sched_monitor=scheduler_cfg.get("monitor", "val_loss"),
        sched_mode=scheduler_cfg.get("mode", "min"),
        mixed_precision=trainer_cfg.get("mixed_precision"),
        grad_accum_steps=trainer_cfg.get("grad_accum_steps", 1),
        log_interval=trainer_cfg.get("log_interval", 50),
        criterion=trainer_cfg.get("criterion", "ce"),
        label_smoothing=float(trainer_cfg.get("label_smoothing", 0.0) or 0.0),
        max_grad_norm=float(trainer_cfg.get("max_grad_norm", 1.0) or 0.0),
        partial_epochs=trainer_cfg.get("partial_epochs"),
        full_epochs=trainer_cfg.get("full_epochs"),
        partial_steps=trainer_cfg.get("partial_steps"),
        full_steps=trainer_cfg.get("full_steps"),
        seed=getattr(cfg.run, "seed", None),
        early_stop=early_cfg.get("enabled", False),
        early_patience=early_cfg.get("patience", 8),
        early_monitor=early_cfg.get("monitor", "val_loss"),
        early_mode=early_cfg.get("mode", "min"),
        ckpt_monitor=monitor_cfg.get("key", "val_loss"),
        ckpt_mode=monitor_cfg.get("mode", "min"),
    )


def _sync_processes(accelerator: Accelerator) -> None:
    """NCCL 경고 없이 프로세스 간 동기화."""

    if accelerator.num_processes <= 1:
        return

    try:
        if dist.is_available() and dist.is_initialized():
            current_device = None
            device_obj = getattr(accelerator, "device", None)
            if isinstance(device_obj, torch.device) and device_obj.type == "cuda":
                current_device = device_obj.index
            if current_device is None and torch.cuda.is_available():
                current_device = torch.cuda.current_device()
            if current_device is not None:
                try:
                    dist.barrier(device_ids=[current_device])
                    return
                except TypeError:
                    pass
            dist.barrier()
            return
    except Exception:
        pass

    accelerator.wait_for_everyone()


def run_training(cfg: DictConfig) -> Path:
    """Hydra DictConfig를 받아 단일 학습을 수행한다."""

    preassigned_device = False
    if torch.cuda.is_available():
        local_rank_env = os.environ.get("LOCAL_RANK")
        if local_rank_env is not None:
            try:
                torch.cuda.set_device(int(local_rank_env))
                preassigned_device = True
            except Exception as exc:
                logger.debug("LOCAL_RANK 기반 torch.cuda.set_device 실패: %s", exc)

    accelerator = Accelerator()
    if torch.cuda.is_available() and not preassigned_device:
        try:
            device = getattr(accelerator, "device", None)
            if isinstance(device, torch.device) and device.type == "cuda":
                torch.cuda.set_device(device)
            else:
                torch.cuda.set_device(accelerator.local_process_index)
        except Exception as exc:
            logger.debug("torch.cuda.set_device 실패: %s", exc)
    set_seed((cfg.run.seed or 0) + accelerator.process_index)
    if cfg.run.seed is not None:
        try:
            accelerator.seed(cfg.run.seed)
        except Exception:
            pass

    experiment_dir = Path(cfg.run.output_dir).expanduser().resolve()
    if accelerator.is_main_process:
        experiment_dir.mkdir(parents=True, exist_ok=True)
    _sync_processes(accelerator)

    logging_cfg = _to_dict(cfg.logging)
    level_name = str(logging_cfg.get("level", "INFO"))
    log_level = getattr(logging, level_name.upper(), logging.INFO)
    if accelerator.is_main_process:
        logging.getLogger().setLevel(log_level)
        logger.setLevel(log_level)

    handlers_cfg = _to_dict(logging_cfg.get("handlers", {}))

    console_enabled = True
    if isinstance(handlers_cfg, dict) and "console" in handlers_cfg:
        console_enabled = bool(handlers_cfg.get("console"))

    file_cfg: Dict[str, Any] = {}
    if isinstance(handlers_cfg, dict):
        file_cfg = _to_dict(handlers_cfg.get("file", {})) or {}

    base_dir = file_cfg.get("dir", "logs")
    train_name = file_cfg.get("train", "train.log")
    system_name = file_cfg.get("system", "system.log")

    if Path(base_dir).is_absolute():
        train_path = Path(base_dir) / train_name
        system_path = Path(base_dir) / system_name
    else:
        base_path = experiment_dir / base_dir
        train_path = base_path / train_name
        system_path = base_path / system_name

    jsonl_cfg: Dict[str, Any] = {}
    jsonl_path: Optional[Path] = None
    if isinstance(handlers_cfg, dict):
        jsonl_cfg = _to_dict(handlers_cfg.get("jsonl", {})) or {}
    if jsonl_cfg.get("enabled", True):
        rel = jsonl_cfg.get("path", "logs/events.jsonl")
        jsonl_path = Path(rel)
        if not jsonl_path.is_absolute():
            jsonl_path = experiment_dir / jsonl_path

    world_size = accelerator.num_processes
    rank = accelerator.process_index
    setup_kwargs = dict(
        exp_dir=experiment_dir,
        level=log_level,
        console=console_enabled,
        train_log=train_path,
        system_log=system_path,
        jsonl_path=jsonl_path,
    )
    if accelerator.is_main_process:
        setup_experiment_loggers(**setup_kwargs)
    _sync_processes(accelerator)

    dataset_cfg = cfg.dataset
    train_loader, val_loader, class_names, val_infer_transform, train_augment = build_dataloaders(
        dataset_cfg,
        world_size=world_size,
        rank=rank,
        seed=cfg.run.seed,
        return_raw_val_images=True,
        return_raw_train_images=True,
    )
    if accelerator.is_main_process:
        logger.info("데이터셋 클래스: %s", class_names)

    params_dict = _to_dict(cfg.model.get("params", {})) or {}
    model = get_model(cfg.model.name, **params_dict)

    train_cfg = _build_train_cfg(cfg)
    monitor = MonitorConfig(key=train_cfg.ckpt_monitor, mode=train_cfg.ckpt_mode)
    manager = ExperimentManager(
        experiment_dir,
        monitor=monitor,
        config=cfg if accelerator.is_main_process else None,
        is_main_process=accelerator.is_main_process,
    )

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        out_dir=str(experiment_dir),
        cfg=train_cfg,
        experiment=manager,
        accelerator=accelerator,
        inference_cfg=_to_dict(getattr(cfg, "inference", {})) or {},
        inference_transform=val_infer_transform,
        train_augment=train_augment,
    )
    final_val_metrics = trainer.fit()

    dataset_info = _to_dict(cfg.dataset)
    optimizer_info = _to_dict(cfg.optimizer)
    model_info = _to_dict(cfg.model)
    summary_row = {
        "timestamp": experiment_dir.name,
        "model": str(model_info.get("name", getattr(cfg.model, "name", "unknown"))),
        "dataset": str(dataset_info.get("name", "unknown")),
        "optimizer": str(optimizer_info.get("name", "unknown")),
        "val_acc": final_val_metrics.get("acc") if isinstance(final_val_metrics, dict) else None,
        "val_loss": final_val_metrics.get("loss") if isinstance(final_val_metrics, dict) else None,
    }
    manager.append_summary(summary_row)

    if accelerator.is_main_process:
        logger.info("학습 종료 - 산출물 경로: %s", experiment_dir)
    _sync_processes(accelerator)
    return experiment_dir


@hydra.main(version_base="1.3", config_path="../core/configs", config_name="defaults")
def main(cfg: DictConfig) -> None:
    run_training(cfg)


if __name__ == "__main__":
    main()
