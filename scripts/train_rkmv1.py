"""RKMv1 학습 전용 스크립트."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional

import hydra
import torch
import torch.distributed as dist
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs
from omegaconf import DictConfig, OmegaConf

from core.datasets import build_dataloaders
from core.models.registry import get_model
from core.trainer.engine import TrainCfg
from core.trainer.experiment_manager import ExperimentManager, MonitorConfig
from core.trainer.rkmv1_engine import RKMv1Trainer
from core.utils.logger import get_logger, setup_experiment_loggers
from core.utils.seed import set_seed

LOGGER = get_logger(__name__)

_NCCL_DEFAULTS: Dict[str, str] = {
    "TORCH_NCCL_ASYNC_ERROR_HANDLING": "1",
    "TORCH_NCCL_BLOCKING_WAIT": "1",
    "NCCL_TIMEOUT": "1800",
}


def _apply_nccl_defaults() -> None:
    for key, value in _NCCL_DEFAULTS.items():
        os.environ.setdefault(key, value)


def _to_dict(cfg: Any) -> Dict[str, Any]:
    if isinstance(cfg, DictConfig):
        return OmegaConf.to_container(cfg, resolve=True)  # type: ignore[return-value]
    return dict(cfg)


def _build_train_cfg(c: DictConfig) -> TrainCfg:
    trainer_cfg = OmegaConf.to_container(c.trainer, resolve=True)
    optimizer_cfg = OmegaConf.to_container(c.optimizer, resolve=True)
    scheduler_cfg = OmegaConf.to_container(c.scheduler, resolve=True)

    early_cfg = trainer_cfg["early_stopping"]
    monitor_cfg = trainer_cfg["monitor"]

    return TrainCfg(
        epochs=int(trainer_cfg["epochs"]),
        lr=float(optimizer_cfg["lr"]),
        weight_decay=float(optimizer_cfg["weight_decay"]),
        optimizer=str(optimizer_cfg["name"]),
        scheduler=str(scheduler_cfg["name"]),
        warmup_epochs=scheduler_cfg.get("warmup_epochs"),
        sched_factor=scheduler_cfg.get("factor"),
        sched_patience=scheduler_cfg.get("patience"),
        sched_min_lr=scheduler_cfg.get("min_lr"),
        sched_monitor=scheduler_cfg.get("monitor"),
        sched_mode=scheduler_cfg.get("mode"),
        mixed_precision=trainer_cfg["mixed_precision"],
        grad_accum_steps=int(trainer_cfg["grad_accum_steps"]),
        log_interval=int(trainer_cfg["log_interval"]),
        criterion="custom",
        label_smoothing=float(trainer_cfg.get("label_smoothing", 0.0)),
        max_grad_norm=float(trainer_cfg.get("max_grad_norm", 0.0)),
        patch_chunk_size=int(trainer_cfg.get("patch_chunk_size", 1)),
        partial_epochs=trainer_cfg.get("partial_epochs"),
        full_epochs=trainer_cfg.get("full_epochs"),
        partial_steps=trainer_cfg.get("partial_steps"),
        full_steps=trainer_cfg.get("full_steps"),
        max_train_steps=trainer_cfg.get("train_max_steps"),
        max_val_steps=trainer_cfg.get("val_max_steps"),
        step_debug_logging=bool(trainer_cfg.get("step_debug_logging", False)),
        seed=getattr(c.run, "seed", None),
        early_stop=bool(early_cfg["enabled"]),
        early_patience=int(early_cfg["patience"]),
        early_monitor=str(early_cfg["monitor"]),
        early_mode=str(early_cfg["mode"]),
        ckpt_monitor=str(monitor_cfg["key"]),
        ckpt_mode=str(monitor_cfg["mode"]),
    )


def _init_distributed(accelerator: Accelerator) -> None:
    if accelerator.num_processes <= 1:
        return
    if dist.is_initialized():
        return
    backend = os.environ.get("ACCELERATE_PROCESS_GROUP_BACKEND")
    if not backend:
        backend = "nccl" if torch.cuda.is_available() else "gloo"
    dist.init_process_group(backend=backend)


def _prepare_logging(cfg: DictConfig, accelerator: Accelerator, exp_dir: Path) -> None:
    logging_cfg = cfg.logging
    level_name = str(logging_cfg.level)
    log_level = getattr(logging, level_name.upper(), logging.INFO)
    if accelerator.is_main_process:
        logging.getLogger().setLevel(log_level)
        LOGGER.setLevel(log_level)

    handlers_cfg = logging_cfg.handlers
    console_enabled = bool(handlers_cfg.console)
    file_cfg = handlers_cfg.file
    log_path: Optional[Path] = None
    json_format = True
    if file_cfg.enabled:
        log_path = Path(str(file_cfg.path))
        json_format = str(getattr(file_cfg, "format", "json")).lower() != "text"

    setup_kwargs = dict(
        exp_dir=exp_dir,
        level=log_level,
        console=console_enabled,
        log_path=log_path,
        json_format=json_format,
    )
    if accelerator.is_main_process:
        setup_experiment_loggers(**setup_kwargs)
    accelerator.wait_for_everyone()


def _resolve_experiment_dir(cfg: DictConfig, accelerator: Accelerator) -> Path:
    exp_dir = Path(cfg.run.output_dir).expanduser().resolve()
    if accelerator.is_main_process:
        exp_dir.mkdir(parents=True, exist_ok=True)
    accelerator.wait_for_everyone()
    return exp_dir


def _configure_device() -> None:
    _apply_nccl_defaults()
    if not os.environ.get("MASTER_ADDR"):
        os.environ["MASTER_ADDR"] = "127.0.0.1"
    local_rank = os.environ.get("LOCAL_RANK")
    if torch.cuda.is_available() and local_rank is not None:
        device_id = int(local_rank)
        torch.cuda.set_device(device_id)


def _maybe_reduce_logging(accelerator: Accelerator) -> None:
    if accelerator.is_main_process:
        return
    for name in ("timm", "huggingface_hub", "gravifox.system", "gravifox.train"):
        logging.getLogger(name).setLevel(logging.ERROR)


def _build_model(cfg: DictConfig) -> torch.nn.Module:
    params = _to_dict(cfg.model.get("params", {}))
    model = get_model(cfg.model.name, **params)
    return model


def _train(cfg: DictConfig) -> Path:
    _configure_device()
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(kwargs_handlers=[ddp_kwargs])
    _init_distributed(accelerator)
    _maybe_reduce_logging(accelerator)

    set_seed((cfg.run.seed or 0) + accelerator.process_index)

    train_cfg = _build_train_cfg(cfg)
    exp_dir = _resolve_experiment_dir(cfg, accelerator)
    _prepare_logging(cfg, accelerator, exp_dir)

    dataset_cfg = cfg.dataset
    eval_only = bool(getattr(cfg.run, "eval_only", False) or getattr(cfg.run, "validate_only", False))

    train_loader, val_loader, class_names, _, _ = build_dataloaders(
        dataset_cfg,
        build_train=not eval_only,
        build_val=True,
        world_size=accelerator.num_processes,
        rank=accelerator.process_index,
        seed=cfg.run.seed,
        shuffle_train=not eval_only,
    )
    if val_loader is None:
        raise RuntimeError("검증 DataLoader가 필요합니다.")
    if accelerator.is_main_process:
        LOGGER.info("클래스 목록: %s", class_names)

    model = _build_model(cfg)
    monitor = MonitorConfig(key=train_cfg.ckpt_monitor, mode=train_cfg.ckpt_mode)
    manager = ExperimentManager(
        exp_dir,
        monitor=monitor,
        config=cfg if accelerator.is_main_process else None,
        is_main_process=accelerator.is_main_process,
    )

    trainer = RKMv1Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        cfg=train_cfg,
        experiment=manager,
        accelerator=accelerator,
    )
    if eval_only:
        metrics = trainer.validate_only()
    else:
        metrics = trainer.fit()

    summary_row = {
        "timestamp": exp_dir.name,
        "model": str(cfg.model.name),
        "dataset": str(cfg.dataset.name),
        "optimizer": str(cfg.optimizer.name),
        "val_acc": metrics.get("acc") if isinstance(metrics, dict) else None,
        "val_loss": metrics.get("loss") if isinstance(metrics, dict) else None,
    }
    manager.append_summary(summary_row)

    if accelerator.is_main_process:
        LOGGER.info("RKMv1 학습 완료 - 경로: %s", exp_dir)
    accelerator.wait_for_everyone()
    return exp_dir


@hydra.main(version_base="1.3", config_path="../core/configs", config_name="defaults_rkmv1")
def main(cfg: DictConfig) -> None:
    _train(cfg)


if __name__ == "__main__":
    main()
