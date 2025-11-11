from __future__ import annotations

import logging
import os

import hydra
import torch
import torch.distributed as dist
from pathlib import Path
from typing import Any, Dict, Optional
from accelerate import Accelerator
from omegaconf import DictConfig, OmegaConf

from core.datasets import build_dataloaders
from core.models.registry import get_model
from core.trainer.engine import TrainCfg, Trainer
from core.trainer.experiment_manager import ExperimentManager, MonitorConfig
from core.utils.logger import get_logger, setup_experiment_loggers
from core.utils.seed import set_seed

logger = get_logger(__name__)

_NCCL_ENV_DEFAULTS: Dict[str, str] = {
    "TORCH_NCCL_ASYNC_ERROR_HANDLING": "1",
    "TORCH_NCCL_BLOCKING_WAIT": "1",
    "NCCL_TIMEOUT": "1800",
}


def _apply_nccl_env_defaults() -> None:
    """NCCL 관련 기본 환경 변수를 설정한다."""

    for key, value in _NCCL_ENV_DEFAULTS.items():
        os.environ.setdefault(key, value)


def _to_dict(cfg: Any) -> Dict[str, Any]:
    if isinstance(cfg, DictConfig):
        return OmegaConf.to_container(cfg, resolve=True)  # type: ignore[return-value]
    return cfg

def _build_train_config(c: DictConfig) -> TrainCfg:
    """
    여기 파라미터가 변경되는 로직 없습니다. 모든 파라미터 yaml에 의해 정의됩니다.
    풀백 로직 존재하지 않습니다. / 에러 발생 시 yaml에 정의된 부분 오타나 상속 관계 파악하십시오.
    :param c: hydra Config (yaml)을 정리해 최종적인 딕셔너리를 받아옵니다.
    :return: engine.py에 정의된 TrainCfg DataClass에 값을 채워 반환합니다.
    """
    trainer_cfg = OmegaConf.to_container(c.trainer, resolve=True)
    optimizer_cfg = OmegaConf.to_container(c.optimizer, resolve=True)
    scheduler_cfg = OmegaConf.to_container(c.scheduler, resolve=True)

    early_cfg = trainer_cfg["early_stopping"]
    monitor_cfg = trainer_cfg["monitor"]

    def _optional_positive_int(value: Any) -> Optional[int]:
        if value is None:
            return None
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            return None
        return parsed if parsed > 0 else None

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
        criterion=str(trainer_cfg["criterion"]),
        label_smoothing=float(trainer_cfg["label_smoothing"]),
        max_grad_norm=float(trainer_cfg["max_grad_norm"]),
        patch_chunk_size=int(trainer_cfg["patch_chunk_size"]),
        partial_epochs=trainer_cfg.get("partial_epochs"),
        full_epochs=trainer_cfg.get("full_epochs"),
        partial_steps=trainer_cfg.get("partial_steps"),
        full_steps=trainer_cfg.get("full_steps"),
        max_train_steps=_optional_positive_int(trainer_cfg.get("train_max_steps")),
        max_val_steps=_optional_positive_int(trainer_cfg.get("val_max_steps")),
        step_debug_logging=bool(trainer_cfg["step_debug_logging"]),
        seed=getattr(c.run, "seed", None),
        early_stop=bool(early_cfg["enabled"]),
        early_patience=int(early_cfg["patience"]),
        early_monitor=str(early_cfg["monitor"]),
        early_mode=str(early_cfg["mode"]),
        ckpt_monitor=str(monitor_cfg["key"]),
        ckpt_mode=str(monitor_cfg["mode"]),
    )

def r(c: DictConfig) -> Path:
    """Hydra DictConfig를 받아 단일 학습을 수행한다."""
    cfg = c
    train_config = _build_train_config(c)
    train_cfg = train_config

    # GPU 세팅 확인
    _apply_nccl_env_defaults()
    master_addr = os.environ.get("MASTER_ADDR")
    if not master_addr or master_addr.lower() == "localhost":
        os.environ["MASTER_ADDR"] = "127.0.0.1"

    local_rank_env = os.environ.get("LOCAL_RANK")
    if torch.cuda.is_available() and local_rank_env is not None:
        device_id = int(local_rank_env)
        torch.cuda.set_device(device_id)
        if dist.is_available() and not dist.is_initialized():
            backend = os.environ.get("ACCELERATE_PROCESS_GROUP_BACKEND")
            if not backend:
                backend = "nccl" if torch.cuda.is_available() else "gloo"
            rank_env = os.environ.get("RANK")
            world_env = os.environ.get("WORLD_SIZE")
            rank = int(rank_env) if rank_env is not None else None
            world_size = int(world_env) if world_env is not None else None
            device = torch.device("cuda", device_id) if backend == "nccl" else None
            init_kwargs: Dict[str, Any] = {"backend": backend}
            if rank is not None:
                init_kwargs["rank"] = rank
            if world_size is not None:
                init_kwargs["world_size"] = world_size
            if device is not None:
                init_kwargs["device_id"] = device
            dist.init_process_group(**init_kwargs)
    accelerator = Accelerator()

    if accelerator.num_processes != 2:
        raise RuntimeError(
            f"분산 학습은 반드시 2개의 GPU에서만 수행해야 합니다. "
            f"현재 프로세스 수: {accelerator.num_processes}."
        )

    if not accelerator.is_main_process:
        suppressed_loggers = (
            "timm",
            "huggingface_hub",
            "gravifox.system",
            "gravifox.train",
        )
        for logger_name in suppressed_loggers:
            logging.getLogger(logger_name).setLevel(logging.ERROR)

    # 시드 동기화
    set_seed((c.run.seed or 0) + accelerator.process_index)
    if c.run.seed is not None:
        from accelerate.utils import set_seed as accel_set_seed
        accel_set_seed(c.run.seed, device_specific=True)

    # 실험 파일 생성 후 GPU 프로세스 동기화
    experiment_dir = Path(c.run.output_dir).expanduser().resolve()
    if accelerator.is_main_process:
        experiment_dir.mkdir(parents=True, exist_ok=True)
    accelerator.wait_for_everyone()
    # 로깅 설정
    logging_cfg = c.logging
    level_name = str(logging_cfg.level)
    log_level = getattr(logging, level_name.upper(), logging.INFO)
    if accelerator.is_main_process:
        logging.getLogger().setLevel(log_level)
        logger.setLevel(log_level)

    handlers_cfg = logging_cfg.handlers
    console_enabled = bool(handlers_cfg.console)

    file_cfg = handlers_cfg.file
    file_enabled = bool(file_cfg.enabled)
    log_path: Optional[str | Path]
    log_path = None
    json_format = True
    if file_enabled:
        log_path = Path(str(file_cfg.path))
        format_value = str(getattr(file_cfg, "format", "json")).lower()
        json_format = format_value != "text"

    world_size = accelerator.num_processes
    rank = accelerator.process_index
    setup_kwargs = dict(
        exp_dir=experiment_dir,
        level=log_level,
        console=console_enabled,
        log_path=log_path,
        json_format=json_format,
    )
    if accelerator.is_main_process:
        setup_experiment_loggers(**setup_kwargs)
    accelerator.wait_for_everyone()

    dataset_cfg = c.dataset
    eval_only = bool(getattr(cfg.run, "eval_only", False) or getattr(cfg.run, "validate_only", False))

    train_loader, val_loader, class_names, _, _ = build_dataloaders(
        dataset_cfg,
        build_train=not eval_only,
        build_val=True,
        world_size=world_size,
        rank=rank,
        seed=cfg.run.seed,
        shuffle_train=not eval_only,
    )
    if val_loader is None:
        raise RuntimeError("검증 DataLoader를 생성하지 못했습니다.")
    if accelerator.is_main_process and train_loader is None and not eval_only:
        logger.warning("학습용 DataLoader가 비어 있습니다. 설정을 확인하세요.")
    if accelerator.is_main_process:
        logger.info("데이터셋 클래스: %s", class_names)

    params_dict = _to_dict(cfg.model.get("params", {})) or {}
    model = get_model(cfg.model.name, **params_dict)

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
        cfg=train_cfg,
        experiment=manager,
        accelerator=accelerator,
    )
    if eval_only:
        final_val_metrics = trainer.validate_only()
    else:
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
    accelerator.wait_for_everyone()
    return experiment_dir


@hydra.main(version_base="1.3", config_path="../core/configs", config_name="defaults")
def main(c: DictConfig) -> None:
    r(c)

if __name__ == "__main__":
    main()
