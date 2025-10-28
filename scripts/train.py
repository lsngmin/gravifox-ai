from __future__ import annotations

import logging, os, hydra, torch
import torch.distributed as dist

from pathlib import Path
from typing import Any, Dict, Optional
from accelerate import Accelerator
from accelerate.utils import DistributedType
from omegaconf import DictConfig, OmegaConf

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

def _build_train_config(c: DictConfig) -> TrainCfg:
    """
    여기서 파라미터가 변경되는 로직 없습니다. 모든 파라미터 yaml에 의해 정의됩니다.
    풀백 로직 존재하지 않습니다. / 에러 발생 시 yaml에 정의된 부분 오타나 상속 관계 파악하십시오.
    :param c: hydra Config (yaml)을 정리해 최종적인 딕셔너리를 받아옵니다.
    :return: engine.py에 정의된 TrainCfg DataClass에 값을 채워 반환합니다.
    """
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
        criterion=str(trainer_cfg["criterion"]),
        label_smoothing=float(trainer_cfg["label_smoothing"]),
        max_grad_norm=float(trainer_cfg["max_grad_norm"]),
        patch_chunk_size=int(trainer_cfg["patch_chunk_size"]),
        partial_epochs=trainer_cfg.get("partial_epochs"),
        full_epochs=trainer_cfg.get("full_epochs"),
        partial_steps=trainer_cfg.get("partial_steps"),
        full_steps=trainer_cfg.get("full_steps"),
        service_val_pipeline=bool(trainer_cfg["service_val_pipeline"]),
        step_debug_logging=bool(trainer_cfg["step_debug_logging"]),
        seed=getattr(c.run, "seed", None),
        early_stop=bool(early_cfg["enabled"]),
        early_patience=int(early_cfg["patience"]),
        early_monitor=str(early_cfg["monitor"]),
        early_mode=str(early_cfg["mode"]),
        ckpt_monitor=str(monitor_cfg["key"]),
        ckpt_mode=str(monitor_cfg["mode"]),
    )

def _sync_processes(accelerator: Accelerator) -> None:
    """NCCL 경고 없이 프로세스 간 동기화."""

    if accelerator.num_processes <= 1:
        return

    def _dist_ready() -> bool:
        try:
            return dist.is_available() and dist.is_initialized()
        except Exception:
            return False

    try:
        if _dist_ready():
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
    except Exception as exc:  # pragma: no cover - 로그 정도만 남김
        logger.debug("torch.distributed.barrier 호출 실패: %s", exc)

    state = getattr(accelerator, "state", None)
    distributed_type = getattr(state, "distributed_type", None)
    if distributed_type in (None, DistributedType.NO) or not _dist_ready():
        return

    try:
        accelerator.wait_for_everyone()
    except Exception as exc:  # pragma: no cover - 호출 실패 시 무시
        logger.debug("accelerator.wait_for_everyone 호출 실패: %s", exc)


def r(c: DictConfig) -> Path:
    """Hydra DictConfig를 받아 단일 학습을 수행한다."""
    cfg = c
    train_config = _build_train_config(c)
    train_cfg = train_config

    accelerator = Accelerator()

    if torch.cuda.is_available() and (local_rank := os.environ.get("LOCAL_RANK")) is not None:
        torch.cuda.set_device(int(local_rank))

    set_seed((cfg.run.seed or 0) + accelerator.process_index)
    if cfg.run.seed is not None:
        from accelerate.utils import set_seed as accel_set_seed
        accel_set_seed(cfg.run.seed, device_specific=True)


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
        return_raw_val_images=train_cfg.service_val_pipeline,
        return_raw_train_images=True,
        multipatch_cfg=_to_dict(getattr(cfg, "inference", {})) or {},
        precompute_val_patches=train_cfg.service_val_pipeline,
    )
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
def main(c: DictConfig) -> None:
    r(c)

if __name__ == "__main__":
    main()
