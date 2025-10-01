"""MVP 이미지 진위 판별 모델 학습 스크립트."""

from __future__ import annotations

import argparse
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import yaml

import torch

import sys
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.data.datasets import AugConfig, build_dataloader
from core.data.sampler import sample_datasets
from core.models.registry import get_model
from core.trainer.engine import TrainCfg, Trainer
from core.utils.logger import get_logger
from core.utils.seed import set_seed


def _setup_file_logging(log_dir: Path, level: int) -> None:
    """파일 핸들러를 추가해 로그를 저장한다.

    Args:
        log_dir: 로그 파일을 보관할 디렉터리.
        level: 로깅 레벨.

    Returns:
        None
    """

    log_dir.mkdir(parents=True, exist_ok=True)
    fmt = "%(asctime)s | %(name)s | %(levelname)s | %(message)s"
    handler = logging.FileHandler(log_dir / "train.log", encoding="utf-8")
    handler.setFormatter(logging.Formatter(fmt))
    root = logging.getLogger()
    root.addHandler(handler)
    root.setLevel(level)


def parse_args() -> argparse.Namespace:
    """명령행 인자를 파싱한다.

    Returns:
        argparse.Namespace 객체.
    """

    parser = argparse.ArgumentParser(description="Real vs GenAI 이미지 분류 학습")
    parser.add_argument("--config", required=True, help="학습 설정 YAML 경로")
    return parser.parse_args()


def _prepare_model_kwargs(model_cfg: Dict[str, Any], num_classes: int) -> Dict[str, Any]:
    """모델 빌더에 전달할 인자를 구성한다.

    Args:
        model_cfg: YAML에서 읽은 모델 설정.
        num_classes: 출력을 위한 클래스 개수.

    Returns:
        빌더에게 전달할 인자 딕셔너리.
    """

    kwargs = {k: v for k, v in model_cfg.items() if k not in {"name", "residual"}}
    kwargs.setdefault("num_classes", num_classes)
    residual_cfg = model_cfg.get("residual", {})
    if "embed_dim" in residual_cfg:
        kwargs["residual_embed_dim"] = residual_cfg["embed_dim"]
    return kwargs


def main() -> None:
    """학습 파이프라인 전체를 실행한다.

    Returns:
        None
    """

    args = parse_args()
    with open(args.config, "r", encoding="utf-8") as cfg_file:
        config = yaml.safe_load(cfg_file)

    logging_cfg = config.get("logging", {})
    level = getattr(logging, logging_cfg.get("level", "INFO").upper())
    logger = get_logger(__name__, level=level)

    set_seed(config.get("seed"))

    run_name = f"{config['model']['name']}_{datetime.utcnow().strftime('%Y%m%dT%H%M%S')}"
    run_dir = ROOT / "experiments" / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    _setup_file_logging(run_dir / "logs", level)
    logger.info("실험 디렉터리: %s", run_dir)

    try:
        data_cfg = config["data"]
        train_cfg = config["train"]
        model_cfg = config["model"]
    except KeyError as exc:
        logger.error("필수 설정 키가 누락되었습니다: %s", exc)  # TODO: 필요 없을 시 삭제 가능
        raise

    aug_cfg = None
    if config.get("augment", {}).get("sns"):
        aug_cfg = AugConfig(**config["augment"]["sns"])

    sample_datasets(data_cfg)

    train_loader, val_loader, class_names = build_dataloader(
        train_dir=data_cfg["train_dir"],
        val_dir=data_cfg.get("val_dir"),
        batch_size=train_cfg["batch_size"],
        num_workers=data_cfg.get("num_workers", 4),
        img_size=train_cfg["img_size"],
        use_sns_aug=aug_cfg is not None,
        aug_cfg=aug_cfg,
    )

    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(torch.cuda.current_device())
        logger.info("CUDA 사용 가능 - %s", device_name)
    else:
        logger.warning("CUDA를 사용할 수 없습니다. CPU 모드로 학습합니다.")

    builder = get_model(model_cfg["name"])
    model_kwargs = _prepare_model_kwargs(model_cfg, len(class_names))
    model = builder(**model_kwargs)

    optimizer_cfg = train_cfg.get("optimizer")
    if optimizer_cfg is None:
        logger.error("optimizer 설정이 존재하지 않습니다")  # TODO: 필요 없을 시 삭제 가능
        raise KeyError("optimizer")

    scheduler_cfg = train_cfg.get("scheduler", {})

    train_settings = TrainCfg(
        epochs=train_cfg["epochs"],
        lr=optimizer_cfg["lr"],
        weight_decay=optimizer_cfg.get("weight_decay", 0.05),
        optimizer=optimizer_cfg.get("name", "adamw"),
        scheduler=scheduler_cfg.get("name", "cosine"),
        warmup_epochs=scheduler_cfg.get("warmup_epochs", 0),
        mixed_precision=train_cfg.get("mixed_precision"),
        grad_accum_steps=train_cfg.get("grad_accum_steps", 1),
        log_interval=train_cfg.get("log_interval", 50),
        criterion=train_cfg.get("criterion", "ce"),
    )

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        out_dir=str(run_dir),
        cfg=train_settings,
    )
    trainer.fit()

    logger.info("학습 완료 - 아티팩트 경로: %s", run_dir)
if __name__ == "__main__":
    main()
