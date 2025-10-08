"""실험 산출물(체크포인트/로그)을 관리하는 유틸리티."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

try:
    from omegaconf import DictConfig, OmegaConf
except ImportError:  # pragma: no cover
    DictConfig = None  # type: ignore
    OmegaConf = None  # type: ignore

import yaml

from core.utils.checkpoint import save_checkpoint
from core.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class MonitorConfig:
    """베스트 모델 선택 기준."""

    key: str = "val_loss"
    mode: str = "min"  # "min" | "max"

    def is_improved(self, current: float, best: Optional[float]) -> bool:
        if best is None:
            return True
        if self.mode == "min":
            return current < best
        return current > best


class ExperimentManager:
    """학습 과정에서 생성되는 산출물 경로와 메타데이터를 관리한다."""

    def __init__(
        self,
        root: Path | str,
        monitor: MonitorConfig | Dict[str, Any] | None = None,
        config: Any | None = None,
    ):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.ckpt_dir = self.root / "checkpoints"
        self.ckpt_dir.mkdir(exist_ok=True)
        if monitor is None:
            monitor = MonitorConfig()
        elif isinstance(monitor, dict):
            monitor = MonitorConfig(**monitor)
        self.monitor = monitor
        self.best_value: Optional[float] = None
        self.metrics_file = self.root / "metrics.jsonl"
        self.summary_file = self.root / "best_metrics.json"
        self.meta_file = self.root / "meta.yaml"
        self.config_file = self.root / "config.yaml"
        if config is not None:
            self.capture_config(config)
        logger.info("실험 매니저 초기화 - root=%s monitor=%s/%s", self.root, monitor.key, monitor.mode)

    # ------------------------------------------------------------------
    # Config & metadata helpers
    # ------------------------------------------------------------------
    def capture_config(self, cfg: Any) -> None:
        """실험에 사용된 설정을 메타 스냅샷으로 저장한다."""

        payload = _config_to_dict(cfg)
        with open(self.meta_file, "w", encoding="utf-8") as fp:
            yaml.safe_dump(payload, fp, allow_unicode=True, sort_keys=False)
        with open(self.config_file, "w", encoding="utf-8") as fp:
            yaml.safe_dump(payload, fp, allow_unicode=True, sort_keys=False)
        logger.info("실험 메타 저장: %s", self.meta_file)

    # Backwards compatibility alias
    dump_config = capture_config

    # ------------------------------------------------------------------
    # Metrics & logging
    # ------------------------------------------------------------------
    def log_metrics(self, epoch: int, split: str, metrics: Dict[str, Any]) -> None:
        """메트릭을 JSONL에 축적한다."""

        record = {
            "epoch": int(epoch),
            "split": split,
        }
        for key, value in metrics.items():
            record[key] = _to_serializable(value)
        with open(self.metrics_file, "a", encoding="utf-8") as fp:
            fp.write(json.dumps(record, ensure_ascii=False) + "\n")

    def update_best(self, metrics: Dict[str, Any]) -> bool:
        """모니터 지표를 갱신하고 베스트 여부를 반환."""

        key = self.monitor.key
        if key not in metrics:
            return False
        current = float(metrics[key])
        if self.monitor.is_improved(current, self.best_value):
            self.best_value = current
            with open(self.summary_file, "w", encoding="utf-8") as fp:
                json.dump({"best": current, "monitor": key, "mode": self.monitor.mode, "metrics": metrics}, fp, ensure_ascii=False, indent=2)
            return True
        return False

    # ------------------------------------------------------------------
    # Checkpoint handling
    # ------------------------------------------------------------------
    def save_checkpoint(self, state: Dict[str, Any], *, is_best: bool) -> None:
        """latest/best 체크포인트를 저장한다."""

        save_checkpoint(state, self.ckpt_dir, filename="last.pt")
        if is_best:
            save_checkpoint(state, self.ckpt_dir, filename="best.pt")

    # ------------------------------------------------------------------
    # Summary helpers
    # ------------------------------------------------------------------
    def append_summary(self, row: Dict[str, Any]) -> None:
        """모델 전용 summary.csv에 한 줄을 추가한다."""

        summary_path = self.root.parent / "summary.csv"
        exists = summary_path.exists()
        fieldnames = ["timestamp", "model", "dataset", "optimizer", "val_acc", "val_loss"]
        with open(summary_path, "a", encoding="utf-8", newline="") as fp:
            writer = csv.DictWriter(fp, fieldnames=fieldnames)
            if not exists:
                writer.writeheader()
            writer.writerow({key: row.get(key) for key in fieldnames})


def _to_serializable(value: Any) -> Any:
    """JSON 직렬화를 위해 텐서/넘파이 등을 변환."""

    import numpy as np
    import torch

    if isinstance(value, (float, int, str, bool)) or value is None:
        return value
    if isinstance(value, torch.Tensor):
        if value.ndim == 0:
            return value.item()
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (list, tuple)):
        return [_to_serializable(v) for v in value]
    if isinstance(value, dict):
        return {k: _to_serializable(v) for k, v in value.items()}
    try:
        return float(value)
    except Exception:
        return str(value)


def _config_to_dict(cfg: Any) -> Any:
    if OmegaConf is not None and isinstance(cfg, DictConfig):  # type: ignore[arg-type]
        try:
            return OmegaConf.to_container(cfg, resolve=True)
        except Exception:
            return OmegaConf.to_container(cfg, resolve=False)
    return cfg
