"""Hydra 기반 실험 스윕 실행 스크립트."""

from __future__ import annotations

import argparse
import itertools
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf

from core.utils.logger import get_logger
from scripts.train import run_training

logger = get_logger(__name__)


def _parse_grid_expression(expr: str) -> Tuple[str, List[str]]:
    if "=" not in expr:
        raise argparse.ArgumentTypeError(f"올바르지 않은 grid 표현식입니다: {expr}")
    key, values = expr.split("=", 1)
    choices = [v.strip() for v in values.split(",") if v.strip()]
    if not choices:
        raise argparse.ArgumentTypeError(f"값이 비어 있습니다: {expr}")
    return key.strip(), choices


def _build_override_combinations(grid_specs: Sequence[Tuple[str, List[str]]]) -> List[List[str]]:
    if not grid_specs:
        return [[]]
    key_values = [[f"{key}={value}" for value in values] for key, values in grid_specs]
    return [list(combo) for combo in itertools.product(*key_values)]


def _update_output_for_sweep(cfg: DictConfig, index: int) -> None:
    base_dir = Path(cfg.run.output_dir)
    suffix = f"sweep_{index:03d}"
    cfg.run.output_dir = str(base_dir / suffix)

    if "hydra" in cfg and "run" in cfg.hydra:
        cfg.hydra.run.dir = cfg.run.output_dir  # type: ignore[attr-defined]
    if "hydra" in cfg and "sweep" in cfg.hydra:
        cfg.hydra.sweep.dir = str(Path(cfg.hydra.sweep.dir) / suffix)  # type: ignore[attr-defined]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Hydra 멀티런/그리드 스윕 실행기")
    parser.add_argument(
        "--config-name",
        default="defaults",
        help="Hydra 기본 config 이름",
    )
    parser.add_argument(
        "--config-path",
        default="core/configs",
        help="Hydra config 디렉터리 경로 (repo root 기준)",
    )
    parser.add_argument(
        "--set",
        dest="base_overrides",
        action="append",
        default=[],
        help="단일 override (예: model.name=vit_residual_fusion)",
    )
    parser.add_argument(
        "--grid",
        dest="grid_overrides",
        action="append",
        default=[],
        help="그리드 sweep 정의 (예: optimizer.name=adamw,sgd)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    grid_specs = [_parse_grid_expression(expr) for expr in args.grid_overrides]
    combos = _build_override_combinations(grid_specs)

    config_dir = Path(args.config_path).resolve()
    if not config_dir.exists():
        raise FileNotFoundError(f"Hydra config 디렉터리를 찾을 수 없습니다: {config_dir}")

    logger.info("스윕 실행 - config=%s, 조합 수=%d", args.config_name, len(combos))

    with initialize_config_dir(version_base="1.3", config_dir=str(config_dir)):
        for idx, grid_overrides in enumerate(combos, start=1):
            overrides = list(args.base_overrides) + grid_overrides
            cfg = compose(config_name=args.config_name, overrides=overrides)
            OmegaConf.set_struct(cfg, False)
            _update_output_for_sweep(cfg, idx)
            logger.info("[%d/%d] overrides=%s", idx, len(combos), overrides)
            run_training(cfg)


if __name__ == "__main__":
    main()
