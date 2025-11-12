"""RKMv1 구성 로더."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, MutableMapping

import yaml

from core.utils.logger import get_logger

LOGGER = get_logger(__name__)
DEFAULT_CONFIG_PATH = Path(__file__).with_name("config.yaml")


def _merge_dict(base: MutableMapping[str, Any], updates: Mapping[str, Any]) -> None:
    """딥 머지를 수행한다."""

    for key, value in updates.items():
        if isinstance(value, Mapping) and isinstance(base.get(key), MutableMapping):
            _merge_dict(base[key], value)  # type: ignore[index]
        else:
            base[key] = value  # type: ignore[index]


def _to_dataclass(cls, data: Mapping[str, Any]):
    """Dict를 주어진 데이터클래스로 변환한다."""

    return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class BackboneConfig:
    """SigLIP 백본 설정."""

    model_name: str = "vit_large_patch16_siglip_gap_512"
    pretrained: bool = True
    image_size: int = 512
    target_size: int = 512
    output_dim: int = 1024
    freeze_backbone: bool = True


@dataclass
class ResidualConfig:
    """Laplacian + FFT 하이브리드 브랜치 설정."""

    input_channels: int = 3
    embed_dim: int = 256
    spatial_dim: int = 128
    spectral_dim: int = 128
    fft_pool: str = "mean"


@dataclass
class FusionConfig:
    """어텐션 기반 퓨전 설정."""

    siglip_dim: int = 1024
    residual_dim: int = 256
    fused_dim: int = 512
    num_heads: int = 8
    dropout: float = 0.1


@dataclass
class ClassifierConfig:
    """최종 분류 헤드 설정."""

    input_dim: int = 512
    hidden_dim: int = 512
    dropout: float = 0.2
    num_classes: int = 2


@dataclass
class LossConfig:
    """복합 손실 설정."""

    bce_weight: float = 1.0
    triplet_weight: float = 0.2
    domain_weight: float = 0.1
    triplet_margin: float = 0.2
    domain_classes: int = 2
    domain_grl_lambda: float = 1.0


@dataclass
class RKMv1Config:
    """RKMv1 전체 구성."""

    backbone: BackboneConfig = BackboneConfig()
    residual: ResidualConfig = ResidualConfig()
    fusion: FusionConfig = FusionConfig()
    classifier: ClassifierConfig = ClassifierConfig()
    loss: LossConfig = LossConfig()
    model_version: str = "rkm_v1"


def load_rkmv1_config(
    overrides: Mapping[str, Any] | None = None, *, config_path: Path | None = None
) -> RKMv1Config:
    """YAML 구성을 로드하고 덮어쓰기를 적용한다."""

    path = config_path or DEFAULT_CONFIG_PATH
    if not path.is_file():
        raise FileNotFoundError(f"RKMv1 기본 설정을 찾을 수 없습니다: {path}")
    with path.open("r", encoding="utf-8") as fp:
        config_dict: Dict[str, Any] = yaml.safe_load(fp) or {}

    overrides = overrides or {}
    merged = dict(config_dict)
    _merge_dict(merged, dict(overrides))

    backbone = _to_dataclass(BackboneConfig, merged.get("backbone", {}))
    residual = _to_dataclass(ResidualConfig, merged.get("residual", {}))
    fusion = _to_dataclass(FusionConfig, merged.get("fusion", {}))
    classifier = _to_dataclass(ClassifierConfig, merged.get("classifier", {}))
    loss = _to_dataclass(LossConfig, merged.get("loss", {}))
    model_version = str(merged.get("model_version", "rkm_v1"))

    LOGGER.info(
        "RKMv1 설정 로드 - path=%s model_version=%s", path.name, model_version
    )
    return RKMv1Config(
        backbone=backbone,
        residual=residual,
        fusion=fusion,
        classifier=classifier,
        loss=loss,
        model_version=model_version,
    )


__all__ = [
    "BackboneConfig",
    "ResidualConfig",
    "FusionConfig",
    "ClassifierConfig",
    "LossConfig",
    "RKMv1Config",
    "load_rkmv1_config",
]
