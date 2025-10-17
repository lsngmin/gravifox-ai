"""데이터셋 구성을 정의하는 데이터클래스 모듈."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

try:
    from omegaconf import DictConfig, OmegaConf
except ImportError:  # pragma: no cover - Hydra 미사용 환경 대비
    DictConfig = Any  # type: ignore
    OmegaConf = None  # type: ignore


@dataclass
class NormalizationConfig:
    """입력 정규화 파라미터."""

    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406)
    std: Tuple[float, float, float] = (0.229, 0.224, 0.225)


@dataclass
class TransformSpec:
    """개별 torchvision 변환 사양."""

    type: str
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DatasetSplitConfig:
    """학습/검증 Split 공통 속성."""

    root: str
    transforms: List[TransformSpec] = field(default_factory=list)
    limit: Optional[int] = None  # 디버깅용 샘플 제한

    def resolve_root(self) -> Path:
        path = Path(self.root).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"dataset split not found: {path}")
        return path


@dataclass
class LoaderConfig:
    """DataLoader 관련 파라미터."""

    batch_size: int = 64
    num_workers: int = 8
    pin_memory: bool = True
    persistent_workers: bool = True
    drop_last: bool = True
    prefetch_factor: Optional[int] = None
    precompute_patches: bool = False


@dataclass
class DatasetConfig:
    """Hydra YAML과 1:1 대응하는 데이터셋 설정."""

    name: str = "standard"
    image_size: int = 224
    data_root: str = "./data"
    source: str = ""
    sources: List[str] = field(default_factory=list)
    source_weights: Dict[str, float] = field(default_factory=dict)
    augment: Optional[Dict[str, Any]] = None
    normalization: NormalizationConfig = field(default_factory=NormalizationConfig)
    loader: LoaderConfig = field(default_factory=LoaderConfig)
    train: DatasetSplitConfig = field(default_factory=lambda: DatasetSplitConfig(root="./train"))
    val: Optional[DatasetSplitConfig] = None


def _maybe_to_container(cfg: Any) -> Dict[str, Any]:
    """DictConfig → dict 변환 헬퍼."""

    if OmegaConf is not None and isinstance(cfg, DictConfig):
        return OmegaConf.to_container(cfg, resolve=True)  # type: ignore[arg-type]
    return cfg


def transform_specs(entries: Iterable[Dict[str, Any]]) -> List[TransformSpec]:
    """Dict 형태의 변환 리스트를 TransformSpec으로 변환."""

    specs: List[TransformSpec] = []
    for entry in entries or []:
        if isinstance(entry, TransformSpec):
            specs.append(entry)
            continue
        if not isinstance(entry, dict) or "type" not in entry:
            raise ValueError(f"invalid transform config: {entry}")
        params = {k: v for k, v in entry.items() if k != "type"}
        specs.append(TransformSpec(type=str(entry["type"]), params=params))
    return specs


def load_dataset_config(cfg: Any) -> DatasetConfig:
    """임의의 입력(dict/DictConfig/dataclass)을 DatasetConfig로 정규화."""

    if isinstance(cfg, DatasetConfig):
        return cfg

    data = _maybe_to_container(cfg)
    if not isinstance(data, dict):
        raise TypeError("dataset config must be dict-like")

    norm = data.get("normalization", {})
    normalization = NormalizationConfig(**norm) if not isinstance(norm, NormalizationConfig) else norm

    loader_cfg = data.get("loader", {})
    loader = LoaderConfig(**loader_cfg) if not isinstance(loader_cfg, LoaderConfig) else loader_cfg

    train_cfg = data.get("train", {})
    train = DatasetSplitConfig(
        root=train_cfg.get("root", "./train"),
        transforms=transform_specs(train_cfg.get("transforms", [])),
        limit=train_cfg.get("limit"),
    )

    val_cfg = data.get("val")
    val: Optional[DatasetSplitConfig] = None
    if val_cfg:
        val = DatasetSplitConfig(
            root=val_cfg.get("root", "./val"),
            transforms=transform_specs(val_cfg.get("transforms", [])),
            limit=val_cfg.get("limit"),
        )

    return DatasetConfig(
        name=data.get("name", "standard"),
        image_size=int(data.get("image_size", 224)),
        data_root=str(data.get("data_root", "./data")),
        source=str(data.get("source", "")),
        sources=list(data.get("sources", [])),
        source_weights=dict(data.get("source_weights", {})),
        augment=_maybe_to_container(data.get("augment")) if data.get("augment") is not None else None,
        normalization=normalization,
        loader=loader,
        train=train,
        val=val,
    )
