"""데이터셋 구성을 정의하는 데이터클래스 모듈."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

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
    batch_size: int
    num_workers: int
    pin_memory: bool
    persistent_workers: bool
    drop_last: bool
    prefetch_factor: Optional[int]
    precompute_patches: bool
    train_num_workers: Optional[int] = None
    val_num_workers: Optional[int] = None
    train_prefetch_factor: Optional[int] = None
    val_prefetch_factor: Optional[int] = None


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

def _to_plain_dict(value: Any) -> Dict[str, Any]:
    if OmegaConf is not None and isinstance(value, DictConfig):
        return OmegaConf.to_container(value, resolve=True)  # type: ignore[arg-type]
    if isinstance(value, Mapping):
        return dict(value)
    raise TypeError("expected mapping-like configuration")


def load_dataset_config(cfg: DatasetConfig | Mapping[str, Any]) -> DatasetConfig:
    """DatasetConfig 또는 매핑 객체를 DatasetConfig로 정규화한다."""

    if isinstance(cfg, DatasetConfig):
        return cfg

    if OmegaConf is not None and isinstance(cfg, DictConfig):
        config_dict = OmegaConf.to_container(cfg, resolve=True)  # type: ignore[arg-type]
    elif isinstance(cfg, Mapping):
        config_dict = dict(cfg)
    else:
        raise TypeError("dataset config must be mapping-like or DatasetConfig")

    normalization_cfg = config_dict.get("normalization")
    if isinstance(normalization_cfg, NormalizationConfig):
        normalization = normalization_cfg
    else:
        normalization_dict = (
            _to_plain_dict(normalization_cfg) if normalization_cfg is not None else {}
        )
        normalization = NormalizationConfig(**normalization_dict)

    loader_cfg = config_dict.get("loader")
    if isinstance(loader_cfg, LoaderConfig):
        loader = loader_cfg
    else:
        loader_dict = _to_plain_dict(loader_cfg or {})
        loader = LoaderConfig(**loader_dict)

    train_cfg = config_dict.get("train")
    if isinstance(train_cfg, DatasetSplitConfig):
        train = train_cfg
    else:
        train_dict = _to_plain_dict(train_cfg or {})
        train = DatasetSplitConfig(
            root=train_dict.get("root", "./train"),
            transforms=transform_specs(train_dict.get("transforms", [])),
            limit=train_dict.get("limit"),
        )

    val_cfg = config_dict.get("val")
    if isinstance(val_cfg, DatasetSplitConfig):
        val = val_cfg
    elif val_cfg:
        val_dict = _to_plain_dict(val_cfg)
        val = DatasetSplitConfig(
            root=val_dict.get("root", "./val"),
            transforms=transform_specs(val_dict.get("transforms", [])),
            limit=val_dict.get("limit"),
        )
    else:
        val = None

    augment_cfg = config_dict.get("augment")
    if augment_cfg is None:
        augment = None
    elif isinstance(augment_cfg, Mapping):
        augment = dict(augment_cfg)
    elif OmegaConf is not None and isinstance(augment_cfg, DictConfig):
        augment = OmegaConf.to_container(augment_cfg, resolve=True)  # type: ignore[arg-type]
    else:
        augment = augment_cfg

    return DatasetConfig(
        name=str(config_dict.get("name", "standard")),
        image_size=int(config_dict.get("image_size", 224)),
        data_root=str(config_dict.get("data_root", "./data")),
        source=str(config_dict.get("source", "")),
        sources=list(config_dict.get("sources", [])),
        source_weights=dict(config_dict.get("source_weights", {})),
        augment=augment,
        normalization=normalization,
        loader=loader,
        train=train,
        val=val,
    )
