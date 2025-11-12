"""Hydra 구성 기반의 단순 학습/검증 DataLoader 빌더."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Subset
from torch.utils.data.distributed import DistributedSampler
from torchvision import datasets, transforms
from PIL import Image, ImageFile, UnidentifiedImageError

from core.utils.logger import get_logger

from .base import DatasetConfig, DatasetSplitConfig, load_dataset_config, transform_specs

logger = get_logger(__name__)

# 손상된 이미지로 인해 학습이 중단되지 않도록 검증 단계에서 빠르게 걸러낸다.
ImageFile.LOAD_TRUNCATED_IMAGES = False


def _configure_truncated_policy(strict_decode: bool) -> None:
    """PIL에서 잘린 이미지를 허용할지 여부를 설정한다."""

    ImageFile.LOAD_TRUNCATED_IMAGES = not strict_decode


def _build_image_loader(image_size: int, *, strict_decode: bool) -> Callable[[str], Image.Image]:
    """ImageFolder에서 사용할 안전한 로더를 생성한다."""

    fallback = Image.new("RGB", (image_size, image_size))

    def _loader(path: str) -> Image.Image:
        try:
            with Image.open(path) as image:
                return image.convert("RGB")
        except (OSError, UnidentifiedImageError, Image.DecompressionBombError, ValueError) as exc:
            if strict_decode:
                raise
            logger.warning("손상된 이미지를 더미 이미지로 대체합니다 - path=%s reason=%s", path, exc)
            return fallback.copy()

    return _loader


def _filter_corrupted_samples(dataset: datasets.ImageFolder, *, source_name: str, split: str) -> None:
    """손상된 이미지 샘플을 제거한다."""

    valid_samples: List[Tuple[str, int]] = []
    valid_targets: List[int] = []
    removed = 0

    for path, target in dataset.samples:
        try:
            with Image.open(path) as image:
                image.verify()
        except (OSError, UnidentifiedImageError, Image.DecompressionBombError) as exc:
            removed += 1
            logger.warning(
                "손상된 이미지 제거 - source=%s split=%s path=%s reason=%s",
                source_name,
                split,
                path,
                exc,
            )
            continue
        valid_samples.append((path, target))
        valid_targets.append(target)

    if removed > 0:
        dataset.samples = valid_samples
        dataset.imgs = valid_samples
        dataset.targets = valid_targets
        logger.info(
            "손상된 이미지 %d개 제거 - source=%s split=%s, 잔여 샘플 %d개",
            removed,
            source_name,
            split,
            len(dataset),
        )


def _warn_if_uneven_length(dataset: Dataset, *, split: str, world_size: int) -> None:
    """분산 학습 시 GPU 간 배치 불균형 가능성을 경고한다."""

    if world_size <= 1:
        return
    dataset_len = len(dataset)
    if dataset_len == 0:
        logger.warning("분산 %s 데이터셋이 비어 있습니다.", split)
        return
    if dataset_len % world_size != 0:
        logger.warning(
            "분산 %s 데이터셋 크기 %d가 world_size=%d로 나누어떨어지지 않습니다. "
            "마지막 배치가 일부 GPU에서만 소비될 수 있습니다.",
            split,
            dataset_len,
            world_size,
        )

def _default_train_transform(image_size: int, mean: Sequence[float], std: Sequence[float]) -> transforms.Compose:
    """기본 학습 데이터 변환 파이프라인을 생성한다."""

    return transforms.Compose(
        [
            transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )


def _default_val_transform(image_size: int, mean: Sequence[float], std: Sequence[float]) -> transforms.Compose:
    """기본 검증 데이터 변환 파이프라인을 생성한다."""

    return transforms.Compose(
        [
            transforms.Resize(int(image_size * 1.15)),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )


def _resolve_transform(
    split_cfg: DatasetSplitConfig,
    *,
    image_size: int,
    mean: Sequence[float],
    std: Sequence[float],
    is_train: bool,
) -> transforms.Compose:
    """사용자 지정 TransformSpec 목록을 토대로 변환을 구성하거나 기본값을 반환한다."""

    specs = transform_specs(split_cfg.transforms)
    if not specs:
        return _default_train_transform(image_size, mean, std) if is_train else _default_val_transform(image_size, mean, std)

    ops: List[transforms.Transform] = []
    for spec in specs:
        name = spec.type.lower()
        params = dict(spec.params)
        try:
            if name == "resize":
                size = params.get("size", image_size)
                ops.append(transforms.Resize(size))
            elif name == "centercrop":
                size = params.get("size", image_size)
                ops.append(transforms.CenterCrop(size))
            elif name == "randomresizedcrop":
                size = params.get("size", image_size)
                scale = params.get("scale", (0.8, 1.0))
                ratio = params.get("ratio", (3.0 / 4.0, 4.0 / 3.0))
                ops.append(transforms.RandomResizedCrop(size, scale=scale, ratio=ratio))
            elif name == "randomhorizontalflip":
                p = float(params.get("p", 0.5))
                ops.append(transforms.RandomHorizontalFlip(p))
            elif name == "colorjitter":
                ops.append(
                    transforms.ColorJitter(
                        brightness=params.get("brightness"),
                        contrast=params.get("contrast"),
                        saturation=params.get("saturation"),
                        hue=params.get("hue"),
                    )
                )
            elif name == "gaussianblur":
                kernel_size = params.get("kernel_size", 3)
                sigma = params.get("sigma", (0.1, 2.0))
                ops.append(transforms.GaussianBlur(kernel_size=kernel_size, sigma=sigma))
            else:
                logger.warning("알 수 없는 변환 %s을(를) 건너뜁니다.", spec.type)
        except Exception as exc:  # pragma: no cover - 예외 로깅
            logger.warning("변환 %s 구성 중 오류 발생: %s", spec.type, exc)

    ops.append(transforms.ToTensor())
    ops.append(transforms.Normalize(mean=mean, std=std))
    return transforms.Compose(ops)


def _apply_limit(dataset: datasets.ImageFolder, limit: Optional[int]) -> datasets.ImageFolder | Subset:
    """디버깅을 위해 샘플 수를 제한한다."""

    if not limit:
        return dataset
    if limit >= len(dataset):
        return dataset
    indices = list(range(limit))
    return Subset(dataset, indices)


def _resolve_source_root(
    dataset_cfg: DatasetConfig,
    split_cfg: DatasetSplitConfig,
    source_name: str,
    *,
    split: str,
) -> Path:
    """단일 소스에 대한 split 경로를 계산한다."""

    base_root = Path(split_cfg.root).expanduser()
    default_source = dataset_cfg.source
    if default_source and default_source in base_root.parts:
        parts = [source_name if part == default_source else part for part in base_root.parts]
        resolved = Path(*parts)
    else:
        resolved = Path(dataset_cfg.data_root).expanduser() / source_name / split
    return resolved


def _iterate_sources(cfg: DatasetConfig) -> Iterable[Tuple[str, float]]:
    """소스 이름과 가중치를 순회한다."""

    if cfg.sources:
        weight_map: Dict[str, float] = {name: float(value) for name, value in cfg.source_weights.items()}
        for name in cfg.sources:
            weight = weight_map.get(name, 1.0)
            if weight <= 0.0:
                logger.warning("소스 %s 가중치가 0 이하입니다. 건너뜁니다.", name)
                continue
            yield name, weight
    elif cfg.source:
        weight = float(cfg.source_weights.get(cfg.source, 1.0))
        if weight > 0.0:
            yield cfg.source, weight
    else:
        logger.warning("사용 가능한 데이터 소스가 없습니다.")


def _combine_datasets(datasets_and_weights: List[Tuple[Dataset, float]], *, max_repeat: int = 8) -> Optional[Dataset]:
    """여러 데이터셋을 가중치에 따라 결합한다."""

    if not datasets_and_weights:
        return None

    if len(datasets_and_weights) == 1:
        return datasets_and_weights[0][0]

    weights = [weight for _, weight in datasets_and_weights if weight > 0.0]
    min_weight = min(weights) if weights else 1.0
    repeats: List[int] = []
    for _, weight in datasets_and_weights:
        if weight <= 0.0:
            repeats.append(0)
            continue
        scaled = max(1, int(round(weight / min_weight)))
        repeats.append(min(scaled, max_repeat))

    expanded: List[Dataset] = []
    for (dataset, _), repeat in zip(datasets_and_weights, repeats):
        if repeat <= 0:
            continue
        expanded.extend([dataset] * repeat)

    if not expanded:
        return None
    if len(expanded) == 1:
        return expanded[0]
    return ConcatDataset(expanded)


def _build_loader(
    dataset: Dataset,
    *,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
    persistent_workers: bool,
    drop_last: bool,
    prefetch_factor: Optional[int],
    sampler: Optional[DistributedSampler],
    shuffle: bool,
) -> DataLoader:
    """기본 DataLoader를 생성한다."""

    loader_kwargs = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        persistent_workers=persistent_workers,
        sampler=sampler,
        shuffle=shuffle if sampler is None else False,
    )
    if prefetch_factor is not None:
        loader_kwargs["prefetch_factor"] = prefetch_factor
    return DataLoader(dataset, **loader_kwargs)


def _prepare_sampler(
    dataset: Dataset,
    *,
    world_size: int,
    rank: int,
    seed: Optional[int],
    shuffle: bool,
) -> Optional[DistributedSampler]:
    """분산 학습 환경에서 사용할 샘플러를 준비한다."""

    if world_size <= 1:
        return None
    return DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=shuffle,
        seed=seed or 0,
        drop_last=False,
    )


def _build_split_dataset(
    dataset_cfg: DatasetConfig,
    split_cfg: DatasetSplitConfig,
    transform: transforms.Compose,
    limit: Optional[int],
    *,
    split: str,
    replicate: bool,
    skip_corrupt_check: bool,
    loader_fn: Callable[[str], Image.Image],
) -> Tuple[Optional[Dataset], List[str]]:
    """여러 소스를 결합한 Dataset과 클래스 이름을 생성한다."""

    datasets_with_weights: List[Tuple[Dataset, float]] = []
    class_names: List[str] = []
    for source_name, weight in _iterate_sources(dataset_cfg):
        root = _resolve_source_root(dataset_cfg, split_cfg, source_name, split=split)
        if not root.exists():
            logger.warning("소스 %s의 %s 경로가 존재하지 않아 건너뜁니다: %s", source_name, split, root)
            continue
        logger.info("데이터 소스 로드 - source=%s split=%s path=%s", source_name, split, root)
        base_dataset = datasets.ImageFolder(root, transform=transform, loader=loader_fn)
        if skip_corrupt_check:
            logger.info("손상 이미지 검사를 건너뜁니다 - source=%s split=%s", source_name, split)
        else:
            _filter_corrupted_samples(base_dataset, source_name=source_name, split=split)
        limited = _apply_limit(base_dataset, limit)
        if not class_names:
            class_names = list(getattr(base_dataset, "classes", []))
        datasets_with_weights.append((limited, weight))

    if not datasets_with_weights:
        return None, class_names

    if replicate:
        combined = _combine_datasets(datasets_with_weights)
    else:
        combined = datasets_with_weights[0][0] if len(datasets_with_weights) == 1 else ConcatDataset(
            [dataset for dataset, _ in datasets_with_weights]
        )

    return combined, class_names


def build_dataloaders(
    cfg: DatasetConfig | Dict[str, Any],
    *,
    build_train: bool = True,
    build_val: bool = True,
    world_size: int = 1,
    rank: int = 0,
    seed: Optional[int] = None,
    shuffle_train: bool = True,
    **_: Any,
) -> Tuple[Optional[DataLoader], Optional[DataLoader], List[str], transforms.Compose, transforms.Compose]:
    """Hydra 설정을 토대로 기본적인 학습/검증 DataLoader를 생성한다."""

    dataset_cfg = load_dataset_config(cfg)
    mean = dataset_cfg.normalization.mean
    std = dataset_cfg.normalization.std

    train_loader: Optional[DataLoader] = None
    val_loader: Optional[DataLoader] = None
    class_names: List[str] = []

    train_transform = _resolve_transform(
        dataset_cfg.train,
        image_size=dataset_cfg.image_size,
        mean=mean,
        std=std,
        is_train=True,
    )
    val_transform = _resolve_transform(
        dataset_cfg.val or dataset_cfg.train,
        image_size=dataset_cfg.image_size,
        mean=mean,
        std=std,
        is_train=False,
    )
    skip_corrupt_check = bool(getattr(dataset_cfg.loader, "skip_corrupt_check", False))
    strict_decode = not skip_corrupt_check
    _configure_truncated_policy(strict_decode)
    loader_fn = _build_image_loader(dataset_cfg.image_size, strict_decode=strict_decode)

    if build_train:
        train_dataset, class_names = _build_split_dataset(
            dataset_cfg,
            dataset_cfg.train,
            train_transform,
            getattr(dataset_cfg.train, "limit", None),
            split="train",
            replicate=True,
            skip_corrupt_check=skip_corrupt_check,
            loader_fn=loader_fn,
        )
        if train_dataset is not None:
            _warn_if_uneven_length(train_dataset, split="train", world_size=world_size)
            loader_cfg = dataset_cfg.loader
            train_sampler = _prepare_sampler(
                train_dataset,
                world_size=world_size,
                rank=rank,
                seed=seed,
                shuffle=shuffle_train,
            )
            train_loader = _build_loader(
                train_dataset,
                batch_size=loader_cfg.batch_size,
                num_workers=loader_cfg.train_num_workers or loader_cfg.num_workers,
                pin_memory=loader_cfg.pin_memory,
                persistent_workers=loader_cfg.persistent_workers,
                drop_last=loader_cfg.drop_last,
                prefetch_factor=loader_cfg.train_prefetch_factor or loader_cfg.prefetch_factor,
                sampler=train_sampler,
                shuffle=shuffle_train,
            )

    if build_val and dataset_cfg.val is not None:
        val_dataset, val_class_names = _build_split_dataset(
            dataset_cfg,
            dataset_cfg.val,
            val_transform,
            getattr(dataset_cfg.val, "limit", None),
            split="val",
            replicate=False,
            skip_corrupt_check=skip_corrupt_check,
            loader_fn=loader_fn,
        )
        if val_dataset is not None:
            _warn_if_uneven_length(val_dataset, split="val", world_size=world_size)
            loader_cfg = dataset_cfg.loader
            val_sampler = _prepare_sampler(val_dataset, world_size=world_size, rank=rank, seed=seed, shuffle=False)
            val_loader = _build_loader(
                val_dataset,
                batch_size=loader_cfg.batch_size,
                num_workers=loader_cfg.val_num_workers or loader_cfg.num_workers,
                pin_memory=loader_cfg.pin_memory,
                persistent_workers=loader_cfg.persistent_workers,
                drop_last=False,
                prefetch_factor=loader_cfg.val_prefetch_factor or loader_cfg.prefetch_factor,
                sampler=val_sampler,
                shuffle=False,
            )
            if not class_names:
                class_names = val_class_names

    if not class_names:
        class_names = []

    return train_loader, val_loader, class_names, val_transform, train_transform
