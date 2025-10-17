"""Hydra 구성 기반 DataLoader 빌더."""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

from PIL import Image, UnidentifiedImageError
import torch
from torch.utils.data import ConcatDataset, DataLoader, Subset, WeightedRandomSampler, get_worker_info
from torch.utils.data.distributed import DistributedSampler
from torchvision import datasets, transforms

from core.utils.logger import get_logger

from .base import DatasetConfig, load_dataset_config
from .sns_augment import build_sns_augment
from .transforms import build_train_transforms, build_val_transforms

logger = get_logger(__name__)
_DEBUG_WORKERS = os.environ.get("TVB_DATALOADER_DEBUG_WORKERS")


def _identity_collate(batch):
    return batch


def _get_worker_init_fn(tag: str):
    if not _DEBUG_WORKERS:
        return None

    def _init(worker_id: int) -> None:
        info = get_worker_info()
        rank = os.environ.get("LOCAL_RANK") or os.environ.get("RANK") or "0"
        pid = os.getpid()
        num_workers = getattr(info, "num_workers", "?") if info else "?"
        seed = getattr(info, "seed", "?") if info else "?"
        logger.info(
            "[DataLoader worker:%s] rank=%s pid=%s id=%s num_workers=%s seed=%s",
            tag,
            rank,
            pid,
            worker_id,
            num_workers,
            seed,
        )

    return _init


def _safe_loader(path: str) -> Image.Image:
    try:
        return Image.open(path).convert("RGB")
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        logger.warning("⚠️ 손상된 이미지를 건너뜁니다: %s (%s)", path, exc)
        return Image.new("RGB", (1, 1))


def _apply_limit(dataset, limit: Optional[int]):
    if not limit:
        return dataset
    size = min(limit, len(dataset))
    indices = list(range(size))
    return Subset(dataset, indices)


def _enforce_class_order(image_dataset, *, order: Tuple[str, ...] = ("nature", "ai")) -> None:
    """ImageFolder 클래스 인덱스를 지정된 순서로 재정렬한다."""

    if not hasattr(image_dataset, "classes") or not hasattr(image_dataset, "samples"):
        return

    desired = [name for name in order if any(name == cls for cls in getattr(image_dataset, "classes", []))]
    if len(desired) < 2:
        return

    mapping = {name: idx for idx, name in enumerate(desired)}

    new_samples = []
    new_targets = []
    for path, _ in image_dataset.samples:
        label = Path(path).parent.name
        if label not in mapping:
            continue
        class_idx = mapping[label]
        new_samples.append((path, class_idx))
        new_targets.append(class_idx)

    if not new_samples:
        return

    image_dataset.samples = new_samples
    image_dataset.imgs = new_samples
    image_dataset.targets = new_targets
    image_dataset.classes = desired
    image_dataset.class_to_idx = mapping


def _extract_targets(dataset) -> List[int]:
    if isinstance(dataset, Subset):
        parent_targets = _extract_targets(dataset.dataset)
        return [parent_targets[idx] for idx in dataset.indices]
    targets = getattr(dataset, "targets", None)
    if targets is None:
        raise AttributeError("dataset does not expose targets")
    return list(targets)


def _get_class_names(dataset) -> List[str]:
    if isinstance(dataset, Subset):
        return _get_class_names(dataset.dataset)
    classes = getattr(dataset, "classes", None)
    return list(classes) if classes is not None else []


def _resolve_sources(cfg: DatasetConfig) -> List[str]:
    if cfg.sources:
        return [s for s in cfg.sources if s]
    if cfg.source:
        return [cfg.source]
    return []


def _build_source_dataset(
    source: str,
    split: str,
    *,
    data_root: str,
    transform,
    limit: Optional[int],
) -> Optional[object]:
    root = Path(data_root) / source / split
    if not root.exists():
        logger.warning("소스 %s 의 %s 경로를 찾을 수 없습니다: %s", source, split, root)
        return None
    dataset = datasets.ImageFolder(str(root), transform=transform, loader=_safe_loader)
    _enforce_class_order(dataset)
    dataset = _apply_limit(dataset, limit)
    if len(dataset) == 0:
        logger.warning("소스 %s (%s) 에서 사용할 샘플이 없습니다", source, split)
        return None
    return dataset


def _build_sample_weights(
    per_source_targets: Sequence[Tuple[str, List[int]]],
    source_weights: Sequence[float],
    num_classes: int = 2,
) -> torch.DoubleTensor:
    weights: List[float] = []
    for (_, targets), src_weight in zip(per_source_targets, source_weights):
        if not targets:
            continue
        counts_tensor = torch.bincount(torch.tensor(targets, dtype=torch.long), minlength=num_classes)
        counts = counts_tensor.tolist()
        for target in targets:
            class_count = counts[target] if target < len(counts) else 0
            denom = class_count if class_count > 0 else 1
            weights.append(src_weight / denom)
    if not weights:
        return torch.DoubleTensor()
    return torch.DoubleTensor(weights)


def build_dataloaders(
    cfg: DatasetConfig | dict,
    *,
    shuffle_train: bool = True,
    world_size: int = 1,
    rank: int = 0,
    seed: Optional[int] = None,
    return_raw_val_images: bool = False,
    return_raw_train_images: bool = False,
) -> Tuple[DataLoader, Optional[DataLoader], list[str], transforms.Compose, Optional[callable]]:
    """DatasetConfig를 받아 학습/검증 DataLoader를 생성한다."""

    dataset_cfg = load_dataset_config(cfg)
    loader_kwargs = dict(
        batch_size=dataset_cfg.loader.batch_size,
        num_workers=dataset_cfg.loader.num_workers,
        pin_memory=dataset_cfg.loader.pin_memory,
        persistent_workers=dataset_cfg.loader.persistent_workers and dataset_cfg.loader.num_workers > 0,
    )
    if dataset_cfg.loader.prefetch_factor and dataset_cfg.loader.num_workers > 0:
        loader_kwargs["prefetch_factor"] = dataset_cfg.loader.prefetch_factor
    train_loader_kwargs = dict(loader_kwargs)
    val_loader_kwargs = dict(loader_kwargs)

    train_worker_init = _get_worker_init_fn("train")
    if train_worker_init is not None:
        train_loader_kwargs["worker_init_fn"] = train_worker_init
    val_worker_init = _get_worker_init_fn("val")
    if val_worker_init is not None:
        val_loader_kwargs["worker_init_fn"] = val_worker_init

    sns_config = None
    if dataset_cfg.augment and isinstance(dataset_cfg.augment, dict):
        sns_config = dataset_cfg.augment.get("sns")
    sns_aug = build_sns_augment(sns_config)
    train_tf = build_train_transforms(dataset_cfg, sns_aug)
    val_service_tf = build_val_transforms(dataset_cfg)
    val_tf = val_service_tf

    if return_raw_val_images:
        val_tf = None
        val_loader_kwargs["collate_fn"] = _identity_collate
    if return_raw_train_images:
        train_tf = None
        train_loader_kwargs["collate_fn"] = _identity_collate

    sources = _resolve_sources(dataset_cfg)
    train_datasets = []
    per_source_targets: List[Tuple[str, List[int]]] = []
    included_sources: List[str] = []
    weight_config = dataset_cfg.source_weights or {}
    raw_weights: List[float] = []

    if sources:
        for source in sources:
            dataset = _build_source_dataset(
                source,
                "train",
                data_root=dataset_cfg.data_root,
                transform=train_tf,
                limit=dataset_cfg.train.limit,
            )
            if dataset is None:
                continue
            try:
                targets = _extract_targets(dataset)
            except AttributeError:
                logger.warning("소스 %s 의 타깃 정보를 가져올 수 없습니다", source)
                continue
            if not targets:
                logger.warning("소스 %s 에 유효한 학습 샘플이 없습니다", source)
                continue
            train_datasets.append(dataset)
            per_source_targets.append((source, targets))
            included_sources.append(source)
            raw_weights.append(float(weight_config.get(source, 1.0)))

    if not train_datasets:
        # fallback: 단일 경로 기반 로딩
        train_root = dataset_cfg.train.resolve_root()
        dataset = datasets.ImageFolder(str(train_root), transform=train_tf, loader=_safe_loader)
        _enforce_class_order(dataset)
        dataset = _apply_limit(dataset, dataset_cfg.train.limit)
        train_datasets = [dataset]
        targets = _extract_targets(dataset)
        per_source_targets = [(dataset_cfg.source or "default", targets)]
        included_sources = [dataset_cfg.source or str(train_root)]
        raw_weights = [1.0]

    num_sources = len(train_datasets)
    if num_sources == 0:
        raise RuntimeError("유효한 학습 데이터 소스를 찾을 수 없습니다.")

    normalized_weights: List[float]
    positive_weights = [w if w > 0 else 1.0 for w in raw_weights]
    total_weight = sum(positive_weights)
    if total_weight <= 0:
        normalized_weights = [1.0 / num_sources] * num_sources
    else:
        normalized_weights = [w / total_weight for w in positive_weights]

    if num_sources == 1:
        train_dataset = train_datasets[0]
    else:
        train_dataset = ConcatDataset(train_datasets)

    sample_weights = _build_sample_weights(per_source_targets, normalized_weights)
    distributed = world_size > 1
    sampler = None
    use_sampler = sample_weights.numel() > 0 and not distributed

    train_loader_params = dict(loader_kwargs)
    if return_raw_train_images:
        train_loader_params["collate_fn"] = _identity_collate

    val_loader_params = dict(val_loader_kwargs)
    if return_raw_val_images:
        val_loader_params["collate_fn"] = _identity_collate
    if distributed:
        sampler = DistributedSampler(
            train_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=shuffle_train,
            seed=seed if seed is not None else 0,
        )
        params = dict(train_loader_params)
        params.update(train_loader_kwargs)
        train_loader = DataLoader(
            train_dataset,
            sampler=sampler,
            drop_last=dataset_cfg.loader.drop_last,
            **params,
        )
    elif use_sampler:
        sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
        params = dict(train_loader_params)
        params.update(train_loader_kwargs)
        train_loader = DataLoader(
            train_dataset,
            sampler=sampler,
            drop_last=dataset_cfg.loader.drop_last,
            **params,
        )
    else:
        params = dict(train_loader_params)
        params.update(train_loader_kwargs)
        train_loader = DataLoader(
            train_dataset,
            shuffle=shuffle_train,
            drop_last=dataset_cfg.loader.drop_last,
            **params,
        )

    val_loader: Optional[DataLoader] = None
    val_size = 0
    if dataset_cfg.val is not None:
        val_datasets = []
        for source in included_sources:
            dataset = _build_source_dataset(
                source,
                "val",
                data_root=dataset_cfg.data_root,
                transform=val_tf,
                limit=dataset_cfg.val.limit if dataset_cfg.val else None,
            )
            if dataset is None:
                continue
            val_datasets.append(dataset)
        if not val_datasets and not sources:
            val_root = dataset_cfg.val.resolve_root()
            dataset = datasets.ImageFolder(str(val_root), transform=val_tf, loader=_safe_loader)
            _enforce_class_order(dataset)
            dataset = _apply_limit(dataset, dataset_cfg.val.limit)
            val_datasets = [dataset]
        if val_datasets:
            if len(val_datasets) == 1:
                val_dataset = val_datasets[0]
            else:
                val_dataset = ConcatDataset(val_datasets)
            val_size = len(val_dataset)

            val_sampler = None
            if distributed:
                val_sampler = DistributedSampler(
                    val_dataset,
                    num_replicas=world_size,
                    rank=rank,
                    shuffle=False,
                )
            params = dict(val_loader_params)
            params.update(val_loader_kwargs)
            val_loader = DataLoader(
                val_dataset,
                sampler=val_sampler,
                drop_last=False,
                **params,
            )
        else:
            logger.warning("유효한 검증 데이터를 찾을 수 없어 val_loader를 생성하지 않습니다.")
            params = dict(val_loader_params)
            params.update(val_loader_kwargs)
            val_loader = None

    train_size = len(train_dataset)
    try:
        class_names = _get_class_names(train_datasets[0])
    except Exception:
        class_names = []

    preset_label = None
    if sns_config is not None and hasattr(sns_config, "get"):
        preset_label = sns_config.get("name") or sns_config.get("type")

    source_label = "+".join(included_sources)
    if int(os.environ.get("RANK", "0")) == 0:
        sampler_label = "distributed" if distributed else ("Weighted" if use_sampler else ("shuffle" if shuffle_train else "sequential"))
        logger.info(
            "데이터로더 준비 완료 - train=%d, val=%d, batch=%d, workers=%d, augment=%s, sources=%s, sampler=%s",
            train_size,
            val_size,
            dataset_cfg.loader.batch_size,
            dataset_cfg.loader.num_workers,
            str(preset_label),
            source_label,
            sampler_label,
        )
    train_augment = sns_aug if return_raw_train_images else None
    return train_loader, val_loader, list(class_names), val_service_tf, train_augment
