"""데이터셋 전처리 파이프라인 빌더."""

from __future__ import annotations

from typing import Callable, Iterable, List, Optional

from torchvision import transforms

from .base import DatasetConfig, TransformSpec, transform_specs
from .rkmv1_preprocess import (
    build_rkmv1_train_transform,
    build_rkmv1_val_transform,
)


def _instantiate(spec: TransformSpec) -> Callable:
    """TransformSpec을 torchvision 변환으로 변환."""

    cls = getattr(transforms, spec.type, None)
    if cls is None:
        raise ValueError(f"unsupported transform: {spec.type}")

    params = dict(spec.params)
    if "transforms" in params:
        nested = params.pop("transforms")
        params["transforms"] = [
            _instantiate(nested_spec)
            for nested_spec in transform_specs(nested)
        ]
    return cls(**params)


def _ensure_normalize(ops: List[Callable], mean, std) -> List[Callable]:
    if not any(isinstance(op, transforms.Normalize) for op in ops):
        ops.append(transforms.Normalize(mean=mean, std=std))
    return ops


def build_train_transforms(cfg: DatasetConfig, sns_augment: Optional[Callable]) -> transforms.Compose:
    """학습용 변환을 구성한다."""

    if (cfg.preprocess or "").lower() == "rkmv1":
        return build_rkmv1_train_transform(cfg, sns_augment)

    if cfg.train.transforms:
        ops = [_instantiate(spec) for spec in cfg.train.transforms]
    else:
        ops = [
            transforms.RandomResizedCrop(cfg.image_size, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
        ]
    if sns_augment is not None:
        ops.insert(0, transforms.Lambda(lambda image: sns_augment(image)))

    ops.extend(
        [
            transforms.ToTensor(),
        ]
    )
    ops = _ensure_normalize(ops, cfg.normalization.mean, cfg.normalization.std)
    return transforms.Compose(ops)


def build_val_transforms(cfg: DatasetConfig) -> transforms.Compose:
    """검증용 변환을 구성한다."""

    if (cfg.preprocess or "").lower() == "rkmv1":
        return build_rkmv1_val_transform(cfg)

    if cfg.val and cfg.val.transforms:
        ops = [_instantiate(spec) for spec in cfg.val.transforms]
    else:
        resize = int(round(cfg.image_size * 1.12))
        ops = [
            transforms.Resize(resize),
            transforms.CenterCrop(cfg.image_size),
        ]
    ops.extend(
        [
            transforms.ToTensor(),
        ]
    )
    ops = _ensure_normalize(ops, cfg.normalization.mean, cfg.normalization.std)
    return transforms.Compose(ops)
