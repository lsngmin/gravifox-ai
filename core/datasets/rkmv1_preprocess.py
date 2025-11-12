"""RKMv1 전용 이미지 전처리 파이프라인."""

from __future__ import annotations

import random
from typing import Callable, Optional

from PIL import Image, ImageEnhance, ImageFilter
from torchvision import transforms

from .base import DatasetConfig

CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


class EnsureRGB:
    """입력을 강제로 RGB 모드로 변환한다."""

    def __call__(self, image: Image.Image) -> Image.Image:
        if image.mode == "RGB":
            return image
        return image.convert("RGB")


class RandomToneCurve:
    """밝기/대비를 소폭 변경해 SNS 재인코딩을 흉내 낸다."""

    def __init__(self, max_shift: float = 0.12):
        self.max_shift = max_shift

    def __call__(self, image: Image.Image) -> Image.Image:
        if random.random() > 0.6:
            return image
        enhancer = ImageEnhance.Brightness(image)
        image = enhancer.enhance(1.0 + random.uniform(-self.max_shift, self.max_shift))
        contrast = ImageEnhance.Contrast(image)
        image = contrast.enhance(1.0 + random.uniform(-self.max_shift, self.max_shift))
        return image


class RandomLowpassBlur:
    """SNS 업로드 중 발생하는 경미한 블러를 시뮬레이션한다."""

    def __init__(self, radius: float = 1.0, probability: float = 0.3):
        self.radius = radius
        self.probability = probability

    def __call__(self, image: Image.Image) -> Image.Image:
        if random.random() > self.probability:
            return image
        return image.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.1, self.radius)))


def _compose(
    ops: list[Callable],
    *,
    normalize_mean: tuple[float, ...] = CLIP_MEAN,
    normalize_std: tuple[float, ...] = CLIP_STD,
) -> transforms.Compose:
    ops.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=normalize_mean, std=normalize_std),
        ]
    )
    return transforms.Compose(ops)


def build_rkmv1_train_transform(
    cfg: DatasetConfig, sns_augment: Optional[Callable] = None
) -> transforms.Compose:
    """RKMv1 학습용 전처리를 반환한다."""

    ops: list[Callable] = [EnsureRGB()]
    if sns_augment is not None:
        ops.append(transforms.Lambda(lambda img: sns_augment(img)))
    ops.extend(
        [
            transforms.RandomResizedCrop(
                cfg.image_size,
                scale=(0.85, 1.0),
                ratio=(0.75, 1.35),
                interpolation=Image.BICUBIC,
            ),
            transforms.RandomHorizontalFlip(),
            transforms.RandomChoice(
                [
                    transforms.Identity(),
                    transforms.ColorJitter(
                        brightness=0.15, contrast=0.15, saturation=0.1, hue=0.05
                    ),
                ]
            ),
            RandomToneCurve(max_shift=0.1),
            RandomLowpassBlur(radius=1.2, probability=0.25),
        ]
    )
    return _compose(ops)


def build_rkmv1_val_transform(cfg: DatasetConfig) -> transforms.Compose:
    """RKMv1 검증/추론용 전처리를 반환한다."""

    ops: list[Callable] = [
        EnsureRGB(),
        transforms.Resize(
            int(round(cfg.image_size * 1.08)),
            interpolation=Image.BICUBIC,
        ),
        transforms.CenterCrop(cfg.image_size),
    ]
    return _compose(ops)


__all__ = ["build_rkmv1_train_transform", "build_rkmv1_val_transform"]
