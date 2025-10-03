"""이미지 진위 판별용 데이터로더 구성 모듈."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from core.data.transforms_sns import AugConfig, generate_sns_augmentations
from core.utils.logger import get_logger
from PIL import Image, UnidentifiedImageError


logger = get_logger(__name__)


def safe_loader(path: str) -> Image.Image:
    """깨진 이미지를 만나도 학습이 중단되지 않도록 안전하게 로드한다.

    무엇을/왜:
        PIL 로딩 단계에서 UnidentifiedImageError, OSError 등이 발생하는 경우가 있어
        학습이 중단될 수 있다. 이 함수는 예외 발생 시 경고 로그를 남기고 1x1 더미
        이미지를 반환해 학습 루프가 계속 진행되도록 한다.

    Args:
        path: 이미지 파일 경로

    Returns:
        RGB PIL.Image (정상 로드되면 원본, 실패 시 1x1 더미)
    """

    try:
        return Image.open(path).convert("RGB")
    except (UnidentifiedImageError, OSError, ValueError) as e:
        logger.warning("⚠️ Skipping corrupted image: %s (%s)", path, str(e))
        return Image.new("RGB", (1, 1))


def _verify_directory(path: Path) -> None:
    """디렉터리 존재 여부와 내용물을 확인한다."""

    if not path.exists() or not any(path.iterdir()):
        logger.error("데이터 디렉터리가 존재하지 않거나 비어 있습니다: %s", str(path))  # TODO: 필요 없을 시 삭제 가능
        raise FileNotFoundError(f"Dataset directory missing or empty: {path}")


def _build_transforms(img_size: int, use_sns_aug: bool, aug_cfg: Optional[AugConfig]) -> Tuple[transforms.Compose, transforms.Compose]:
    """학습/검증용 변환 파이프라인을 생성한다."""

    sns_aug = generate_sns_augmentations(config=aug_cfg) if use_sns_aug else None

    train_ops = []
    if sns_aug is not None:
        train_ops.append(transforms.Lambda(lambda img: sns_aug(img)))
    train_ops.extend([
        transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])

    val_ops = [
        transforms.Resize(int(img_size * 1.1)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ]

    return transforms.Compose(train_ops), transforms.Compose(val_ops)


def build_dataloader(
    train_dir: str,
    val_dir: Optional[str],
    batch_size: int,
    num_workers: int,
    img_size: int,
    use_sns_aug: bool = True,
    aug_cfg: Optional[AugConfig] = None,
) -> Tuple[DataLoader, Optional[DataLoader], List[str]]:
    """ImageFolder 기반 학습/검증 데이터로더를 생성한다.

    Args:
        train_dir: 학습 데이터 루트 경로(ImageFolder 포맷).
        val_dir: 검증 데이터 루트 경로. 없으면 None.
        batch_size: 배치 크기.
        num_workers: DataLoader 워커 수.
        img_size: 모델 입력 크기.
        use_sns_aug: SNS 증강 사용 여부.
        aug_cfg: 증강 강도 설정.

    Returns:
        (train_loader, val_loader, class_names) 튜플.
    """

    train_path = Path(train_dir)
    _verify_directory(train_path)
    val_loader: Optional[DataLoader] = None

    val_path = Path(val_dir) if val_dir else None
    if val_path is not None:
        _verify_directory(val_path)

    train_tf, val_tf = _build_transforms(img_size, use_sns_aug, aug_cfg)

    train_dataset = datasets.ImageFolder(str(train_path), transform=train_tf, loader=safe_loader)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
        drop_last=True,
    )

    class_names = train_dataset.classes

    if val_path is not None:
        val_dataset = datasets.ImageFolder(str(val_path), transform=val_tf, loader=safe_loader)
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=(num_workers > 0),
        )
        logger.info(
            "데이터 로더 구성 완료 - Train=%d, Val=%d, img_size=%d",
            len(train_dataset),
            len(val_dataset),
            img_size,
        )
    else:
        logger.info(
            "데이터 로더 구성 완료 - Train=%d, Val=None, img_size=%d",
            len(train_dataset),
            img_size,
        )

    return train_loader, val_loader, class_names
