"""Hydra 구성 기반 DataLoader 빌더."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from PIL import Image, UnidentifiedImageError
import torch
from torch.utils.data import ConcatDataset, DataLoader, Subset, WeightedRandomSampler, get_worker_info
from torch.utils.data.distributed import DistributedSampler
from torchvision import datasets, transforms

from core.utils.logger import get_logger

from .base import DatasetConfig, load_dataset_config
from .sns_augment import build_sns_augment
from .transforms import build_train_transforms, build_val_transforms
from core.models.multipatch import (
    generate_patches,
    estimate_priority_regions,
    compute_patch_weights,
)

logger = get_logger(__name__)
_DEBUG_WORKERS = os.environ.get("TVB_DATALOADER_DEBUG_WORKERS")


class MultipatchDataset(torch.utils.data.Dataset):
    """원본 이미지를 멀티패치 텐서로 변환해 반환하는 Dataset 래퍼."""

    def __init__(
        self,
        base_dataset: torch.utils.data.Dataset,
        *,
        augment: Optional[Callable],
        patch_transform: Callable[[Image.Image], torch.Tensor],
        patch_params: Dict[str, Any],
    ) -> None:
        self.base = base_dataset
        self.augment = augment
        self.patch_transform = patch_transform
        self.patch_params = patch_params
        self._to_pil = transforms.ToPILImage()

    def __len__(self) -> int:
        return len(self.base)

    def _ensure_pil(self, image: Any) -> Image.Image:
        if isinstance(image, Image.Image):
            return image
        if isinstance(image, torch.Tensor):
            if image.dim() == 4:
                image = image[0]
            return self._to_pil(image.cpu())
        raise TypeError(f"Unsupported image type for multipatch dataset: {type(image)}")

    def __getitem__(self, index: int):
        image, target = self.base[index]
        image = self._ensure_pil(image)

        if self.augment is not None:
            augmented = self.augment(image)
            if isinstance(augmented, Image.Image):
                image = augmented
            elif isinstance(augmented, torch.Tensor):
                image = self._ensure_pil(augmented)
            else:
                raise TypeError("augment callable must return PIL.Image or Tensor")

        base_cell_size: Optional[int] = None
        cell_cfg = self.patch_params.get("cell_sizes")
        if isinstance(cell_cfg, (list, tuple)):
            for value in cell_cfg:
                if value:
                    base_cell_size = int(value)
                    break
        elif cell_cfg:
            base_cell_size = int(cell_cfg)

        priority_regions = estimate_priority_regions(
            image,
            base_cell_size=base_cell_size,
        )
        patch_samples = generate_patches(
            image,
            sizes=self.patch_params["sizes"],
            n_patches=self.patch_params["n_patches"],
            min_cell_size=self.patch_params["cell_sizes"],
            overlap=self.patch_params["overlap"],
            jitter=self.patch_params["jitter"],
            max_patches=self.patch_params["max_patches"],
            priority_regions=priority_regions,
        )

        if not patch_samples:
            tensor = self.patch_transform(image if image.mode == "RGB" else image.convert("RGB"))
            return tensor.unsqueeze(0), [
                {
                    "scale_index": 0,
                    "priority": True,
                    "complexity": 0.0,
                    "weight": 1.0,
                    "center_x": 0.5,
                    "center_y": 0.5,
                }
            ], target

        patch_tensors: List[torch.Tensor] = []
        metadata: List[Dict[str, Any]] = []
        weights = compute_patch_weights(patch_samples)
        if not weights:
            weights = [1.0 for _ in patch_samples]
        for idx, sample in enumerate(patch_samples):
            patch_img = sample.image if sample.image.mode == "RGB" else sample.image.convert("RGB")
            tensor = self.patch_transform(patch_img)
            patch_tensors.append(tensor)
            metadata.append(
                {
                    "scale_index": int(sample.scale_index),
                    "priority": bool(sample.priority),
                    "complexity": float(sample.complexity),
                    "weight": float(weights[idx] if idx < len(weights) else 1.0),
                    "center_x": float((sample.bbox[0] + sample.bbox[2]) * 0.5),
                    "center_y": float((sample.bbox[1] + sample.bbox[3]) * 0.5),
                }
            )

        stacked = torch.stack(patch_tensors, dim=0).contiguous()
        return stacked, metadata, target


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


def _build_train_pil_augment(cfg: DatasetConfig, sns_augment: Optional[Callable]) -> Optional[Callable]:
    """패치 전처리용 PIL 학습 증강을 구성한다."""

    ops: List[Callable] = []
    if sns_augment is not None:
        ops.append(transforms.Lambda(lambda img: sns_augment(img)))

    if cfg.train.transforms:
        for spec in cfg.train.transforms:
            cls = getattr(transforms, spec.type, None)
            if cls is None:
                raise ValueError(f"unsupported transform for PIL augment: {spec.type}")
            params = dict(spec.params)
            if "transforms" in params:
                params["transforms"] = [
                    _instantiate(nested_spec) for nested_spec in transform_specs(params.pop("transforms"))
                ]
            ops.append(cls(**params))
    else:
        ops.extend(
            [
                transforms.RandomResizedCrop(cfg.image_size, scale=(0.8, 1.0)),
                transforms.RandomHorizontalFlip(),
            ]
        )

    if not ops:
        return None
    return transforms.Compose(ops)


def build_dataloaders(
    cfg: DatasetConfig | dict,
    *,
    shuffle_train: bool = True,
    world_size: int = 1,
    rank: int = 0,
    seed: Optional[int] = None,
    return_raw_val_images: bool = False,
    return_raw_train_images: bool = False,
    multipatch_cfg: Optional[Dict[str, Any]] = None,
    precompute_val_patches: bool = False,
    build_train: bool = True,
    build_val: bool = True,
) -> Tuple[Optional[DataLoader], Optional[DataLoader], list[str], transforms.Compose, Optional[callable]]:
    """DatasetConfig를 받아 학습/검증 DataLoader를 생성한다.

    Args:
        cfg: Hydra YAML을 파싱한 DatasetConfig 또는 동일 구조의 dict.
        shuffle_train: 학습 DataLoader에 셔플을 적용할지 여부.
        world_size: 분산 학습 시 전체 프로세스 수.
        rank: 현재 프로세스의 rank (0-based).
        seed: 분산 샘플러에 전달할 시드;
            None이면 0을 기준으로 재현성 유지 없이 동작.
        return_raw_val_images: True이면 검증 DataLoader가 원본 이미지를 그대로 반환하여 후처리 단계에서 직접 처리할 수 있게 한다.
        return_raw_train_images: True이면 학습 DataLoader도 전처리 없이 PIL 이미지를 그대로 반환한다.
        multipatch_cfg: 멀티패치 전처리를 위한 설정 dict; precompute 플래그가 켜져 있을 때 필수.
        precompute_val_patches: 검증 데이터에도 멀티패치를 적용하여 collate_fn 없이 원본 패치를 직접 반환한다.

    Returns:
        (train_loader, val_loader, class_names, val_transform, train_augment):
            train_loader: 학습용 DataLoader
            val_loader: 검증용 DataLoader (없으면 None)
            class_names: 학습 소스에서 추출한 클래스 이름 목록
            val_transform: 서비스 추론용 기본 검증 transform
            train_augment: raw train 이미지 요청 시 적용 가능한 추가 증강 callable
    """

    dataset_cfg = load_dataset_config(cfg)
    loader_cfg = dataset_cfg.loader
    train_workers = loader_cfg.train_num_workers if loader_cfg.train_num_workers is not None else loader_cfg.num_workers
    val_workers = loader_cfg.val_num_workers if loader_cfg.val_num_workers is not None else loader_cfg.num_workers
    train_prefetch = loader_cfg.train_prefetch_factor if loader_cfg.train_prefetch_factor is not None else loader_cfg.prefetch_factor
    val_prefetch = loader_cfg.val_prefetch_factor if loader_cfg.val_prefetch_factor is not None else loader_cfg.prefetch_factor

    loader_kwargs = dict(
        batch_size=loader_cfg.batch_size,
        num_workers=train_workers,
        pin_memory=loader_cfg.pin_memory,
        persistent_workers=loader_cfg.persistent_workers,
    )
    if train_prefetch is not None:
        loader_kwargs["prefetch_factor"] = train_prefetch
    train_loader_kwargs = dict(loader_kwargs)
    val_loader_kwargs = dict(loader_kwargs)
    train_loader_params = dict(loader_kwargs)
    val_loader_params = dict(loader_kwargs)
    val_loader_kwargs["num_workers"] = val_workers
    val_loader_params["num_workers"] = val_workers
    if val_prefetch is not None:
        val_loader_kwargs["prefetch_factor"] = val_prefetch
        val_loader_params["prefetch_factor"] = val_prefetch
    else:
        val_loader_kwargs.pop("prefetch_factor", None)
        val_loader_params.pop("prefetch_factor", None)

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
    precompute_patches = bool(getattr(loader_cfg, "precompute_patches", False))
    val_precompute_patches = precompute_val_patches or bool(getattr(loader_cfg, "val_precompute_patches", False))
    if precompute_patches:
        train_tf = None
        train_pil_augment = _build_train_pil_augment(dataset_cfg, sns_aug)
    else:
        train_tf = build_train_transforms(dataset_cfg, sns_aug)
        train_pil_augment = None
    val_service_tf = build_val_transforms(dataset_cfg)
    val_tf = val_service_tf

    if return_raw_val_images and build_val:
        val_loader_kwargs["collate_fn"] = _identity_collate
        val_loader_params["collate_fn"] = _identity_collate
    if return_raw_train_images and build_train:
        train_tf = None
        train_loader_kwargs["collate_fn"] = _identity_collate
        train_loader_params["collate_fn"] = _identity_collate
    if val_precompute_patches and build_val:
        val_loader_kwargs["collate_fn"] = _identity_collate
        val_loader_params["collate_fn"] = _identity_collate

    val_loader_transform = val_tf if not (return_raw_val_images or val_precompute_patches) else None

    sources = _resolve_sources(dataset_cfg)
    train_loader: Optional[DataLoader] = None
    train_augment: Optional[Callable] = None
    class_names: List[str] = []
    included_sources: List[str] = []

    patch_params: Optional[Dict[str, Any]] = None
    distributed = world_size > 1
    use_sampler = False

    if build_train:
        train_datasets: List[torch.utils.data.Dataset] = []
        per_source_targets: List[Tuple[str, List[int]]] = []
        weight_config = dataset_cfg.source_weights or {}

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
                targets = _extract_targets(dataset)
                if not targets:
                    logger.warning("소스 %s 에 유효한 학습 샘플이 없습니다", source)
                    continue
                train_datasets.append(dataset)
                per_source_targets.append((source, targets))
                included_sources.append(source)

        if not train_datasets:
            train_root = dataset_cfg.train.resolve_root()
            dataset = datasets.ImageFolder(str(train_root), transform=train_tf, loader=_safe_loader)
            _enforce_class_order(dataset)
            dataset = _apply_limit(dataset, dataset_cfg.train.limit)
            train_datasets = [dataset]
            targets = _extract_targets(dataset)
            base_source = dataset_cfg.source or ""
            if not base_source:
                raise ValueError("dataset.source must be 지정되어야 합니다.")
            per_source_targets = [(base_source, targets)]
            included_sources = [base_source]

        num_sources = len(train_datasets)
        if num_sources == 0:
            raise RuntimeError("유효한 학습 데이터 소스를 찾을 수 없습니다.")

        if num_sources > 1:
            weight_config = dataset_cfg.source_weights or {}
            missing_weights = [src for src in included_sources if src not in weight_config]
            if missing_weights:
                raise ValueError(f"source_weights 설정에 누락된 소스가 있습니다: {missing_weights}")
            provided_weights = [float(weight_config[src]) for src in included_sources]
        else:
            weight_config = dataset_cfg.source_weights or {}
            provided_weights = (
                [float(weight_config[included_sources[0]])] if included_sources and included_sources[0] in weight_config else []
            )
        if provided_weights:
            positive_weights = [w if w > 0 else 1.0 for w in provided_weights]
            total_weight = sum(positive_weights)
            if total_weight <= 0:
                raise ValueError("source_weights 값이 모두 0 이하입니다.")
            normalized_weights: List[float] = [w / total_weight for w in positive_weights]
        else:
            normalized_weights = []

        train_dataset = train_datasets[0] if len(train_datasets) == 1 else ConcatDataset(train_datasets)

        need_multipatch = precompute_patches or val_precompute_patches
        if need_multipatch:
            if not multipatch_cfg:
                raise ValueError("precompute_patches를 사용하려면 multipatch 설정이 필요합니다.")
            mp_cfg = multipatch_cfg
            try:
                sizes_iter = mp_cfg["multiscale"]
            except KeyError as exc:
                raise KeyError("multipatch 설정에 'multiscale' 키가 필요합니다.") from exc
            sizes = tuple(int(x) for x in sizes_iter)
            try:
                cell_cfg = mp_cfg["cell_sizes"]
            except KeyError as exc:
                raise KeyError("multipatch 설정에 'cell_sizes' 키가 필요합니다.") from exc
            cell_sizes = tuple(int(x) for x in (cell_cfg if isinstance(cell_cfg, (list, tuple)) else (cell_cfg,)))
            if len(cell_sizes) == 1 and len(sizes) > 1:
                cell_sizes = tuple([cell_sizes[0]] * len(sizes))
            try:
                n_patches = int(mp_cfg["n_patches"])
            except KeyError as exc:
                raise KeyError("multipatch 설정에 'n_patches' 키가 필요합니다.") from exc
            try:
                overlap = float(mp_cfg["patch_overlap"])
            except KeyError as exc:
                raise KeyError("multipatch 설정에 'patch_overlap' 키가 필요합니다.") from exc
            try:
                jitter = float(mp_cfg["patch_jitter"])
            except KeyError as exc:
                raise KeyError("multipatch 설정에 'patch_jitter' 키가 필요합니다.") from exc
            try:
                max_patches_cfg = mp_cfg["max_patches"]
            except KeyError as exc:
                raise KeyError("multipatch 설정에 'max_patches' 키가 필요합니다.") from exc
            patch_params = {
                "sizes": sizes,
                "cell_sizes": cell_sizes,
                "n_patches": n_patches,
                "overlap": overlap,
                "jitter": jitter,
                "max_patches": int(max_patches_cfg),
            }
            if patch_params["max_patches"] <= 0:
                patch_params["max_patches"] = None
        if precompute_patches and patch_params is not None:
            train_dataset = MultipatchDataset(
                train_dataset,
                augment=train_pil_augment,
                patch_transform=val_service_tf,
                patch_params=patch_params,
            )
            train_loader_kwargs["collate_fn"] = _identity_collate
            train_loader_params["collate_fn"] = _identity_collate

        sample_weights = _build_sample_weights(per_source_targets, normalized_weights) if normalized_weights else torch.DoubleTensor()
        sampler = None
        use_sampler = sample_weights.numel() > 0 and not distributed

        if return_raw_train_images:
            train_loader_params["collate_fn"] = _identity_collate

        train_drop_last = loader_cfg.drop_last
        if return_raw_train_images or precompute_patches:
            train_drop_last = False

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
                drop_last=train_drop_last,
                **params,
            )
        elif use_sampler:
            sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
            params = dict(train_loader_params)
            params.update(train_loader_kwargs)
            train_loader = DataLoader(
                train_dataset,
                sampler=sampler,
                drop_last=train_drop_last,
                **params,
            )
        else:
            params = dict(train_loader_params)
            params.update(train_loader_kwargs)
            train_loader = DataLoader(
                train_dataset,
                shuffle=shuffle_train,
                drop_last=train_drop_last,
                **params,
            )

        class_names = _get_class_names(train_datasets[0]) if train_datasets else []
        if precompute_patches:
            train_augment = None
        else:
            train_augment = sns_aug if return_raw_train_images else None

    val_loader: Optional[DataLoader] = None
    val_size = 0
    if build_val and dataset_cfg.val is not None:
        eval_sources = included_sources if included_sources else (sources if sources else ([dataset_cfg.source] if dataset_cfg.source else []))
        val_datasets = []
        for source in eval_sources:
            dataset = _build_source_dataset(
                source,
                "val",
                data_root=dataset_cfg.data_root,
                transform=val_loader_transform,
                limit=dataset_cfg.val.limit if dataset_cfg.val else None,
            )
            if dataset is None:
                continue
            val_datasets.append(dataset)
        if not val_datasets and not eval_sources:
            val_root = dataset_cfg.val.resolve_root()
            dataset = datasets.ImageFolder(str(val_root), transform=val_loader_transform, loader=_safe_loader)
            _enforce_class_order(dataset)
            dataset = _apply_limit(dataset, dataset_cfg.val.limit)
            val_datasets = [dataset]
        if val_datasets:
            if val_precompute_patches and patch_params is not None:
                wrapped: List[torch.utils.data.Dataset] = []
                for dataset in val_datasets:
                    wrapped.append(
                        MultipatchDataset(
                            dataset,
                            augment=None,
                            patch_transform=val_service_tf,
                            patch_params=patch_params,
                        )
                    )
                val_datasets = wrapped
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
            val_loader = None

        if not class_names and val_datasets:
            class_names = _get_class_names(val_datasets[0])

    train_size = len(train_loader.dataset) if train_loader is not None else 0

    preset_label = None
    if sns_config is not None and hasattr(sns_config, "get"):
        preset_label = sns_config.get("name") or sns_config.get("type")

    source_label = "+".join(included_sources) if included_sources else "+".join(sources) if sources else (dataset_cfg.source or "")
    if int(os.environ.get("RANK", "0")) == 0 and (build_train or build_val):
        sampler_label = "distributed" if distributed else ("weighted" if use_sampler else ("shuffle" if shuffle_train else "sequential"))
        logger.info(
            "데이터로더 준비 완료 - train=%d, val=%d, batch=%d, workers(train=%d/val=%d), augment=%s, sources=%s, sampler=%s",
            train_size,
            val_size,
            loader_cfg.batch_size,
            train_workers,
            val_workers,
            str(preset_label),
            source_label,
            sampler_label,
        )
    return train_loader, val_loader, list(class_names), val_service_tf, train_augment
