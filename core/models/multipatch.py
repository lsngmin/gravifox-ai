"""멀티패치·멀티스케일 추론 유틸리티와 간단한 ViT 멀티패치 모델."""

from __future__ import annotations
import math
import random
from collections import defaultdict
from dataclasses import dataclass, replace
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np

import timm
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms

from core.utils.logger import get_logger
from .registry import register


logger = get_logger(__name__)
_DEFAULT_INFER_TRANSFORM = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        ),
    ]
)


@dataclass
class MultiPatchConfig:
    """ViTMultiPatch 모델 구성."""

    num_classes: int = 2
    pretrained: bool = True
    vit_name: str = "vit_base_patch16_224"


@dataclass(frozen=True)
class PatchSample:
    """단일 패치 이미지와 위치 메타데이터."""

    image: Image.Image
    bbox: Tuple[float, float, float, float]
    """원본 이미지 기준 정규화된 [x1, y1, x2, y2]."""
    scale: int
    """추론 스케일(모델 입력 리사이즈 전 목표 변)."""
    grid_index: Tuple[int, int]
    """(row, col) 그리드 위치."""
    scale_index: int
    """sizes 리스트 내 인덱스."""
    patch_index: int
    """전역 패치 인덱스."""
    priority: bool = False
    """우선순위 패치 여부."""
    complexity: float = 0.0
    """복잡도 점수(variance/gradient 기반)."""
    source: str = "grid"
    """패치 생성 출처. grid | priority."""


class ViTMultiPatch(nn.Module):
    """여러 패치를 평균해 최종 로짓을 생성하는 간단한 모델.

    SNS 환경에서 한 장의 이미지를 다양한 크기와 위치로 잘라서 추론하면
    압축 노이즈나 마스킹의 영향을 줄일 수 있다. 이 모델은 학습 시에도
    동일한 전략을 활용할 수 있도록 했다.
    """

    def __init__(self, cfg: MultiPatchConfig):
        super().__init__()
        self.cfg = cfg
        self.vit = timm.create_model(cfg.vit_name, pretrained=cfg.pretrained, num_classes=cfg.num_classes)

    def forward(self, image_tensor: torch.Tensor) -> torch.Tensor:
        """하나의 텐서를 입력받아 멀티패치 평균 로짓을 반환한다."""

        b, c, h, w = image_tensor.shape
        center = torch.nn.functional.interpolate(image_tensor, size=(224, 224), mode="bilinear", align_corners=False)
        logits = self.vit(center)
        return logits


@register("vit_multipatch")
def build_vit_multipatch(**kwargs) -> ViTMultiPatch:
    """멀티패치 모델을 레지스트리에 등록한다."""

    cfg = MultiPatchConfig(**kwargs)
    return ViTMultiPatch(cfg)


def _normalize_per_scale(
    value: Optional[Union[int, Sequence[Optional[int]]]],
    length: int,
    default_fn,
) -> List[int]:
    """값을 스케일별 리스트로 확장한다."""

    if value is None:
        return [int(max(1, default_fn(i))) for i in range(length)]
    if isinstance(value, (list, tuple)):
        if len(value) == 0:
            return [int(max(1, default_fn(i))) for i in range(length)]
        if len(value) == 1:
            raw = value[0]
            return [int(max(1, raw if raw is not None else default_fn(i))) for i in range(length)]
        if len(value) != length:
            raise ValueError("per-scale 값의 길이가 스케일 개수와 일치해야 합니다.")
        normalized: List[int] = []
        for idx, raw in enumerate(value):
            base = default_fn(idx)
            normalized.append(int(max(1, raw if raw is not None else base)))
        return normalized
    base_value = int(value)
    return [int(max(1, base_value)) for _ in range(length)]


def _normalize_optional_ints(
    value: Optional[Union[int, Sequence[Optional[int]]]],
    length: int,
) -> List[Optional[int]]:
    if value is None:
        return [None] * length
    if isinstance(value, (list, tuple)):
        if len(value) == 0:
            return [None] * length
        if len(value) == 1:
            raw = value[0]
            casted = int(raw) if raw is not None else None
            return [casted] * length
        if len(value) != length:
            raise ValueError("per-scale 격자 값의 길이가 스케일 개수와 일치해야 합니다.")
        normalized: List[Optional[int]] = []
        for raw in value:
            normalized.append(int(raw) if raw is not None else None)
        return normalized
    return [int(value)] * length


def _normalize_float_sequence(
    value: Optional[Union[float, Sequence[Optional[float]]]],
    length: int,
    *,
    default: float = 0.0,
    min_value: float = 0.0,
    max_value: float = 1.0,
) -> List[float]:
    """스케일별 부동소수 리스트를 정규화한다."""

    def _clamp(raw: float) -> float:
        return float(max(min_value, min(max_value, raw)))

    if value is None:
        return [_clamp(default)] * length
    if isinstance(value, (list, tuple)):
        if len(value) == 0:
            return [_clamp(default)] * length
        if len(value) == 1:
            raw = value[0]
            val = default if raw is None else float(raw)
            return [_clamp(val)] * length
        if len(value) != length:
            raise ValueError("per-scale 부동소수 값의 길이가 스케일 개수와 일치해야 합니다.")
        normalized: List[float] = []
        for raw in value:
            val = default if raw is None else float(raw)
            normalized.append(_clamp(val))
        return normalized
    return [_clamp(float(value))] * length


def _estimate_complexity(image: Image.Image) -> float:
    """간단한 Sobel/분산 기반 복잡도 점수를 계산한다."""

    gray = np.asarray(image.convert("L"), dtype=np.float32)
    if gray.size == 0:
        return 0.0

    variance = float(gray.var())
    grad_x = 0.0
    grad_y = 0.0
    if gray.shape[1] > 1:
        dx = np.abs(np.diff(gray, axis=1))
        if dx.size > 0:
            grad_x = float(dx.mean())
    if gray.shape[0] > 1:
        dy = np.abs(np.diff(gray, axis=0))
        if dy.size > 0:
            grad_y = float(dy.mean())
    gradient_score = grad_x + grad_y

    # 분산과 경사 정보를 결합해 로그 스케일 가중치를 적용
    combined = math.log1p(max(0.0, variance)) + math.log1p(max(0.0, gradient_score))
    return combined


def _bbox_iou(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    if iw <= 0 or ih <= 0:
        return 0.0
    inter = iw * ih
    a_area = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    b_area = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    denom = a_area + b_area - inter
    if denom <= 0:
        return 0.0
    return inter / denom


def _generate_centers(length: float, patch_edge: float, overlap: float) -> List[float]:
    if length <= 0 or patch_edge <= 0:
        return [length / 2.0 if length > 0 else 0.0]
    stride = patch_edge * max(1.0 - overlap, 0.1)
    stride = max(stride, patch_edge * 0.25)
    centers: List[float] = []
    pos = 0.0
    max_start = max(0.0, length - patch_edge)
    while pos <= max_start + 1e-6:
        centers.append(pos + patch_edge / 2.0)
        pos += stride
    if not centers:
        centers.append(length / 2.0)
    return centers


def estimate_priority_regions(
    pil_image: Image.Image,
    *,
    grid_size: Optional[int] = None,
    max_regions: int = 4,
    base_cell_size: Optional[int] = None,
    max_analysis_resolution: Optional[int] = 1024,
) -> List[Tuple[float, float, float, float]]:
    """이미지 복잡도 기반 우선순위 영역을 추정한다."""

    gray = pil_image.convert("L")
    arr = np.asarray(gray, dtype=np.float32)
    if arr.size == 0:
        return []
    height, width = arr.shape
    if height == 0 or width == 0:
        return []

    if grid_size is None:
        if base_cell_size is None or base_cell_size <= 0:
            raise ValueError("base_cell_size must be provided when grid_size is None.")
        approx_grid = int(math.ceil(min(height, width) / float(base_cell_size)))
        grid_size = max(1, approx_grid)
    grid_size = max(1, grid_size)
    cell_h = max(1, height // grid_size)
    cell_w = max(1, width // grid_size)

    candidates: List[Tuple[float, Tuple[float, float, float, float]]] = []
    for row in range(grid_size):
        for col in range(grid_size):
            y1 = row * cell_h
            x1 = col * cell_w
            y2 = height if row == grid_size - 1 else min(height, y1 + cell_h)
            x2 = width if col == grid_size - 1 else min(width, x1 + cell_w)
            patch = arr[y1:y2, x1:x2]
            if patch.size == 0:
                continue
            variance = float(patch.var())
            if patch.shape[0] > 1 and patch.shape[1] > 1:
                gx = np.abs(np.diff(patch, axis=1))
                gy = np.abs(np.diff(patch, axis=0))
                gradient = float(gx.mean() + gy.mean())
            else:
                gradient = variance
            score = math.log1p(max(0.0, variance)) + math.log1p(max(0.0, gradient))
            candidates.append(
                (
                    score,
                    (
                        float(x1),
                        float(y1),
                        float(x2),
                        float(y2),
                    ),
                )
            )

    if not candidates:
        return []

    candidates.sort(key=lambda item: item[0], reverse=True)
    limit = min(max_regions, len(candidates))
    selected: List[Tuple[float, float, float, float]] = []
    seen: List[Tuple[float, float, float, float]] = []
    for score, (x1, y1, x2, y2) in candidates[: limit * 2]:
        width_pad = (x2 - x1) * 0.5
        height_pad = (y2 - y1) * 0.5
        left = max(0.0, x1 - width_pad)
        top = max(0.0, y1 - height_pad)
        right = min(float(width), x2 + width_pad)
        bottom = min(float(height), y2 + height_pad)
        if right - left <= 0 or bottom - top <= 0:
            continue
        norm_bbox = (
            left / float(width),
            top / float(height),
            right / float(width),
            bottom / float(height),
        )
        if any(_bbox_iou(norm_bbox, existing) >= 0.8 for existing in seen):
            continue
        seen.append(norm_bbox)
        selected.append(norm_bbox)
        if len(selected) >= limit:
            break

    return selected


def compute_patch_weights(
    patches: Sequence[PatchSample],
    *,
    priority_boost: float = 1.25,
    min_weight: float = 1e-4,
) -> List[float]:
    """패치 메타데이터를 기반으로 가중치를 계산한다."""

    if not patches:
        return []

    complexities = [max(0.0, float(patch.complexity)) for patch in patches]
    max_complexity = max(complexities) if complexities else 0.0
    weights: List[float] = []

    for patch, complexity in zip(patches, complexities):
        if max_complexity > 0.0:
            normalized = complexity / max_complexity
        else:
            normalized = 0.0
        weight = math.exp(normalized)
        if patch.priority:
            weight *= priority_boost
        # 스케일이 클수록 약간의 보너스를 줘 전역 문맥을 반영한다.
        weight *= 1.0 + 0.05 * max(0, patch.scale_index)
        weights.append(weight)

    total = sum(weights)
    if total <= 0.0:
        fallback = 1.0 / float(len(patches))
        return [fallback for _ in patches]

    normalized_weights = [max(min_weight, weight / total) for weight in weights]
    norm_total = sum(normalized_weights)
    if norm_total <= 0.0:
        fallback = 1.0 / float(len(patches))
        return [fallback for _ in patches]
    return [weight / norm_total for weight in normalized_weights]


def generate_patches(
    pil_image: Image.Image,
    sizes: Sequence[int] = (224, 336),
    n_patches: int = 0,
    grid_rows: Optional[Union[int, Sequence[Optional[int]]]] = None,
    grid_cols: Optional[Union[int, Sequence[Optional[int]]]] = None,
    min_cell_size: Optional[Union[int, Sequence[Optional[int]]]] = None,
    *,
    overlap: float = 0.0,
    jitter: float = 0.0,
    max_patches: Optional[int] = None,
    priority_regions: Optional[Sequence[Tuple[float, float, float, float]]] = None,
) -> List[PatchSample]:
    """원본 이미지를 고정 격자 기반으로 잘라 패치 리스트를 생성한다.

    Args:
        pil_image: 입력 이미지.
        sizes: 추론 시 사용할 스케일(픽셀 단위). 리스트 내 값이 커질수록
            더 넓은 영역을 포착한다.
        n_patches: 생성할 패치 개수.
        grid_rows: 격자 행 수를 직접 지정하고 싶을 때 사용.
        grid_cols: 격자 열 수를 직접 지정하고 싶을 때 사용.
        min_cell_size: 이미지 전체를 커버하기 위해 사용할 최소 셀 크기.

    Returns:
        PatchSample 리스트. 각 항목은 패치 이미지와 정규화된 위치,
        스케일 정보를 포함한다.
    """

    def _clamp_bbox(left: float, top: float, edge: float) -> Tuple[float, float, float, float]:
        right = min(float(w), left + edge)
        bottom = min(float(h), top + edge)
        left_clamped = max(0.0, right - edge)
        top_clamped = max(0.0, bottom - edge)
        return left_clamped, top_clamped, right, bottom

    def _merge_centers(
        centers: List[float],
        hint: Optional[int],
        length: float,
        patch_edge: float,
    ) -> List[float]:
        if hint is None or hint <= 0:
            return centers
        if len(centers) >= hint:
            return centers
        extra: List[float] = []
        if hint == 1:
            extra = [length / 2.0]
        else:
            step = (max(0.0, length - patch_edge)) / float(hint - 1)
            for idx in range(hint):
                start = min(max(0.0, step * idx), max(0.0, length - patch_edge))
                extra.append(start + patch_edge / 2.0)
        merged: List[float] = []
        seen = set()
        for value in centers + extra:
            clamped = min(max(patch_edge / 2.0, value), length - patch_edge / 2.0)
            key = round(clamped, 4)
            if key in seen:
                continue
            seen.add(key)
            merged.append(clamped)
        merged.sort()
        return merged

    def _apply_jitter(
        center: float,
        jitter_value: float,
        patch_edge: float,
        length: float,
        seed: int,
    ) -> float:
        if jitter_value <= 0.0:
            return min(max(patch_edge / 2.0, center), length - patch_edge / 2.0)
        rng = random.Random(seed)
        max_shift = jitter_value * patch_edge
        shifted = center + (rng.random() * 2.0 - 1.0) * max_shift
        return min(max(patch_edge / 2.0, shifted), length - patch_edge / 2.0)

    def _is_duplicate(new_sample: PatchSample, existing: Sequence[PatchSample], threshold: float) -> bool:
        return any(_bbox_iou(new_sample.bbox, sample.bbox) >= threshold for sample in existing)

    if pil_image.mode != "RGB":
        pil_image = pil_image.convert("RGB")

    if not sizes:
        sizes = (224,)

    sizes = tuple(int(max(1, size)) for size in sizes)
    base_size = min(sizes)
    ratios = [size / base_size for size in sizes]

    w, h = pil_image.size
    if w == 0 or h == 0:
        raise ValueError("이미지 크기가 0입니다.")

    num_scales = max(1, len(sizes))
    per_scale_cell_sizes = _normalize_per_scale(
        min_cell_size,
        num_scales,
        lambda idx: sizes[idx],
    )
    per_scale_rows_hint = _normalize_optional_ints(grid_rows, num_scales)
    per_scale_cols_hint = _normalize_optional_ints(grid_cols, num_scales)
    per_scale_overlap = _normalize_float_sequence(overlap, num_scales, default=0.0, min_value=0.0, max_value=0.9)
    per_scale_jitter = _normalize_float_sequence(jitter, num_scales, default=0.0, min_value=0.0, max_value=0.5)

    max_edge_global = float(min(w, h))
    scale_infos: List[Dict[str, object]] = []
    total_required = 0

    for idx, ratio in enumerate(ratios):
        cell = per_scale_cell_sizes[idx]
        coverage_rows = max(1, math.ceil(h / cell))
        coverage_cols = max(1, math.ceil(w / cell))
        rows_hint = per_scale_rows_hint[idx]
        cols_hint = per_scale_cols_hint[idx]
        rows = max(rows_hint or coverage_rows, coverage_rows)
        cols = max(cols_hint or coverage_cols, coverage_cols)
        step_x = w / cols
        step_y = h / rows
        base_edge = max(1.0, min(step_x, step_y))
        desired_edge = min(max(1.0, base_edge * ratio), max_edge_global)
        overlap_ratio = per_scale_overlap[idx]
        centers_x = _generate_centers(float(w), desired_edge, overlap_ratio)
        centers_y = _generate_centers(float(h), desired_edge, overlap_ratio)
        centers_x = _merge_centers(centers_x, cols_hint, float(w), desired_edge)
        centers_y = _merge_centers(centers_y, rows_hint, float(h), desired_edge)

        scale_infos.append(
            {
                "scale": sizes[idx],
                "scale_index": idx,
                "patch_edge": desired_edge,
                "centers_x": centers_x,
                "centers_y": centers_y,
                "jitter": per_scale_jitter[idx],
            }
        )
        total_required += max(1, len(centers_x)) * max(1, len(centers_y))

    if n_patches and n_patches > 0 and n_patches < total_required:
        logger.info(
            "요청한 n_patches=%d 가 전체 커버리지(%d)를 만족하지 못해 조정합니다.",
            n_patches,
            total_required,
        )

    priority_boxes: List[Tuple[float, float, float, float]] = []
    if priority_regions:
        for region in priority_regions:
            if region is None or len(region) != 4:
                continue
            x1, y1, x2, y2 = region
            left_n = float(min(x1, x2))
            top_n = float(min(y1, y2))
            right_n = float(max(x1, x2))
            bottom_n = float(max(y1, y2))
            left_n = min(max(left_n, 0.0), 1.0)
            top_n = min(max(top_n, 0.0), 1.0)
            right_n = min(max(right_n, 0.0), 1.0)
            bottom_n = min(max(bottom_n, 0.0), 1.0)
            if right_n - left_n <= 0 or bottom_n - top_n <= 0:
                continue
            priority_boxes.append((left_n, top_n, right_n, bottom_n))

    scale_candidates: Dict[int, List[PatchSample]] = defaultdict(list)
    priority_candidates: List[PatchSample] = []
    patch_counter = 0

    for info in scale_infos:
        scale_index = int(info["scale_index"])
        patch_edge = float(info["patch_edge"])
        jitter_value = float(info["jitter"])
        centers_x = list(info["centers_x"])
        centers_y = list(info["centers_y"])

        for row_idx, base_center_y in enumerate(centers_y):
            for col_idx, base_center_x in enumerate(centers_x):
                seed_base = hash((w, h, scale_index, row_idx, col_idx)) & 0xFFFFFFFF
                center_x = _apply_jitter(base_center_x, jitter_value, patch_edge, float(w), seed_base)
                center_y = _apply_jitter(base_center_y, jitter_value, patch_edge, float(h), seed_base ^ 0xABCDEF)
                left, top, right, bottom = _clamp_bbox(center_x - patch_edge / 2.0, center_y - patch_edge / 2.0, patch_edge)
                crop = pil_image.crop((left, top, right, bottom))
                bbox_norm = (
                    float(left / w),
                    float(top / h),
                    float(right / w),
                    float(bottom / h),
                )
                complexity = _estimate_complexity(crop)
                sample = PatchSample(
                    image=crop,
                    bbox=bbox_norm,
                    scale=int(info["scale"]),
                    grid_index=(row_idx, col_idx),
                    scale_index=scale_index,
                    patch_index=patch_counter,
                    priority=False,
                    complexity=complexity,
                    source="grid",
                )
                scale_candidates[scale_index].append(sample)
                patch_counter += 1

    for box in priority_boxes:
        left_n, top_n, right_n, bottom_n = box
        left = left_n * w
        top = top_n * h
        right = right_n * w
        bottom = bottom_n * h
        region_center_x = (left + right) / 2.0
        region_center_y = (top + bottom) / 2.0
        region_edge = max(right - left, bottom - top)

        # 스케일 중 가장 근접한 패치 크기를 선택한다.
        target_index = 0
        best_diff = float("inf")
        for info in scale_infos:
            patch_edge = float(info["patch_edge"])
            diff = abs(patch_edge - region_edge)
            if diff < best_diff:
                best_diff = diff
                target_index = int(info["scale_index"])
        info = scale_infos[target_index]
        patch_edge = float(info["patch_edge"])
        patch_edge = min(max(region_edge * 1.1, patch_edge), max_edge_global)
        if patch_edge <= 1.0:
            patch_edge = 1.0

        center_x = min(max(patch_edge / 2.0, region_center_x), w - patch_edge / 2.0)
        center_y = min(max(patch_edge / 2.0, region_center_y), h - patch_edge / 2.0)
        left, top, right, bottom = _clamp_bbox(center_x - patch_edge / 2.0, center_y - patch_edge / 2.0, patch_edge)
        crop = pil_image.crop((left, top, right, bottom))
        bbox_norm = (
            float(left / w),
            float(top / h),
            float(right / w),
            float(bottom / h),
        )

        centers_x = list(info["centers_x"]) or [center_x]
        centers_y = list(info["centers_y"]) or [center_y]

        def _nearest_index(values: Sequence[float], point: float) -> int:
            if not values:
                return 0
            best_idx = 0
            best_dist = float("inf")
            for idx, value in enumerate(values):
                dist = abs(value - point)
                if dist < best_dist:
                    best_dist = dist
                    best_idx = idx
            return best_idx

        row_idx = _nearest_index(centers_y, center_y)
        col_idx = _nearest_index(centers_x, center_x)
        complexity = _estimate_complexity(crop)
        sample = PatchSample(
            image=crop,
            bbox=bbox_norm,
            scale=int(info["scale"]),
            grid_index=(row_idx, col_idx),
            scale_index=int(info["scale_index"]),
            patch_index=patch_counter,
            priority=True,
            complexity=complexity,
            source="priority",
        )
        priority_candidates.append(sample)
        patch_counter += 1

    limit = None
    if max_patches and max_patches > 0:
        limit = max(max_patches, len(priority_candidates))
    elif n_patches and n_patches > 0:
        limit = max(n_patches, len(priority_candidates))

    for idx in scale_candidates:
        scale_candidates[idx].sort(key=lambda sample: sample.complexity, reverse=True)

    selected: List[PatchSample] = []
    selected_by_scale: Dict[int, int] = defaultdict(int)

    if priority_candidates:
        priority_sorted = sorted(priority_candidates, key=lambda sample: sample.complexity, reverse=True)
        for sample in priority_sorted:
            if _is_duplicate(sample, selected, threshold=0.95):
                continue
            selected.append(sample)
            selected_by_scale[sample.scale_index] += 1

    working_candidates: Dict[int, List[PatchSample]] = {
        idx: list(candidates) for idx, candidates in scale_candidates.items()
    }

    while True:
        if limit is not None and len(selected) >= limit:
            break
        best_scale = None
        best_score: Tuple[float, float, int] | None = None
        for idx, candidates in working_candidates.items():
            while candidates and _is_duplicate(candidates[0], selected, threshold=0.7):
                candidates.pop(0)
            if not candidates:
                continue
            candidate = candidates[0]
            coverage = selected_by_scale.get(idx, 0)
            score = (-coverage, candidate.complexity, -idx)
            if best_score is None or score > best_score:
                best_score = score
                best_scale = idx
        if best_scale is None:
            break
        candidate = working_candidates[best_scale].pop(0)
        selected.append(candidate)
        selected_by_scale[best_scale] += 1

    # 스케일 누락 시 최고 복잡도 패치를 추가로 보완
    for info in scale_infos:
        idx = int(info["scale_index"])
        if selected_by_scale.get(idx, 0) > 0:
            continue
        fallback_list = scale_candidates.get(idx, [])
        for candidate in fallback_list:
            if _is_duplicate(candidate, selected, threshold=0.9):
                continue
            if limit is not None and len(selected) >= limit:
                break
            selected.append(candidate)
            selected_by_scale[idx] += 1
            break

    final_samples: List[PatchSample] = []
    for new_index, sample in enumerate(selected):
        final_samples.append(
            replace(
                sample,
                patch_index=new_index,
            )
        )

    final_samples.sort(key=lambda sample: sample.patch_index)
    logger.debug(
        "멀티패치 생성 - total=%d priority=%d limit=%s scales=%s",
        len(final_samples),
        len(priority_candidates),
        str(limit) if limit is not None else "auto",
        list(sizes),
    )
    return final_samples


def _suggest_grid(n_patches: int) -> Tuple[int, int]:
    """요청된 패치 수에 맞는 격자 크기를 추정한다."""

    if n_patches <= 0:
        return 1, 1

    best_rows = 1
    best_cols = n_patches
    best_difference = best_cols - best_rows

    for rows in range(1, n_patches + 1):
        cols = -(-n_patches // rows)  # ceil division
        cells = rows * cols
        if cells < n_patches:
            continue
        difference = abs(rows - cols)
        current_area = rows * cols
        best_area = best_rows * best_cols
        if current_area < best_area or (current_area == best_area and difference < best_difference):
            best_rows, best_cols = rows, cols
            best_difference = difference
        if rows > n_patches:
            break
    return best_rows, best_cols


def infer_patches(
    model: nn.Module,
    patches: Iterable[Union[PatchSample, Image.Image]],
    device: str = "cuda",
    fp16: bool = True,
    preprocess: Optional[Callable[[Image.Image], torch.Tensor]] = None,
) -> List[dict]:
    """여러 패치를 모델에 통과시켜 점수를 반환한다."""

    model.eval()
    model.to(device)
    transform = preprocess or _DEFAULT_INFER_TRANSFORM

    scores: List[dict] = []
    from contextlib import nullcontext

    use_cuda = device.startswith("cuda")
    autocast = torch.cuda.amp.autocast if use_cuda and fp16 else nullcontext
    with torch.inference_mode():
        with autocast():
            for patch in patches:
                image = patch.image if isinstance(patch, PatchSample) else patch
                rgb_image = image if image.mode == "RGB" else image.convert("RGB")
                tensor = transform(rgb_image).unsqueeze(0).to(device)
                logits = model(tensor)
                probs = torch.softmax(logits, dim=1)[0]
                scores.append({"ai": float(probs[1].item()), "real": float(probs[0].item())})
    return scores


def aggregate_scores(
    scores: Sequence[dict],
    method: str = "mean",
    weights: Optional[Sequence[float]] = None,
) -> dict:
    """패치별 점수를 하나의 확률로 통합한다.

    SNS 재압축으로 인해 패치마다 확신도가 달라질 수 있으므로, 평균/최대/
    품질 가중 방식 중 상황에 맞게 선택할 수 있도록 했다.
    """

    if not scores:
        raise ValueError("scores 리스트가 비어 있습니다.")

    ai_scores = torch.tensor([s["ai"] for s in scores])
    real_scores = torch.tensor([s["real"] for s in scores])

    if weights is not None and len(weights) != len(scores):
        raise ValueError("weights length must match scores length")

    weight_tensor: Optional[torch.Tensor] = None
    if weights is not None:
        weight_tensor = torch.tensor(weights, dtype=torch.float32)
        if torch.any(weight_tensor < 0):
            weight_tensor = torch.clamp(weight_tensor, min=0.0)
        weight_sum = torch.sum(weight_tensor)
        if weight_sum <= 0:
            weight_tensor = None
        else:
            weight_tensor = weight_tensor / weight_sum

    if method == "max":
        ai = ai_scores.max().item()
        real = real_scores.min().item()
    elif method == "quality_weighted":
        weights = torch.linspace(1.0, 2.0, steps=len(scores))
        weights = weights / weights.sum()
        ai = (ai_scores * weights).sum().item()
        real = (real_scores * weights).sum().item()
    elif weight_tensor is not None:
        ai = (ai_scores * weight_tensor.to(ai_scores.device)).sum().item()
        real = (real_scores * weight_tensor.to(real_scores.device)).sum().item()
    else:
        ai = ai_scores.mean().item()
        real = real_scores.mean().item()
    total = ai + real + 1e-8
    aggregated = {"ai": ai / total, "real": real / total}
    logger.debug("패치 점수 집계 - method=%s, ai=%.4f, real=%.4f", method, aggregated["ai"], aggregated["real"])
    return aggregated
