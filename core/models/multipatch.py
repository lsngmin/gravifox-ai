"""멀티패치·멀티스케일 추론 유틸리티와 간단한 ViT 멀티패치 모델."""

from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple, Union

import timm
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms

from core.utils.logger import get_logger
from .registry import register


logger = get_logger(__name__)


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


def generate_patches(
    pil_image: Image.Image,
    sizes: Sequence[int] = (224, 336),
    n_patches: int = 0,
    grid_rows: Optional[Union[int, Sequence[Optional[int]]]] = None,
    grid_cols: Optional[Union[int, Sequence[Optional[int]]]] = None,
    min_cell_size: Optional[Union[int, Sequence[Optional[int]]]] = None,
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

    layouts: List[Tuple[int, int, float, float, float]] = []
    total_required = 0
    max_edge_global = float(min(w, h))

    for idx in range(num_scales):
        cell = per_scale_cell_sizes[idx]
        coverage_rows = max(1, math.ceil(h / cell))
        coverage_cols = max(1, math.ceil(w / cell))
        rows = max(per_scale_rows_hint[idx] or coverage_rows, coverage_rows)
        cols = max(per_scale_cols_hint[idx] or coverage_cols, coverage_cols)
        step_x = w / cols
        step_y = h / rows
        base_edge = max(1.0, min(step_x, step_y))
        layouts.append((rows, cols, step_x, step_y, base_edge))
        total_required += rows * cols

    if n_patches and n_patches > 0 and n_patches < total_required:
        logger.info(
            "요청한 n_patches=%d 가 전체 커버리지(%d)를 만족하지 못해 조정합니다.",
            n_patches,
            total_required,
        )
        n_patches = total_required

    samples: List[PatchSample] = []
    patch_counter = 0

    for scale_index, ratio in enumerate(ratios):
        rows, cols, step_x, step_y, base_edge = layouts[scale_index]
        for row in range(rows):
            center_y = (row + 0.5) * step_y
            for col in range(cols):
                center_x = (col + 0.5) * step_x

                desired_edge = min(base_edge * ratio, max_edge_global)
                if desired_edge < 1.0:
                    desired_edge = 1.0
                half = desired_edge / 2.0

                left = max(0.0, min(center_x - half, w - desired_edge))
                top = max(0.0, min(center_y - half, h - desired_edge))
                right = min(w, left + desired_edge)
                bottom = min(h, top + desired_edge)

                # 보정 이후에도 간격을 유지하기 위해 재계산
                left = max(0.0, right - desired_edge)
                top = max(0.0, bottom - desired_edge)

                crop = pil_image.crop((left, top, right, bottom))
                bbox_norm = (
                    float(left / w),
                    float(top / h),
                    float(right / w),
                    float(bottom / h),
                )
                scale_value = sizes[scale_index]

                samples.append(
                    PatchSample(
                        image=crop,
                        bbox=bbox_norm,
                        scale=scale_value,
                        grid_index=(row, col),
                        scale_index=scale_index,
                        patch_index=patch_counter,
                    )
                )
                patch_counter += 1

                if n_patches and n_patches > 0 and len(samples) >= n_patches:
                    logger.info(
                        "격자 기반 멀티패치 생성 (제한 적용) - 생성=%d 제한=%d 스케일=%s",
                        len(samples),
                        n_patches,
                        list(sizes),
                    )
                    return samples[:n_patches]

    logger.info(
        "격자 기반 멀티패치 생성 - 총 %d개, 스케일=%s",
        len(samples),
        list(sizes),
    )
    return samples


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
) -> List[dict]:
    """여러 패치를 모델에 통과시켜 점수를 반환한다."""

    model.eval()
    model.to(device)
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])

    scores: List[dict] = []
    from contextlib import nullcontext

    use_cuda = device.startswith("cuda")
    autocast = torch.cuda.amp.autocast if use_cuda and fp16 else nullcontext
    with torch.inference_mode():
        with autocast():
            for patch in patches:
                image = patch.image if isinstance(patch, PatchSample) else patch
                tensor = preprocess(image).unsqueeze(0).to(device)
                logits = model(tensor)
                probs = torch.softmax(logits, dim=1)[0]
                scores.append({"ai": float(probs[1].item()), "real": float(probs[0].item())})
    return scores


def aggregate_scores(scores: Sequence[dict], method: str = "mean") -> dict:
    """패치별 점수를 하나의 확률로 통합한다.

    SNS 재압축으로 인해 패치마다 확신도가 달라질 수 있으므로, 평균/최대/
    품질 가중 방식 중 상황에 맞게 선택할 수 있도록 했다.
    """

    if not scores:
        raise ValueError("scores 리스트가 비어 있습니다.")

    ai_scores = torch.tensor([s["ai"] for s in scores])
    real_scores = torch.tensor([s["real"] for s in scores])

    if method == "max":
        ai = ai_scores.max().item()
        real = real_scores.min().item()
    elif method == "quality_weighted":
        weights = torch.linspace(1.0, 2.0, steps=len(scores))
        weights = weights / weights.sum()
        ai = (ai_scores * weights).sum().item()
        real = (real_scores * weights).sum().item()
    else:
        ai = ai_scores.mean().item()
        real = real_scores.mean().item()
    total = ai + real + 1e-8
    aggregated = {"ai": ai / total, "real": real / total}
    logger.info("패치 점수 집계 - method=%s, ai=%.4f, real=%.4f", method, aggregated["ai"], aggregated["real"])
    return aggregated
