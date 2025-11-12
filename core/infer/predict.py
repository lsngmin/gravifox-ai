"""모델 추론 스니펫 모듈."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Tuple, Union

import torch
import yaml
from PIL import Image
from torchvision import transforms

from core.datasets.base import DatasetConfig, load_dataset_config
from core.datasets.transforms import build_val_transforms
from core.models.multipatch import (
    aggregate_scores,
    compute_patch_weights,
    estimate_priority_regions,
    generate_patches,
    infer_patches,
)
from core.models.registry import get_model
from core.utils.logger import get_logger

logger = get_logger(__name__)
_DEFAULT_PREPROCESS = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        ),
    ]
)
_DEFAULT_SCALES: Tuple[int, int] = (224, 336)
_RKMV1_SCALE: int = 512
_RKMV1_PATCHES: int = 3


def _is_rkmv1(model: torch.nn.Module) -> bool:
    """모델 버전/이름을 기준으로 RKMv1 계열인지 판별한다."""

    version = str(getattr(model, "model_version", "")).lower()
    name = model.__class__.__name__.lower()
    if "rkmv1" in version or version.startswith("rkm"):
        return True
    return "rkmv1" in name or name.startswith("rkm")


def build_inference_transform(
    dataset: DatasetConfig | Mapping[str, Any] | str | Path | None = None
) -> Callable[[Image.Image], torch.Tensor]:
    """데이터셋 구성을 받아 추론용 전처리를 생성한다."""

    cfg = _normalize_dataset_config(dataset)
    return build_val_transforms(cfg)


def _normalize_dataset_config(
    dataset: DatasetConfig | Mapping[str, Any] | str | Path | None
) -> DatasetConfig:
    """DatasetConfig 입력을 표준 형태로 정규화한다."""

    if dataset is None:
        return DatasetConfig()
    if isinstance(dataset, DatasetConfig):
        return dataset
    if isinstance(dataset, (str, Path)):
        path = Path(dataset).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"dataset config not found: {path}")
        with path.open("r", encoding="utf-8") as fp:
            payload = yaml.safe_load(fp) or {}
        return load_dataset_config(payload)
    if isinstance(dataset, Mapping):
        return load_dataset_config(dataset)
    raise TypeError(f"Unsupported dataset config type: {type(dataset)!r}")


def load_model_from_checkpoint(model_name: str, ckpt_path: str, device: str = "cuda") -> Tuple[torch.nn.Module, str]:
    """체크포인트에서 모델을 로드한다."""

    model = get_model(model_name)
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    state_dict = checkpoint.get("model", checkpoint)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        logger.warning("체크포인트 로드 경고 - missing=%s unexpected=%s", missing, unexpected)
    model.to(device)
    version = getattr(model, "model_version", model_name)
    logger.info("모델 로드 완료 - %s (%s)", model_name, version)
    return model, version


def _resolve_preprocess(
    transform: Optional[Callable[[Image.Image], torch.Tensor]]
) -> Callable[[Image.Image], torch.Tensor]:
    """None 입력 시 기본 전처리를 반환한다."""

    return transform or _DEFAULT_PREPROCESS


def _run_single(
    image: Image.Image,
    model: torch.nn.Module,
    device: str,
    transform: Optional[Callable[[Image.Image], torch.Tensor]] = None,
) -> Dict[str, float]:
    """단일 이미지를 전처리하여 모델 점수를 얻는다."""

    preprocess = _resolve_preprocess(transform)
    rgb_image = image if image.mode == "RGB" else image.convert("RGB")
    tensor = preprocess(rgb_image).unsqueeze(0).to(device)
    model.eval()
    with torch.inference_mode():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1)[0]
    return {"ai": float(probs[1].item()), "real": float(probs[0].item())}


def run_inference(
    pil_image: Image.Image,
    model: torch.nn.Module,
    mode: str = "single",
    n_patches: int = 0,
    scales: Sequence[int] = _DEFAULT_SCALES,
    cell_sizes: Optional[Sequence[int]] = None,
    device: str = "cuda",
    uncertain_band: Tuple[float, float] = (0.45, 0.55),
    overlap: Union[float, Sequence[Optional[float]]] = 0.0,
    jitter: Union[float, Sequence[Optional[float]]] = 0.0,
    max_patches: Optional[int] = None,
    priority_regions: Optional[Sequence[Tuple[float, float, float, float]]] = None,
    transform: Optional[Callable[[Image.Image], torch.Tensor]] = None,
    aggregate_method: str = "mean",
    fp16: bool = True,
) -> Dict[str, object]:
    """PIL 이미지를 받아 진위 여부를 추론한다.

    Args:
        pil_image: 입력 이미지.
        model: 추론에 사용할 모델.
        mode: "single" 또는 "multi".
        n_patches: 멀티패치 요청 수(0이면 자동).
        scales: 패치 생성 시 사용할 스케일 목록.
        cell_sizes: 스케일별 최소 셀 크기.
        device: torch 디바이스 문자열.
        uncertain_band: 불확실 구간 임계값 (low, high).
        overlap: 패치 중첩 비율(단일 값 또는 스케일별 리스트).
        jitter: 패치 중심 좌표에 적용할 지터 비율.
        max_patches: 패치 수 상한.
        priority_regions: 우선 탐색할 정규화 영역 리스트.
        transform: PIL 이미지를 텐서로 변환할 전처리 함수.
        aggregate_method: 패치 점수 집계 방식(mean|max|quality_weighted 등).
        fp16: CUDA 추론 시 autocast 사용 여부.
    """

    start = time.perf_counter()
    patch_count = 1
    mode = (mode or "single").lower()
    normalized_scales = tuple(int(s) for s in scales) if scales else ()
    scales = normalized_scales or _DEFAULT_SCALES
    min_cell_values: Optional[Sequence[int]] = (
        tuple(int(s) for s in cell_sizes) if cell_sizes is not None else None
    )
    computed_priority_regions: Optional[Sequence[Tuple[float, float, float, float]]] = None
    is_rkmv1 = _is_rkmv1(model)

    if is_rkmv1:
        if not scales or scales == _DEFAULT_SCALES:
            scales = (_RKMV1_SCALE,)
        if (
            min_cell_values is None
            or len(min_cell_values) != len(scales)
            or any(int(val) <= 0 for val in min_cell_values)
        ):
            min_cell_values = tuple([_RKMV1_SCALE] * len(scales))
        if n_patches <= 0:
            n_patches = _RKMV1_PATCHES
        if max_patches is None:
            max_patches = _RKMV1_PATCHES
        if mode != "multi":
            mode = "multi"

    if mode == "multi":
        if min_cell_values is None:
            min_cell_values = tuple(scales)
        computed_priority_regions = priority_regions
        base_cell_size: Optional[int] = None
        if isinstance(min_cell_values, (list, tuple)):
            for value in min_cell_values:
                if value:
                    base_cell_size = int(value)
                    break
        elif min_cell_values:
            base_cell_size = int(min_cell_values)
        if computed_priority_regions is None:
            computed_priority_regions = estimate_priority_regions(
                pil_image,
                base_cell_size=base_cell_size,
            )
        patches = generate_patches(
            pil_image,
            sizes=scales,
            n_patches=n_patches,
            min_cell_size=min_cell_values,
            overlap=overlap,
            jitter=jitter,
            max_patches=max_patches,
            priority_regions=computed_priority_regions,
        )
        patch_count = max(1, len(patches))
        patch_scores = infer_patches(
            model,
            patches,
            device=device,
            fp16=fp16,
            preprocess=_resolve_preprocess(transform),
        )
        weights = compute_patch_weights(patches)
        scores = aggregate_scores(
            patch_scores,
            weights=weights if weights else None,
            method=aggregate_method,
        )
    else:
        scores = _run_single(pil_image, model, device, transform)

    ai_score = scores["ai"]
    low, high = uncertain_band
    if ai_score >= high:
        predicted = "AI"
        confidence = ai_score
    elif ai_score <= low:
        predicted = "Real"
        confidence = scores["real"]
    else:
        predicted = "Uncertain"
        confidence = max(ai_score, scores["real"])

    if isinstance(overlap, (list, tuple)):
        overlap_meta: Optional[object] = [
            float(v) if v is not None else None for v in overlap
        ]
    else:
        try:
            overlap_meta = float(overlap) if overlap is not None else None
        except (TypeError, ValueError):
            overlap_meta = None

    if isinstance(jitter, (list, tuple)):
        jitter_meta: Optional[object] = [
            float(v) if v is not None else None for v in jitter
        ]
    else:
        try:
            jitter_meta = float(jitter) if jitter is not None else None
        except (TypeError, ValueError):
            jitter_meta = None

    try:
        max_patches_meta = int(max_patches) if max_patches is not None else None
    except (TypeError, ValueError):
        max_patches_meta = None

    latency = (time.perf_counter() - start) * 1000.0
    result = {
        "class": predicted,
        "confidence": float(confidence),
        "scores": scores,
        "model_version": getattr(model, "model_version", "unknown"),
        "inference": {
            "mode": mode,
            "n_patches": patch_count if mode == "multi" else 1,
            "scales": list(scales),
            "cell_sizes": list(min_cell_values) if mode == "multi" and min_cell_values is not None else None,
            "latency_ms": float(latency),
            "overlap": overlap_meta,
            "jitter": jitter_meta,
            "max_patches": max_patches_meta,
            "aggregate": aggregate_method,
            "priority_regions": (
                [
                    [float(coord) for coord in region]
                    for region in computed_priority_regions
                ]
                if computed_priority_regions
                else None
            ),
        },
    }
    logger.info(
        "추론 완료 - class=%s confidence=%.2f latency=%.1fms",
        result["class"],
        result["confidence"],
        latency,
    )
    return result
