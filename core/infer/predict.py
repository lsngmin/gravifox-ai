"""모델 추론 스니펫 모듈."""

from __future__ import annotations

import time
from typing import Dict, Sequence, Tuple

import torch
from PIL import Image
from torchvision import transforms

from core.models.multipatch import aggregate_scores, generate_patches, infer_patches
from core.models.registry import get_model
from core.utils.logger import get_logger

logger = get_logger(__name__)


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


def _run_single(image: Image.Image, model: torch.nn.Module, device: str) -> Dict[str, float]:
    """단일 이미지를 전처리하여 모델 점수를 얻는다."""

    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])
    tensor = preprocess(image).unsqueeze(0).to(device)
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
    scales: Sequence[int] = (224, 336),
    device: str = "cuda",
    uncertain_band: Tuple[float, float] = (0.45, 0.55),
) -> Dict[str, object]:
    """PIL 이미지를 받아 진위 여부를 추론한다."""

    start = time.perf_counter()
    patch_count = 1
    if mode == "multi":
        min_cell_size = min(scales) if scales else 224
        patches = generate_patches(
            pil_image,
            sizes=scales,
            n_patches=n_patches,
            min_cell_size=min_cell_size,
        )
        patch_count = max(1, len(patches))
        patch_scores = infer_patches(model, patches, device=device)
        scores = aggregate_scores(patch_scores)
    else:
        scores = _run_single(pil_image, model, device)

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
            "latency_ms": float(latency),
        },
    }
    logger.info(
        "추론 완료 - class=%s confidence=%.2f latency=%.1fms",
        result["class"],
        result["confidence"],
        latency,
    )
    return result
