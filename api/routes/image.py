"""이미지 관련 FastAPI 라우터."""

from __future__ import annotations

import io
import time
from typing import Optional

from fastapi import APIRouter, Depends, File, Header, HTTPException, UploadFile
from PIL import Image

from api.dependencies.inference import (
    get_model_registry,
    get_runtime_settings,
    get_storage_service,
    get_vit_service,
)
from api.schemas.image import (
    ImagePredictionResponse,
    ModelItem,
    ModelListResponse,
    UploadResponse,
)
from api.services.inference import VitInferenceService
from api.services.registry import ModelRegistryService
from api.services.storage import MediaStorageService

router = APIRouter()


@router.post("/predict/image", response_model=ImagePredictionResponse)
async def predict_image(
    file: UploadFile = File(...),
    vit_service: VitInferenceService = Depends(get_vit_service),
) -> ImagePredictionResponse:
    """이미지 위조 확률을 예측한다.

    Args:
        file: 업로드된 이미지 파일.
        vit_service: ViT 추론 서비스 인스턴스.

    Returns:
        이미지 추론 결과 응답 모델.
    """

    raw_bytes = await file.read()
    try:
        image = Image.open(io.BytesIO(raw_bytes))
    except Exception as exc:  # pragma: no cover - Pillow 예외 메시지 의존
        raise HTTPException(status_code=400, detail="invalid image file") from exc
    start = time.perf_counter()
    try:
        probs = await vit_service.predict_image(image)
    finally:
        image.close()
    pipeline = await vit_service.get_pipeline()
    real_index = pipeline.real_index if pipeline.real_index < len(probs) else 0
    p_real = float(probs[real_index]) if probs else 0.0
    p_ai = float(1.0 - p_real)
    confidence = float(max(probs)) if probs else 0.0
    latency_ms = (time.perf_counter() - start) * 1000.0
    return ImagePredictionResponse(
        timestamp=time.time(),
        model_version=pipeline.model_name,
        latency_ms=round(latency_ms, 2),
        p_real=round(p_real, 6),
        p_ai=round(p_ai, 6),
        confidence=round(confidence, 6),
        class_names=pipeline.class_names,
        probabilities=[float(round(p, 6)) for p in probs],
    )


@router.post("/upload", response_model=UploadResponse)
async def upload_media(
    file: UploadFile = File(...),
    storage: MediaStorageService = Depends(get_storage_service),
    token: Optional[str] = Header(default=None, alias="X-Upload-Token"),
    settings=Depends(get_runtime_settings),
) -> UploadResponse:
    """이미지 또는 동영상을 업로드하여 식별자를 반환한다.

    Args:
        file: 업로드된 미디어 파일.
        storage: 파일 저장 서비스.
        token: 업로드 토큰 헤더 값.
        settings: 런타임 설정 의존성.

    Returns:
        업로드 식별자 응답 모델.
    """

    if settings.upload_token and token != settings.upload_token:
        raise HTTPException(status_code=401, detail="invalid upload token")
    kind = storage.infer_media_kind(
        file.filename or "", getattr(file, "content_type", None)
    )
    upload_id = await storage.save_upload(file, kind)
    return UploadResponse(uploadId=upload_id)


@router.get("/models", response_model=ModelListResponse)
async def list_models(
    registry: ModelRegistryService = Depends(get_model_registry),
) -> ModelListResponse:
    """사용 가능한 모델 목록을 반환한다.

    Args:
        registry: 모델 카탈로그 서비스.

    Returns:
        모델 목록 응답 모델.
    """

    items = registry.list_models()
    default_model = registry.get_default_model()
    payload = [
        ModelItem(
            key=model.key,
            name=model.name,
            version=model.version,
            description=model.description,
            type=model.type,
            input=model.input,
            threshold=model.threshold,
            labels=list(model.labels),
        )
        for model in items
    ]
    return ModelListResponse(defaultKey=default_model.key, items=payload)


__all__ = ["router"]
