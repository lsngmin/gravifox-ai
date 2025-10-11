"""이미지 관련 FastAPI 라우터."""

from __future__ import annotations

import io
import time
from typing import Optional

from fastapi import APIRouter, Depends, File, Form, Header, HTTPException, UploadFile
from PIL import Image

from api.config import RuntimeSettings
from api.dependencies.inference import (
    get_calibrator,
    get_model_registry,
    get_runtime_settings,
    get_storage_service,
    get_vit_service,
    get_upload_token_registry,
    get_upload_token_verifier,
)
from api.schemas.image import (
    ImagePredictionResponse,
    ModelItem,
    ModelListResponse,
    UploadResponse,
)
from api.services.calibration import AdaptiveThresholdCalibrator
from api.services.inference import VitInferenceService
from api.services.registry import ModelRegistryService
from api.services.storage import MediaStorageService
from api.services.upload_token import UploadTokenRegistryClient, UploadTokenVerifier

router = APIRouter()


@router.post("/predict/image", response_model=ImagePredictionResponse)
async def predict_image(
    file: UploadFile = File(...),
    vit_service: VitInferenceService = Depends(get_vit_service),
    calibrator: AdaptiveThresholdCalibrator = Depends(get_calibrator),
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
    calibration = calibrator.calibrate(probs, pipeline.real_index)
    latency_ms = (time.perf_counter() - start) * 1000.0
    return ImagePredictionResponse(
        timestamp=time.time(),
        model_version=pipeline.model_name,
        latency_ms=round(latency_ms, 2),
        p_real=round(calibration.p_real, 6),
        p_ai=round(calibration.p_ai, 6),
        confidence=round(calibration.confidence, 6),
        decision=calibration.decision,
        class_names=pipeline.class_names,
        probabilities=[float(round(p, 6)) for p in calibration.distribution],
    )


@router.post("/upload", response_model=UploadResponse)
async def upload_media(
    file: UploadFile = File(...),
    upload_id: Optional[str] = Form(default=None, alias="uploadId"),
    storage: MediaStorageService = Depends(get_storage_service),
    settings: RuntimeSettings = Depends(get_runtime_settings),
    verifier: UploadTokenVerifier = Depends(get_upload_token_verifier),
    registry: UploadTokenRegistryClient = Depends(get_upload_token_registry),
    upload_token: Optional[str] = Header(default=None, alias="Upload-Token"),
    authorization: Optional[str] = Header(default=None, alias="Authorization"),
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

    resolved_token = _resolve_upload_token(upload_token, authorization)
    legacy_token = (settings.upload_token or "").strip()
    if not resolved_token:
        raise HTTPException(status_code=401, detail="upload token required")

    provided_upload_id = (upload_id or "").strip() or None
    registry_context = None

    if legacy_token:
        if resolved_token != legacy_token:
            raise HTTPException(status_code=401, detail="invalid upload token")
    else:
        claims = await verifier.verify(resolved_token)
        if provided_upload_id and provided_upload_id != claims.upload_id:
            raise HTTPException(status_code=409, detail="uploadId mismatch")
        provided_upload_id = provided_upload_id or claims.upload_id
        registry_context = await registry.authorize(resolved_token)
        if (
            registry_context is not None
            and registry_context.upload_id != claims.upload_id
        ):
            await registry.complete_failure(
                registry_context, "uploadId mismatch"
            )
            raise HTTPException(status_code=409, detail="upload token mismatch")

    kind = storage.infer_media_kind(
        file.filename or "", getattr(file, "content_type", None)
    )
    try:
        saved_upload_id = await storage.save_upload(
            file, kind, upload_id=provided_upload_id
        )
    except HTTPException as exc:
        await registry.complete_failure(registry_context, str(exc.detail))
        raise
    except Exception as exc:  # pragma: no cover - 파일 시스템 예외 대비
        await registry.complete_failure(registry_context, "upload failed")
        raise

    await registry.complete_success(registry_context)
    return UploadResponse(uploadId=saved_upload_id)


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


def _resolve_upload_token(
    upload_token: Optional[str], authorization: Optional[str]
) -> Optional[str]:
    if upload_token and upload_token.strip():
        return upload_token.strip()
    if authorization:
        candidate = authorization.strip()
        if candidate.lower().startswith("bearer "):
            token = candidate[7:].strip()
            if token:
                return token
    return None
