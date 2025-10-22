"""이미지 관련 FastAPI 라우터."""

from __future__ import annotations

import io
import logging
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
logger = logging.getLogger("tvb.upload")


@router.post("/predict/image", response_model=ImagePredictionResponse)
async def predict_image(
    file: UploadFile = File(...),
    model_key: Optional[str] = Form(default=None, alias="modelKey"),
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
        probs = await vit_service.predict_image(image, model_key=model_key)
    finally:
        image.close()
    pipeline = await vit_service.get_pipeline(model_key=model_key)
    calibration = calibrator.calibrate(probs, pipeline.real_index)
    latency_ms = (time.perf_counter() - start) * 1000.0
    model_version = (
        pipeline.model_info.version or pipeline.model_name or pipeline.model_info.key
    )
    return ImagePredictionResponse(
        timestamp=time.time(),
        model_version=str(model_version),
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

    provided_upload_id = (upload_id or "").strip() or None
    registry_context = None
    logger.info(
        "[UploadPipeline] request filename=%s uploadId=%s contentType=%s",
        file.filename,
        provided_upload_id or "<none>",
        getattr(file, "content_type", None),
    )

    resolved_token = _resolve_upload_token(upload_token, authorization)
    legacy_token = (settings.upload_token or "").strip()
    logger.info(
        "[UploadPipeline] tokens header=%s authorization=%s resolved=%s",
        _mask_token(upload_token),
        _mask_bearer(authorization),
        _mask_token(resolved_token),
    )

    if settings.upload_token_disabled:
        expected = (settings.upload_token_disabled_token or legacy_token).strip()
        if expected and resolved_token and resolved_token != expected:
            logger.warning(
                "[UploadPipeline] disabled-mode token mismatch expected=%s actual=%s",
                _mask_token(expected),
                _mask_token(resolved_token),
            )
            raise HTTPException(status_code=401, detail="invalid upload token")
        resolved_token = resolved_token or expected or ""
    else:
        if not resolved_token:
            logger.warning("[UploadPipeline] upload token missing")
            raise HTTPException(status_code=401, detail="upload token required")
        if legacy_token:
            if resolved_token != legacy_token:
                logger.warning(
                    "[UploadPipeline] legacy token mismatch expected=%s actual=%s",
                    _mask_token(legacy_token),
                    _mask_token(resolved_token),
                )
                raise HTTPException(status_code=401, detail="invalid upload token")
        else:
            claims = await verifier.verify(resolved_token)
            logger.info(
                "[UploadPipeline] verified upload token uploadId=%s jti=%s exp=%s",
                claims.upload_id,
                claims.jti,
                claims.expires_at.isoformat(),
            )
            if provided_upload_id and provided_upload_id != claims.upload_id:
                logger.warning(
                    "[UploadPipeline] uploadId mismatch provided=%s claims=%s",
                    provided_upload_id,
                    claims.upload_id,
                )
                raise HTTPException(status_code=409, detail="uploadId mismatch")
            provided_upload_id = provided_upload_id or claims.upload_id
            registry_context = await registry.authorize(resolved_token)
            if (
                registry_context is not None
                and registry_context.upload_id != claims.upload_id
            ):
                logger.warning(
                    "[UploadPipeline] registry mismatch uploadId=%s registry=%s",
                    claims.upload_id,
                    registry_context.upload_id,
                )
                await registry.complete_failure(
                    registry_context, "uploadId mismatch"
                )
                raise HTTPException(status_code=409, detail="upload token mismatch")
            if registry_context is not None:
                logger.info(
                    "[UploadPipeline] registry authorized tokenId=%s uploadId=%s jti=%s",
                    registry_context.token_id,
                    registry_context.upload_id,
                    registry_context.jti,
                )

    kind = storage.infer_media_kind(
        file.filename or "", getattr(file, "content_type", None)
    )
    try:
        saved_upload_id = await storage.save_upload(
            file, kind, upload_id=provided_upload_id
        )
    except HTTPException as exc:
        logger.exception(
            "[UploadPipeline] upload rejected uploadId=%s reason=%s",
            provided_upload_id,
            getattr(exc, "detail", None),
        )
        await registry.complete_failure(registry_context, str(exc.detail))
        raise
    except Exception as exc:  # pragma: no cover - 파일 시스템 예외 대비
        logger.exception(
            "[UploadPipeline] upload failed uploadId=%s", provided_upload_id
        )
        await registry.complete_failure(registry_context, "upload failed")
        raise

    await registry.complete_success(registry_context)
    logger.info(
        "[UploadPipeline] upload stored uploadId=%s kind=%s", saved_upload_id, kind
    )
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


def _mask_token(token: Optional[str]) -> str:
    if not token:
        return "null"
    trimmed = token.strip()
    if len(trimmed) <= 8:
        return f"{trimmed[0]}***"
    return f"{trimmed[:4]}…{trimmed[-4:]}"


def _mask_bearer(header: Optional[str]) -> str:
    if not header:
        return "null"
    value = header.strip()
    if not value:
        return "null"
    if value.lower().startswith("bearer "):
        return f"Bearer {_mask_token(value[7:])}"
    return value[:10] + "…"
