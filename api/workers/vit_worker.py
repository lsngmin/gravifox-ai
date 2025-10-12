"""Vit 기반 비동기 워커."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, Optional

from PIL import Image

from core.utils.logger import get_logger

from api.config import get_settings
from api.services.calibration import AdaptiveThresholdCalibrator
from api.services.inference import VitInferenceService
from api.services.mq import MQService, publish_failed, publish_progress, publish_result

LOGGER = get_logger("api.workers.vit_worker")


async def run_analysis(
    mq: MQService,
    job_id: str,
    upload_id: str,
    params: Optional[Dict[str, Any]],
    *,
    settings: Any | None = None,
    vit_service: VitInferenceService | None = None,
    calibrator: AdaptiveThresholdCalibrator | None = None,
    model: Optional[Dict[str, Any]] = None,
) -> None:
    """업로드된 이미지를 분석하고 결과를 MQ로 전송한다.

    Args:
        mq: MQ 서비스 인스턴스.
        job_id: 분석 작업 식별자.
        upload_id: 업로드 파일 식별자.
        params: 추가 파라미터(미사용).

    Returns:
        없음.
    """

    settings_obj = settings or getattr(vit_service, "_settings", None) or get_settings()
    service = vit_service or VitInferenceService(settings_obj)
    owns_service = vit_service is None
    calibrator_obj = calibrator or AdaptiveThresholdCalibrator(settings_obj)

    await service.ensure_ready()
    media_path = Path(settings_obj.file_store_root) / upload_id
    if not media_path.is_file():
        await publish_failed(
            mq, job_id, f"upload not found: {upload_id}", reason_code="FILE_NOT_FOUND"
        )
        return
    start = time.perf_counter()
    try:
        with Image.open(media_path) as image:
            probs, inference_meta = await service.predict_image_with_metadata(image)
        pipeline = await service.get_pipeline()

        calibration = calibrator_obj.calibrate(probs, pipeline.real_index)
        latency_ms = round((time.perf_counter() - start) * 1000.0, 2)

        await publish_progress(
            mq,
            job_id,
            {
                "stage": "inference",
                "latencyMs": latency_ms,
            },
        )

        decision = calibration.decision.lower()
        label_map = {"real": "REAL", "ai": "FAKE", "retry": "RETRY"}
        label = label_map.get(decision, decision.upper())
        threshold_fake = float(1.0 - settings_obj.uncertainty_band_low)

        result_payload = {
            "modelVersion": pipeline.model_name,
            "classNames": pipeline.class_names,
            "probabilities": [float(p) for p in calibration.distribution],
            "pReal": float(calibration.p_real),
            "pAi": float(calibration.p_ai),
            "confidence": float(calibration.confidence),
            "decision": calibration.decision,
            "label": label,
            "prob_fake": float(calibration.p_ai),
            "prob_real": float(calibration.p_real),
            "threshold": threshold_fake,
            "uncertainty_band": list(pipeline.uncertainty_band),
            "inference": {
                "mode": inference_meta.get("mode"),
                "n_patches": inference_meta.get("n_patches"),
                "patch_count": inference_meta.get("patch_count"),
                "scales": inference_meta.get("scales"),
                "aggregate": inference_meta.get("aggregate"),
                "latencyMs": latency_ms,
                "device": str(pipeline.device),
            },
            "params": params or {},
            "model": model or {},
        }

        LOGGER.info(
            "워커 추론 완료 - jobId=%s label=%s pAi=%.4f latencyMs=%.2f",
            job_id,
            label,
            float(calibration.p_ai),
            latency_ms,
        )
        LOGGER.info(
            "결과 이벤트 발행 준비 - jobId=%s routing_key=analyze.result.%s",
            job_id,
            job_id,
        )
        await publish_result(mq, job_id, result_payload)
        LOGGER.info("결과 이벤트 발행 완료 - jobId=%s", job_id)
    except Exception as exc:  # pragma: no cover - 예외 상황 로깅
        LOGGER.exception("워커 분석 중 오류 발생")
        await publish_failed(
            mq,
            job_id,
            f"analysis error: {exc}",
            reason_code="WORKER_EXCEPTION",
        )
    finally:
        if owns_service:
            await service.shutdown()


__all__ = ["run_analysis"]
