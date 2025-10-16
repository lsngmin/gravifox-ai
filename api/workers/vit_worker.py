"""Vit 기반 비동기 워커."""

from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

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

        heatmap_meta = (
            inference_meta.get("heatmap")
            if isinstance(inference_meta, dict)
            else None
        )
        heatmap_score = _resolve_heatmap_score(heatmap_meta)

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
                "cell_sizes": inference_meta.get("cell_sizes"),
                "aggregate": inference_meta.get("aggregate"),
                "latencyMs": latency_ms,
                "device": str(pipeline.device),
                "grid": inference_meta.get("grid"),
                "heatmap": heatmap_meta,
                "patches": inference_meta.get("patches"),
            },
            "params": params or {},
            "model": model or {},
        }
        if heatmap_score is not None:
            result_payload["heatmap_score"] = heatmap_score

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

        grace_seconds = max(0.0, float(settings_obj.result_publish_grace_seconds))
        if grace_seconds > 0:
            LOGGER.debug(
                "결과 발행 후 SSE 플러시 유예 시간 대기 - jobId=%s delay=%.2fs",
                job_id,
                grace_seconds,
            )
            try:
                await asyncio.sleep(grace_seconds)
            except asyncio.CancelledError:
                LOGGER.warning(
                    "결과 발행 후 유예 대기 중 취소 감지 - jobId=%s delay=%.2fs",
                    job_id,
                    grace_seconds,
                )
                raise
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


def _resolve_heatmap_score(heatmap: Optional[Dict[str, Any]]) -> Optional[float]:
    """히트맵 셀에서 최대 AI 확률을 추출한다."""

    if not heatmap or not isinstance(heatmap, dict):
        return None
    cells: Sequence[Dict[str, Any]] | None = heatmap.get("cells")
    if not isinstance(cells, Sequence):
        return None
    best = None
    for cell in cells:
        if not isinstance(cell, dict):
            continue
        ai_max = cell.get("ai_max")
        ai_mean = cell.get("ai_mean")
        for candidate in (ai_max, ai_mean):
            if isinstance(candidate, (int, float)):
                value = float(candidate)
                if best is None or value > best:
                    best = value
    if best is None:
        return None
    if best < 0.0:
        return 0.0
    if best > 1.0:
        return 1.0
    return best


__all__ = ["run_analysis"]
