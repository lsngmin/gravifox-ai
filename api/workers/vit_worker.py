"""Vit 기반 비동기 워커."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, Optional

from PIL import Image

from core.utils.logger import get_logger

from api.config import get_settings
from api.services.inference import VitInferenceService
from api.services.mq import MQService, publish_failed, publish_progress, publish_result

LOGGER = get_logger("api.workers.vit_worker")


async def run_analysis(
    mq: MQService, job_id: str, upload_id: str, params: Optional[Dict[str, Any]]
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

    settings = get_settings()
    vit_service = VitInferenceService(settings)
    media_path = Path(settings.file_store_root) / upload_id
    if not media_path.is_file():
        await publish_failed(
            mq, job_id, f"upload not found: {upload_id}", reason_code="FILE_NOT_FOUND"
        )
        return
    start = time.perf_counter()
    try:
        with Image.open(media_path) as image:
            probs = await vit_service.predict_image(image)
        pipeline = await vit_service.get_pipeline()
        real_index = pipeline.real_index if pipeline.real_index < len(probs) else 0
        p_real = float(probs[real_index]) if probs else 0.0
        p_ai = float(1.0 - p_real)
        await publish_progress(
            mq,
            job_id,
            {
                "stage": "inference",
                "latencyMs": round((time.perf_counter() - start) * 1000.0, 2),
            },
        )
        await publish_result(
            mq,
            job_id,
            {
                "modelVersion": pipeline.model_name,
                "probabilities": probs,
                "classNames": pipeline.class_names,
                "pReal": p_real,
                "pAi": p_ai,
            },
        )
    except Exception as exc:  # pragma: no cover - 예외 상황 로깅
        LOGGER.exception("워커 분석 중 오류 발생")
        await publish_failed(
            mq,
            job_id,
            f"analysis error: {exc}",
            reason_code="WORKER_EXCEPTION",
        )


__all__ = ["run_analysis"]
