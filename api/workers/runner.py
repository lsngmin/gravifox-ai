"""MQ consume 루프를 실행하는 독립형 워커 런너."""

from __future__ import annotations

import asyncio
from typing import Any, Dict

from core.utils.logger import get_logger

from api.config import get_settings
from api.services.calibration import AdaptiveThresholdCalibrator
from api.services.inference import VitInferenceService
from api.services.mq import MQService
from api.workers.vit_worker import run_analysis

LOGGER = get_logger("api.workers.runner")


async def _handle_request(
    payload: Dict[str, Any],
    *,
    mq: MQService,
    vit: VitInferenceService,
    calibrator: AdaptiveThresholdCalibrator,
    settings,
) -> None:
    """단일 MQ 요청 메시지를 처리한다."""

    job_id = str(payload.get("jobId") or "").strip()
    upload_id = str(payload.get("uploadId") or "").strip()
    params = payload.get("params") if isinstance(payload.get("params"), dict) else {}
    model_info = payload.get("model") if isinstance(payload.get("model"), dict) else {}

    if not job_id or not upload_id:
        LOGGER.warning("무효한 분석 요청을 무시합니다: payload=%s", payload)
        return

    await run_analysis(
        mq,
        job_id,
        upload_id,
        params,
        settings=settings,
        vit_service=vit,
        calibrator=calibrator,
        model=model_info,
    )


async def main() -> None:
    """MQ 소비 루프를 실행한다."""

    settings = get_settings()
    if not settings.enable_mq or not settings.rabbitmq_url:
        LOGGER.error(
            "MQ가 비활성화되어 워커를 시작할 수 없습니다. ENABLE_MQ / RABBITMQ_URL을 확인하세요."
        )
        return

    mq = MQService(settings)
    await mq.connect()
    vit_service = VitInferenceService(settings)
    calibrator = AdaptiveThresholdCalibrator(settings)
    await vit_service.ensure_ready()
    pipeline = await vit_service.get_pipeline()
    LOGGER.info(
        "분석 워커 준비 완료 - queue=%s device=%s model=%s",
        settings.request_queue,
        pipeline.device,
        pipeline.model_name,
    )

    async def _consumer(payload: Dict[str, Any]) -> None:
        try:
            await _handle_request(
                payload or {},
                mq=mq,
                vit=vit_service,
                calibrator=calibrator,
                settings=settings,
            )
        except Exception:  # pragma: no cover - 방어적 로깅
            LOGGER.exception("MQ 요청 처리 중 예외가 발생했습니다")

    try:
        await mq.consume_requests(_consumer)
    finally:
        await vit_service.shutdown()
        await mq.close()


if __name__ == "__main__":  # pragma: no cover - 실행 엔트리 포인트
    asyncio.run(main())
