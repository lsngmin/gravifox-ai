"""FastAPI 엔트리 포인트."""

from __future__ import annotations

import asyncio
import time
from typing import Any, Dict

from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware

from core.utils.logger import get_logger

from api.config import RuntimeSettings
from api.dependencies.inference import (
    get_mq_service,
    get_runtime_settings,
    get_storage_service,
    get_vit_service,
)
from api.workers.vit_worker import run_analysis
from api.routes import explain, image, metrics, video

LOGGER = get_logger("api.main")

app = FastAPI(title="Gravifox TVB-AI API", version="1.0.0")
_settings = get_runtime_settings()
app.add_middleware(
    CORSMiddleware,
    allow_origins=_settings.cors_allow_origins,
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=False,
)

_consumer_task: asyncio.Task | None = None


@app.on_event("startup")
async def on_startup() -> None:
    """애플리케이션 시작 시 백그라운드 작업을 준비한다.

    Returns:
        없음.
    """

    settings = get_runtime_settings()
    storage = get_storage_service()
    asyncio.create_task(storage.run_cleanup_loop())
    vit_service = get_vit_service()
    await vit_service.startup()
    if settings.enable_mq and settings.rabbitmq_url:
        mq = get_mq_service()
        try:
            await mq.connect()
            LOGGER.info("RabbitMQ 연결이 완료되었습니다")

            async def _handle_request(payload: dict) -> None:
                job_id = str(payload.get("jobId") or "").strip()
                upload_id = str(payload.get("uploadId") or "").strip()
                params = payload.get("params")
                if not job_id or not upload_id:
                    LOGGER.warning("잘못된 analyze.request payload: %s", payload)
                    return
                try:
                    await run_analysis(mq, job_id, upload_id, params)
                except Exception:  # pragma: no cover - 워커 예외 로깅
                    LOGGER.exception("요청 처리 중 예외 발생 jobId=%s uploadId=%s", job_id, upload_id)

            global _consumer_task
            _consumer_task = asyncio.create_task(mq.consume_requests(_handle_request))
            LOGGER.info("요청 큐 소비를 시작했습니다")
        except Exception as exc:  # pragma: no cover - 외부 서비스 의존
            LOGGER.error("MQ 초기화 실패: %s", exc)


@app.on_event("shutdown")
async def on_shutdown() -> None:
    """애플리케이션 종료 시 자원을 정리한다.

    Returns:
        없음.
    """

    # Stop the consumer loop first
    global _consumer_task
    if _consumer_task is not None:
        try:
            _consumer_task.cancel()
        except Exception:
            pass
        _consumer_task = None

    mq = get_mq_service()
    try:
        await mq.close()
    except Exception:  # pragma: no cover - best effort
        LOGGER.warning("MQ 연결 종료 중 오류가 발생했습니다", exc_info=True)
    vit_service = get_vit_service()
    await vit_service.shutdown()


app.include_router(image.router)
app.include_router(video.router)
app.include_router(explain.router)
app.include_router(metrics.router)


@app.get("/")
async def index(
    settings: RuntimeSettings = Depends(get_runtime_settings),
) -> Dict[str, Any]:
    """헬스 체크용 기본 엔드포인트.

    Args:
        settings: 런타임 설정 의존성.

    Returns:
        서비스 상태 정보를 담은 딕셔너리.
    """

    return {
        "message": "tvb-ai api is running",
        "timestamp": time.time(),
        "model_ready": bool(get_vit_service()),
        "mq_enabled": settings.enable_mq and bool(settings.rabbitmq_url),
    }


__all__ = ["app"]
