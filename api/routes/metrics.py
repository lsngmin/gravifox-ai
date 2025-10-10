"""메트릭 수집 라우터."""

from __future__ import annotations

from fastapi import APIRouter, status

router = APIRouter()


@router.post("/metrics/ingest", status_code=status.HTTP_202_ACCEPTED)
async def ingest_metrics() -> dict:
    """수집된 메트릭을 임시로 수락만 수행한다.

    Returns:
        단순 확인 응답.
    """

    return {"status": "accepted"}


__all__ = ["router"]
