"""비디오 관련 FastAPI 라우터."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

router = APIRouter()


@router.post("/predict/video")
async def predict_video() -> dict:
    """동영상 위조 판별은 현재 미구현 상태임을 알린다.

    Returns:
        없음. 501 예외를 발생시킨다.
    """

    raise HTTPException(status_code=501, detail="video inference not implemented yet")


__all__ = ["router"]
