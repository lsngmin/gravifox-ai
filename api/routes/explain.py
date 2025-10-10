"""설명(Explainability) 관련 라우터."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

router = APIRouter()


@router.post("/explain/heatmap")
async def generate_heatmap() -> dict:
    """설명용 히트맵 생성을 아직 지원하지 않음을 알린다.

    Returns:
        없음. 501 예외를 발생시킨다.
    """

    raise HTTPException(
        status_code=501, detail="heatmap explainability not implemented yet"
    )


__all__ = ["router"]
