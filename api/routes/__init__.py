"""FastAPI 라우터 모음."""

from . import explain, image, metrics, video

__all__ = [
    "image",
    "video",
    "explain",
    "metrics",
]
