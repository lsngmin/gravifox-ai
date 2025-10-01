from __future__ import annotations

from typing import Callable, Dict

from core.utils.logger import get_logger


logger = get_logger(__name__)
_REGISTRY: Dict[str, Callable[..., object]] = {}


def register(name: str):
    """모델 빌더를 레지스트리에 등록하는 데코레이터."""

    def deco(fn: Callable[..., object]) -> Callable[..., object]:
        key = name.strip().lower()
        if key in _REGISTRY:
            raise RuntimeError(f"model already registered: {name}")
        _REGISTRY[key] = fn
        logger.info("모델 등록 완료 - %s", key)
        return fn

    return deco


def get_model(name: str) -> Callable[..., object]:
    """등록된 모델 빌더를 조회한다."""

    key = name.strip().lower()
    fn = _REGISTRY.get(key)
    if fn is None:
        logger.error("모델을 찾을 수 없습니다: %s", key)  # TODO: 필요 없을 시 삭제 가능
        raise KeyError(f"unknown model: {name}; available={list(_REGISTRY.keys())}")
    logger.info("모델 조회 - %s", key)
    return fn


# Import side-effects: populate registry
from .vit_b16 import build_vit_b16  # noqa: E402,F401
from .residual import build_vit_residual  # noqa: E402,F401
from .multipatch import build_vit_multipatch  # noqa: E402,F401
