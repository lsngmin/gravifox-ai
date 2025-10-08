"""모델 레지스트리."""

from __future__ import annotations

from typing import Callable, Dict, List

from core.utils.logger import get_logger

logger = get_logger(__name__)

MODEL_REGISTRY: Dict[str, Callable[..., object]] = {}


def register(name: str) -> Callable[[Callable[..., object]], Callable[..., object]]:
    """모델 빌더를 등록하는 데코레이터."""

    def decorator(builder: Callable[..., object]) -> Callable[..., object]:
        key = name.strip().lower()
        if key in MODEL_REGISTRY:
            raise RuntimeError(f"model already registered: {name}")
        MODEL_REGISTRY[key] = builder
        logger.debug("모델 레지스트리에 등록: %s", key)
        return builder

    return decorator


def available_models() -> List[str]:
    return sorted(MODEL_REGISTRY.keys())


def get_builder(name: str) -> Callable[..., object]:
    key = name.strip().lower()
    builder = MODEL_REGISTRY.get(key)
    if builder is None:
        raise KeyError(f"unknown model: {name}; available={available_models()}")
    return builder


def get_model(name: str, **kwargs) -> object:
    """등록된 빌더를 호출해 모델 인스턴스를 생성한다."""

    builder = get_builder(name)
    model = builder(**kwargs)
    logger.info("모델 생성 - name=%s args=%s", name, kwargs)
    return model


# 사이드이펙트 임포트: 데코레이터를 통해 등록 수행
from . import vit_b16  # noqa: E402,F401
from . import residual  # noqa: E402,F401
from . import multipatch  # noqa: E402,F401
from . import vit_residual  # noqa: E402,F401
from . import fusion_head  # noqa: E402,F401
