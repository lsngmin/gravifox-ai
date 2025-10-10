"""FastAPI 의존성 주입 모듈."""

from __future__ import annotations

from functools import lru_cache

from api.config import RuntimeSettings, get_settings
from api.services.calibration import AdaptiveThresholdCalibrator
from api.services.inference import VitInferenceService
from api.services.mq import MQService
from api.services.registry import ModelRegistryService
from api.services.storage import MediaStorageService


@lru_cache(maxsize=1)
def _settings_singleton() -> RuntimeSettings:
    """전역 설정을 싱글톤으로 제공한다.

    Returns:
        런타임 설정 인스턴스.
    """

    return get_settings()


def get_runtime_settings() -> RuntimeSettings:
    """FastAPI용 런타임 설정 의존성.

    Returns:
        런타임 설정 인스턴스.
    """

    return _settings_singleton()


@lru_cache(maxsize=1)
def _vit_service_singleton() -> VitInferenceService:
    """ViT 추론 서비스를 싱글톤으로 생성한다.

    Returns:
        ViT 추론 서비스 인스턴스.
    """

    return VitInferenceService(get_runtime_settings())


def get_vit_service() -> VitInferenceService:
    """FastAPI 라우트에서 사용할 ViT 서비스 의존성.

    Returns:
        ViT 추론 서비스 인스턴스.
    """

    return _vit_service_singleton()


@lru_cache(maxsize=1)
def _calibrator_singleton() -> AdaptiveThresholdCalibrator:
    """추론 결과 보정기를 싱글톤으로 생성한다.

    Returns:
        보정기 인스턴스.
    """

    return AdaptiveThresholdCalibrator(get_runtime_settings())


def get_calibrator() -> AdaptiveThresholdCalibrator:
    """FastAPI 라우트용 보정기 의존성.

    Returns:
        추론 결과 보정기 인스턴스.
    """

    return _calibrator_singleton()


@lru_cache(maxsize=1)
def _storage_service_singleton() -> MediaStorageService:
    """미디어 저장 서비스 싱글톤.

    Returns:
        미디어 저장 서비스 인스턴스.
    """

    return MediaStorageService(get_runtime_settings())


def get_storage_service() -> MediaStorageService:
    """FastAPI 라우트용 저장소 서비스 의존성.

    Returns:
        미디어 저장 서비스 인스턴스.
    """

    return _storage_service_singleton()


@lru_cache(maxsize=1)
def _registry_service_singleton() -> ModelRegistryService:
    """모델 카탈로그 서비스 싱글톤.

    Returns:
        모델 카탈로그 서비스 인스턴스.
    """

    return ModelRegistryService(get_runtime_settings())


def get_model_registry() -> ModelRegistryService:
    """FastAPI 라우트용 모델 카탈로그 의존성.

    Returns:
        모델 카탈로그 서비스 인스턴스.
    """

    return _registry_service_singleton()


@lru_cache(maxsize=1)
def _mq_service_singleton() -> MQService:
    """MQ 서비스 싱글톤.

    Returns:
        MQ 서비스 인스턴스.
    """

    return MQService(get_runtime_settings())


def get_mq_service() -> MQService:
    """FastAPI 라우트용 MQ 서비스 의존성.

    Returns:
        MQ 서비스 인스턴스.
    """

    return _mq_service_singleton()


__all__ = [
    "get_runtime_settings",
    "get_vit_service",
    "get_storage_service",
    "get_model_registry",
    "get_mq_service",
    "get_calibrator",
]
