"""FastAPI 런타임 환경 설정을 정의한다.

목적:
    환경 변수 및 .env 값을 통해 API 서버가 참조할 설정 값을 관리한다.
    모든 설정은 pydantic `BaseSettings` 기반으로 선언하며, 모듈 전체에서
    재사용할 수 있도록 싱글톤 형태의 접근자를 함께 제공한다.

사용법:
    >>> from api.config.runtime import get_settings
    >>> settings = get_settings()
    >>> settings.file_store_root
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import List, Optional

# Pydantic v2 지원: pydantic-settings 사용. v1 환경에서는 이전 import로 폴백.
try:  # Pydantic v2
    from pydantic_settings import BaseSettings, SettingsConfigDict  # type: ignore
    from pydantic import Field, field_validator  # type: ignore
    _IS_PYDANTIC_V2 = True
except Exception:  # Fallbacks for environments without pydantic-settings
    try:
        # Pydantic v2 installed but no pydantic-settings: use v1-compat namespace
        from pydantic.v1 import BaseSettings, Field, validator  # type: ignore

        SettingsConfigDict = None  # type: ignore
        field_validator = None  # type: ignore
        _IS_PYDANTIC_V2 = False
    except Exception:
        # True Pydantic v1 environment
        from pydantic import BaseSettings, Field, validator  # type: ignore

        SettingsConfigDict = None  # type: ignore
        field_validator = None  # type: ignore
        _IS_PYDANTIC_V2 = False


_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_RUN_ROOT = _REPO_ROOT / "experiments" / "vit_residual_fusion"
_DEFAULT_MODEL_CATALOG = _REPO_ROOT / "api" / "models" / "catalog.json"


class RuntimeSettings(BaseSettings):
    """API 서버 전역 설정 값을 보관한다.

    목적:
        FastAPI 애플리케이션과 워커, 서비스 계층에서 참조하는 환경 값을
        중앙에서 관리한다. 기본값은 보수적으로 정의하고, 환경 변수를 통해
        손쉽게 덮어쓸 수 있도록 구성한다.

    Attributes:
        cors_allow_origins: CORS 허용 origin 목록.
        upload_token: 업로드 인증 토큰.
        file_store_root: 업로드 파일이 저장되는 루트 경로.
        max_image_mb: 이미지 업로드 최대 크기(MB).
        max_video_mb: 동영상 업로드 최대 크기(MB).
        file_ttl_hours: 업로드 파일 보존 시간(시간 단위).
        vit_run_dir: 특정 실험 경로를 직접 지정할 때 사용하는 절대 경로.
        vit_run_root: 실험 디렉터리의 기본 루트 경로.
        vit_checkpoint_name: 사용할 체크포인트 파일명.
        vit_device_name: 추론 시 사용할 디바이스 문자열.
        model_catalog_path: 모델 카탈로그 파일 경로.
        enable_mq: MQ 기능 사용 여부.
        tvb_max_concurrency: MQ 처리 동시성 제한 값.
        rabbitmq_url: RabbitMQ 연결 URL.
        rabbitmq_use_tls: TLS 강제 사용 여부.
        rabbitmq_verify_peer: TLS 사용 시 peer 검증 여부.
        rabbitmq_ca_file: TLS 검증에 사용할 CA 인증서 경로.
        rabbitmq_cert_file: 클라이언트 인증서 경로.
        rabbitmq_key_file: 클라이언트 키 파일 경로.
        analyze_exchange: 분석 요청/결과 교환기 이름.
        request_queue: 분석 요청 큐 이름.
        rabbitmq_prefetch: MQ prefetch 개수.
    """

    cors_allow_origins: List[str] = Field(
        default_factory=lambda: ["*"], env="CORS_ALLOW_ORIGINS"
    )
    upload_token: Optional[str] = Field(default=None, env="UPLOAD_TOKEN")
    file_store_root: Path = Field(default=Path("/tmp/uploads"), env="FILE_STORE_ROOT")
    max_image_mb: float = Field(default=5.0, env="MAX_IMAGE_MB")
    max_video_mb: float = Field(default=50.0, env="MAX_VIDEO_MB")
    file_ttl_hours: float = Field(default=24.0, env="FILE_TTL_HOURS")

    vit_run_dir: Optional[Path] = Field(default=None, env="TVB_VIT_RUN_DIR")
    vit_run_root: Path = Field(default=_DEFAULT_RUN_ROOT, env="TVB_VIT_RUN_ROOT")
    vit_checkpoint_name: str = Field(default="best.pt", env="TVB_VIT_CHECKPOINT")
    vit_device_name: str = Field(default="auto", env="TVB_VIT_DEVICE")
    vit_max_batch_size: int = Field(default=8, env="TVB_VIT_MAX_BATCH")
    vit_max_batch_wait_ms: int = Field(default=8, env="TVB_VIT_BATCH_WAIT_MS")
    vit_enable_multipatch: bool = Field(default=True, env="TVB_VIT_ENABLE_MULTIPATCH")

    model_catalog_path: Path = Field(
        default=_DEFAULT_MODEL_CATALOG, env="MODEL_CATALOG_PATH"
    )

    enable_mq: bool = Field(default=True, env="ENABLE_MQ")
    tvb_max_concurrency: int = Field(default=1, env="TVB_MAX_CONCURRENCY")

    calibration_temperature: float = Field(default=1.2, env="CALIBRATION_TEMPERATURE")
    uncertainty_band_low: float = Field(default=0.45, env="UNCERTAINTY_BAND_LOW")
    uncertainty_band_high: float = Field(default=0.55, env="UNCERTAINTY_BAND_HIGH")

    rabbitmq_url: str = Field(default="", env="RABBITMQ_URL")
    rabbitmq_use_tls: Optional[bool] = Field(default=None, env="RABBITMQ_USE_TLS")
    rabbitmq_verify_peer: Optional[bool] = Field(
        default=True, env="RABBITMQ_VERIFY_PEER"
    )
    rabbitmq_ca_file: Optional[Path] = Field(default=None, env="RABBITMQ_CA_FILE")
    rabbitmq_cert_file: Optional[Path] = Field(default=None, env="RABBITMQ_CERT_FILE")
    rabbitmq_key_file: Optional[Path] = Field(default=None, env="RABBITMQ_KEY_FILE")

    analyze_exchange: str = Field(default="analyze.exchange", env="ANALYZE_EXCHANGE")
    request_queue: str = Field(default="analyze.request.fastapi", env="REQUEST_QUEUE")
    rabbitmq_prefetch: int = Field(default=10, env="RABBITMQ_PREFETCH")

    # Pydantic v2에서는 SettingsConfigDict, v1에서는 class Config 사용
    if _IS_PYDANTIC_V2:
        model_config = SettingsConfigDict(  # type: ignore[call-arg]
            env_file=".env", env_file_encoding="utf-8"
        )
    else:
        class Config:  # type: ignore[no-redef]
            """pydantic 설정."""

            env_file = ".env"
            env_file_encoding = "utf-8"

    if _IS_PYDANTIC_V2:
        @field_validator("cors_allow_origins", mode="before")  # type: ignore[misc]
        def _split_origins(cls, value: object) -> List[str]:
            """콤마 구분 문자열을 리스트로 변환한다."""

            if isinstance(value, str):
                return [origin.strip() for origin in value.split(",") if origin.strip()]
            if isinstance(value, (list, tuple)):
                return [str(origin).strip() for origin in value if str(origin).strip()]
            return ["*"]
    else:
        @validator("cors_allow_origins", pre=True)  # type: ignore[misc]
        def _split_origins(cls, value: object) -> List[str]:
            """콤마 구분 문자열을 리스트로 변환한다."""

            if isinstance(value, str):
                return [origin.strip() for origin in value.split(",") if origin.strip()]
            if isinstance(value, (list, tuple)):
                return [str(origin).strip() for origin in value if str(origin).strip()]
            return ["*"]

    if _IS_PYDANTIC_V2:
        @field_validator("file_store_root", "vit_run_root", mode="before")  # type: ignore[misc]
        def _expand_paths(cls, value: object) -> Path:
            """상대 경로를 절대 경로로 확장한다."""

            if isinstance(value, Path):
                return value.expanduser().resolve()
            return Path(str(value)).expanduser().resolve()
    else:
        @validator("file_store_root", "vit_run_root", pre=True)  # type: ignore[misc]
        def _expand_paths(cls, value: object) -> Path:
            """상대 경로를 절대 경로로 확장한다."""

            if isinstance(value, Path):
                return value.expanduser().resolve()
            return Path(str(value)).expanduser().resolve()

    if _IS_PYDANTIC_V2:
        @field_validator("vit_run_dir", mode="before")  # type: ignore[misc]
        def _optional_path(cls, value: object) -> Optional[Path]:
            """옵셔널 경로 값을 처리한다."""

            if value in (None, "", b""):
                return None
            return Path(str(value)).expanduser().resolve()
    else:
        @validator("vit_run_dir", pre=True)  # type: ignore[misc]
        def _optional_path(cls, value: object) -> Optional[Path]:
            """옵셔널 경로 값을 처리한다."""

            if value in (None, "", b""):
                return None
            return Path(str(value)).expanduser().resolve()

    if _IS_PYDANTIC_V2:
        @field_validator("model_catalog_path", mode="before")  # type: ignore[misc]
        def _catalog_path(cls, value: object) -> Path:
            """모델 카탈로그 경로를 절대 경로로 변환한다."""

            if isinstance(value, Path):
                return value.expanduser().resolve()
            return Path(str(value)).expanduser().resolve()
    else:
        @validator("model_catalog_path", pre=True)  # type: ignore[misc]
        def _catalog_path(cls, value: object) -> Path:
            """모델 카탈로그 경로를 절대 경로로 변환한다."""

            if isinstance(value, Path):
                return value.expanduser().resolve()
            return Path(str(value)).expanduser().resolve()


@lru_cache(maxsize=1)
def get_settings() -> RuntimeSettings:
    """싱글톤 형태로 설정 객체를 반환한다."""

    return RuntimeSettings()


__all__ = ["RuntimeSettings", "get_settings"]
