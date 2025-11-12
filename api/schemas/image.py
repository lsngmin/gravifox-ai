"""이미지 관련 API 스키마."""

from __future__ import annotations

from typing import List

from pydantic import BaseModel, ConfigDict, Field


class ApiSchema(BaseModel):
    """FastAPI 스키마 공통 베이스."""

    model_config = ConfigDict(protected_namespaces=())


class ImagePredictionResponse(ApiSchema):
    """이미지 추론 응답."""

    timestamp: float = Field(..., description="예측이 생성된 UNIX 타임스탬프")
    model_version: str = Field(..., description="사용된 모델 버전")
    latency_ms: float = Field(..., description="예측 소요 시간(ms)")
    p_real: float = Field(..., description="실제(Real) 클래스 확률")
    p_ai: float = Field(..., description="생성(FAKE) 클래스 확률")
    confidence: float = Field(..., description="신뢰도 점수")
    decision: str = Field(..., description="임계값 기반 최종 의사결정")
    class_names: List[str] = Field(..., description="모델 클래스 이름 목록")
    probabilities: List[float] = Field(..., description="클래스별 확률 분포")


class UploadResponse(ApiSchema):
    """업로드 식별자를 담는 응답."""

    uploadId: str = Field(..., description="저장된 업로드 식별자")


class ModelItem(ApiSchema):
    """모델 카탈로그 항목."""

    key: str
    name: str
    version: str | None = None
    description: str | None = None
    type: str
    input: str
    threshold: float
    labels: List[str]


class ModelListResponse(ApiSchema):
    """모델 목록 응답."""

    defaultKey: str
    items: List[ModelItem]


__all__ = [
    "ImagePredictionResponse",
    "UploadResponse",
    "ModelItem",
    "ModelListResponse",
]
