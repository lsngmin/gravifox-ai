"""이미지 라우터 테스트."""

from __future__ import annotations

import io
from typing import Any, Iterable, List, Optional

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from api import app as fastapi_app
from api.dependencies.inference import (
    get_calibrator,
    get_model_registry,
    get_runtime_settings,
    get_storage_service,
    get_upload_token_registry,
    get_upload_token_verifier,
    get_vit_service,
)


class _FakeVitService:
    """테스트용 가짜 ViT 서비스."""

    def __init__(self) -> None:
        self.class_names = ["REAL", "FAKE"]

    async def predict_image(self, image: Image.Image) -> List[float]:
        """고정된 확률 값을 반환한다."""

        assert image.mode == "RGB"
        return [0.7, 0.3]

    async def predict_image_with_metadata(
        self, image: Image.Image
    ) -> tuple[List[float], dict[str, Any]]:
        """메타데이터를 포함한 추론 결과를 반환한다."""

        probs = await self.predict_image(image)
        meta = {
            "mode": "single",
            "n_patches": 1,
            "patch_count": 1,
            "scales": [224],
            "aggregate": "mean",
        }
        return probs, meta

    async def get_pipeline(self):
        """테스트용 파이프라인 정보를 반환한다."""

        class _Pipeline:
            real_index = 0
            class_names = ["REAL", "FAKE"]
            model_name = "dummy-vit"
            inference_mode = "single"
            inference_n_patches = 1
            inference_scales = (224,)
            inference_aggregate = "mean"
            uncertainty_band = (0.45, 0.55)
            device = "cpu"

        return _Pipeline()


class _FakeStorage:
    """테스트용 가짜 저장소 서비스."""

    def infer_media_kind(self, filename: str, content_type: Optional[str]):
        return "image"

    async def save_upload(self, file, kind, upload_id=None):
        await file.read()
        return upload_id or "test-upload-id"


class _FakeRegistry:
    """테스트용 가짜 모델 레지스트리."""

    def list_models(self) -> List[Any]:
        return [
            type(
                "Model",
                (),
                {
                    "key": "vit",
                    "name": "ViT",
                    "version": "1.0",
                    "description": "demo",
                    "type": "torch_image",
                    "input": "image",
                    "threshold": 0.5,
                    "labels": ("REAL", "FAKE"),
                },
            )
        ]

    def get_default_model(self):
        return self.list_models()[0]


class _FakeCalibrator:
    """테스트용 보정기."""

    def calibrate(self, probs: Iterable[float], real_index: int):
        """입력 확률을 그대로 사용하면서 의사결정을 계산한다."""

        values = list(probs)
        real_idx = real_index if real_index < len(values) else 0
        p_real = values[real_idx]
        return type(
            "Calibration",
            (),
            {
                "distribution": values,
                "p_real": p_real,
                "p_ai": 1 - p_real,
                "confidence": max(values),
                "decision": "real" if p_real >= 0.5 else "ai",
            },
        )()


class _FakeSettings:
    """테스트용 설정."""

    cors_allow_origins = ["*"]
    upload_token: Optional[str] = "test-token"
    upload_jwks_url: Optional[str] = None
    upload_jwks_cache_seconds: int = 300
    upload_token_api_base: Optional[str] = None
    upload_token_service_key: Optional[str] = None
    file_store_root = ""
    max_image_mb = 5
    max_video_mb = 50
    file_ttl_hours = 1
    enable_mq = False
    rabbitmq_url = ""
    vit_max_batch_size = 4
    vit_max_batch_wait_ms = 5
    vit_enable_multipatch = True
    calibration_temperature = 1.0
    uncertainty_band_low = 0.45
    uncertainty_band_high = 0.55
    result_publish_grace_seconds = 0.0


class _FakeUploadTokenVerifier:
    async def verify(self, token):
        raise AssertionError("verify should not be called when legacy token is configured")


class _FakeUploadTokenRegistry:
    async def authorize(self, token):
        return None

    async def complete_success(self, context):
        return None

    async def complete_failure(self, context, reason=None):
        return None


@pytest.fixture(autouse=True)
def _override_dependencies():
    """FastAPI 의존성을 가짜 구현으로 오버라이드한다."""

    fastapi_app.dependency_overrides[get_vit_service] = lambda: _FakeVitService()
    fastapi_app.dependency_overrides[get_storage_service] = lambda: _FakeStorage()
    fastapi_app.dependency_overrides[get_model_registry] = lambda: _FakeRegistry()
    fastapi_app.dependency_overrides[get_runtime_settings] = lambda: _FakeSettings()
    fastapi_app.dependency_overrides[get_calibrator] = lambda: _FakeCalibrator()
    fastapi_app.dependency_overrides[
        get_upload_token_verifier
    ] = lambda: _FakeUploadTokenVerifier()
    fastapi_app.dependency_overrides[
        get_upload_token_registry
    ] = lambda: _FakeUploadTokenRegistry()
    yield
    fastapi_app.dependency_overrides.clear()


def _make_test_image() -> bytes:
    """작은 PNG 이미지를 생성한다."""

    image = Image.new("RGB", (4, 4), color=(128, 128, 128))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def test_predict_image_success() -> None:
    """이미지 추론 응답이 올바른 구조인지 확인한다."""

    client = TestClient(fastapi_app)
    response = client.post(
        "/predict/image", files={"file": ("test.png", _make_test_image(), "image/png")}
    )
    assert response.status_code == 200
    body = response.json()
    assert body["model_version"] == "dummy-vit"
    assert pytest.approx(body["p_real"], rel=1e-3) == 0.7
    assert body["decision"] == "real"
    assert body["class_names"] == ["REAL", "FAKE"]


def test_upload_media() -> None:
    """업로드 엔드포인트가 식별자를 반환하는지 확인한다."""

    client = TestClient(fastapi_app)
    response = client.post(
        "/upload",
        files={"file": ("test.png", _make_test_image(), "image/png")},
        headers={"Upload-Token": "test-token"},
    )
    assert response.status_code == 200
    assert response.json() == {"uploadId": "test-upload-id"}


def test_list_models() -> None:
    """모델 목록 엔드포인트가 더미 데이터를 반환하는지 확인한다."""

    client = TestClient(fastapi_app)
    response = client.get("/models")
    assert response.status_code == 200
    body = response.json()
    assert body["defaultKey"] == "vit"
    assert body["items"][0]["name"] == "ViT"
