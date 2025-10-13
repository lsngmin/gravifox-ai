"""ViT 워커 동작 테스트."""

from __future__ import annotations
from pathlib import Path
from typing import Any, List

import pytest
from PIL import Image

from api.workers import vit_worker
from api.workers.vit_worker import run_analysis


class _StubMQ:
    """테스트용 MQ 스텁."""

    def __init__(self) -> None:
        self.messages: List[tuple[str, dict[str, Any]]] = []

    async def publish_json(self, routing_key: str, payload: dict[str, Any]) -> None:
        """발행된 메시지를 기록한다."""

        self.messages.append((routing_key, payload))


class _StubService:
    """테스트용 ViT 추론 서비스."""

    async def ensure_ready(self) -> None:
        """사전 준비를 생략한다."""

    async def predict_image_with_metadata(self, image: Image.Image):
        """고정된 추론 결과와 메타데이터를 반환한다."""

        assert image.mode == "RGB"
        return [0.2, 0.8], {
            "mode": "single",
            "n_patches": 1,
            "patch_count": 1,
            "scales": [224],
            "aggregate": "mean",
        }

    async def get_pipeline(self):
        """필요 속성만 포함한 파이프라인 객체를 반환한다."""

        class _Pipeline:
            real_index = 0
            class_names = ["REAL", "FAKE"]
            model_name = "test-model"
            uncertainty_band = (0.45, 0.55)
            device = "cpu"

        return _Pipeline()


class _StubCalibrator:
    """테스트용 보정기."""

    def calibrate(self, probs: List[float], real_index: int):
        """고정된 의사결정 정보를 반환한다."""

        class _Calibration:
            distribution = list(probs)
            p_real = float(probs[real_index])
            p_ai = float(1 - probs[real_index])
            confidence = max(probs)
            decision = "ai"

        return _Calibration()


class _StubSettings:
    """테스트용 런타임 설정."""

    def __init__(self, root: Path, grace_seconds: float) -> None:
        self.file_store_root = root
        self.uncertainty_band_low = 0.45
        self.result_publish_grace_seconds = grace_seconds


@pytest.mark.asyncio
async def test_run_analysis_waits_for_grace_period(monkeypatch, tmp_path) -> None:
    """결과 발행 후 설정된 유예 시간만큼 대기하는지 확인한다."""

    upload_id = "sample.png"
    image_path = tmp_path / upload_id
    Image.new("RGB", (2, 2), color=(200, 200, 200)).save(image_path)

    mq = _StubMQ()
    service = _StubService()
    calibrator = _StubCalibrator()
    settings = _StubSettings(tmp_path, 1.5)

    sleep_calls: List[float] = []

    async def _fake_sleep(delay: float) -> None:
        sleep_calls.append(delay)

    monkeypatch.setattr(vit_worker.asyncio, "sleep", _fake_sleep)

    await run_analysis(
        mq,
        job_id="job-123",
        upload_id=upload_id,
        params=None,
        settings=settings,
        vit_service=service,
        calibrator=calibrator,
        model={"name": "stub"},
    )

    assert any(rk.startswith("analyze.result.") for rk, _ in mq.messages)
    assert sleep_calls == [pytest.approx(1.5)]
