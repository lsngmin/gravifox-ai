"""추론 확률 보정 및 임계값 결정을 담당하는 모듈."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List

import torch

from core.utils.logger import get_logger

from api.config import RuntimeSettings


@dataclass(slots=True)
class CalibrationResult:
    """보정 결과를 표현하는 데이터 클래스."""

    distribution: List[float]
    p_real: float
    p_ai: float
    confidence: float
    decision: str


class AdaptiveThresholdCalibrator:
    """품질 추정 기반으로 임계값을 동적으로 조정하는 보정기."""

    def __init__(self, settings: RuntimeSettings) -> None:
        """보정기를 초기화한다.

        Args:
            settings: 런타임 설정 객체.
        """

        self._temperature = max(float(settings.calibration_temperature), 1e-3)
        self._uncertainty_low = float(settings.uncertainty_band_low)
        self._uncertainty_high = float(settings.uncertainty_band_high)
        if self._uncertainty_low > self._uncertainty_high:
            raise ValueError("불확실성 구간 하한이 상한보다 큽니다.")
        self._logger = get_logger("api.services.calibration")

    def calibrate(self, probs: Iterable[float], real_index: int) -> CalibrationResult:
        """입력 확률 분포를 보정하고 임계값 결정을 수행한다.

        Args:
            probs: 모델에서 반환한 소프트맥스 확률 분포.
            real_index: 실제(REAL) 클래스 인덱스.

        Returns:
            보정된 확률 분포와 의사결정이 담긴 결과.
        """

        tensor = torch.tensor(list(probs), dtype=torch.float32)
        if tensor.numel() == 0:
            raise ValueError("확률 분포가 비어 있습니다.")
        safe = tensor.clamp(min=1e-8)
        scaled = torch.softmax(torch.log(safe) / self._temperature, dim=0)
        distribution = scaled.tolist()
        real_idx = max(0, min(int(real_index), len(distribution) - 1))
        p_real = float(distribution[real_idx])
        p_ai = float(max(0.0, 1.0 - p_real))
        confidence = float(max(distribution))
        decision = self._decide_label(p_real)
        self._logger.debug(
            "calibrate 완료 - p_real=%.4f confidence=%.4f decision=%s",
            p_real,
            confidence,
            decision,
        )
        return CalibrationResult(
            distribution=[float(x) for x in distribution],
            p_real=p_real,
            p_ai=p_ai,
            confidence=confidence,
            decision=decision,
        )

    def _decide_label(self, p_real: float) -> str:
        """불확실성 구간을 고려해 최종 라벨을 결정한다."""

        if p_real < self._uncertainty_low:
            return "ai"
        if p_real > self._uncertainty_high:
            return "real"
        return "retry"


__all__ = ["AdaptiveThresholdCalibrator", "CalibrationResult"]
