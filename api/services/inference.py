"""추론 서비스 계층을 정의한다.

목적:
    ViT 기반 이미지 위조 판별 파이프라인을 초기화하고 FastAPI 및 워커에서
    재사용할 수 있도록 비동기 친화적인 서비스 클래스를 제공한다.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional

import torch
import yaml
from PIL import Image

from core.datasets.base import load_dataset_config
from core.datasets.transforms import build_val_transforms
from core.models.registry import get_model
from core.utils.logger import get_logger

from api.config import RuntimeSettings


@dataclass
class VitPipeline:
    """ViT 추론에 필요한 리소스를 보관한다."""

    model: torch.nn.Module
    transform: Any
    device: torch.device
    class_names: List[str]
    real_index: int
    run_dir: Path
    model_name: str


class VitInferenceService:
    """ViT 이미지 추론을 관리하는 서비스."""

    def __init__(self, settings: RuntimeSettings) -> None:
        """서비스를 초기화한다.

        Args:
            settings: 런타임 환경 설정.
        """

        self._settings = settings
        self._logger = get_logger("api.services.inference")
        self._pipeline: Optional[VitPipeline] = None
        self._init_error: Optional[Exception] = None
        self._init_lock = asyncio.Lock()
        self._predict_lock = asyncio.Lock()

    async def predict_image(self, image: Image.Image) -> List[float]:
        """이미지에 대한 위조 확률 분포를 반환한다.

        Args:
            image: PIL 이미지 객체.

        Returns:
            클래스별 확률 분포 리스트.
        """

        pipeline = await self._ensure_pipeline()
        tensor = (
            pipeline.transform(image.convert("RGB")).unsqueeze(0).to(pipeline.device)
        )
        async with self._predict_lock:
            with torch.inference_mode():
                logits = pipeline.model(tensor)
                probs = torch.softmax(logits, dim=1)
        return probs.squeeze(0).detach().cpu().tolist()

    async def ensure_ready(self) -> None:
        """파이프라인 초기화가 완료되도록 보장한다.

        Returns:
            없음.
        """

        await self._ensure_pipeline()

    async def get_pipeline(self) -> VitPipeline:
        """현재 파이프라인 정보를 반환한다.

        Returns:
            초기화된 파이프라인 객체.
        """

        return await self._ensure_pipeline()

    async def _ensure_pipeline(self) -> VitPipeline:
        """내부 파이프라인을 초기화하거나 반환한다.

        Returns:
            초기화된 파이프라인 객체.
        """

        if self._pipeline is not None:
            return self._pipeline
        if self._init_error is not None:
            raise RuntimeError(
                f"모델 초기화 실패: {self._init_error}"
            ) from self._init_error
        async with self._init_lock:
            if self._pipeline is not None:
                return self._pipeline
            if self._init_error is not None:
                raise RuntimeError(
                    f"모델 초기화 실패: {self._init_error}"
                ) from self._init_error
            try:
                self._pipeline = self._initialize_pipeline()
                self._logger.info(
                    "VIT 추론 파이프라인 초기화 완료 - run=%s checkpoint=%s device=%s",
                    self._pipeline.run_dir,
                    self._settings.vit_checkpoint_name,
                    self._pipeline.device,
                )
            except Exception as exc:  # pragma: no cover - 초기화 실패 시 로그
                self._init_error = exc
                self._logger.exception("VIT 추론 파이프라인 초기화 실패")
                raise
        return self._pipeline

    def _initialize_pipeline(self) -> VitPipeline:
        """파이프라인 구성 요소를 동기적으로 초기화한다.

        Returns:
            초기화된 파이프라인 데이터.
        """

        device = self._resolve_device()
        run_dir = self._resolve_run_dir()
        meta = self._load_meta(run_dir)
        transform = self._build_transform(meta)
        checkpoint = self._resolve_checkpoint(run_dir)
        model = self._load_model(meta, checkpoint, device)
        class_names = self._resolve_class_names(meta)
        real_index = self._resolve_real_index(class_names)
        model_cfg = meta.get("model") or {}
        model_name = str(model_cfg.get("name") or "unknown")
        return VitPipeline(
            model=model,
            transform=transform,
            device=device,
            class_names=class_names,
            real_index=real_index,
            run_dir=run_dir,
            model_name=model_name,
        )

    def _resolve_device(self) -> torch.device:
        """추론에 사용할 디바이스를 결정한다.

        Returns:
            사용할 torch 디바이스.
        """

        name = (self._settings.vit_device_name or "auto").lower()
        if name != "auto":
            return torch.device(name)
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _resolve_run_dir(self) -> Path:
        """실험 실행 디렉터리를 결정한다.

        Returns:
            실행 디렉터리 경로.
        """

        if self._settings.vit_run_dir is not None:
            run_dir = self._settings.vit_run_dir
            if not run_dir.is_dir():
                raise FileNotFoundError(
                    f"VIT_RUN_DIR 경로를 찾을 수 없습니다: {run_dir}"
                )
            return run_dir
        root = self._settings.vit_run_root
        if not root.is_dir():
            raise FileNotFoundError(f"실험 루트 디렉토리를 찾을 수 없습니다: {root}")
        candidates = sorted([path for path in root.iterdir() if path.is_dir()])
        if not candidates:
            raise FileNotFoundError(f"실험 루트에 실행 기록이 없습니다: {root}")
        return candidates[-1]

    def _load_meta(self, run_dir: Path) -> dict[str, Any]:
        """meta.yaml 파일을 로드한다.

        Args:
            run_dir: 실험 실행 디렉터리.

        Returns:
            meta.yaml 파싱 결과.
        """

        meta_path = run_dir / "meta.yaml"
        if not meta_path.is_file():
            raise FileNotFoundError(f"meta.yaml을 찾을 수 없습니다: {meta_path}")
        with meta_path.open("r", encoding="utf-8") as fp:
            return yaml.safe_load(fp) or {}

    def _build_transform(self, meta: dict[str, Any]):
        """검증용 변환 파이프라인을 구성한다.

        Args:
            meta: 실험 메타데이터.

        Returns:
            torch 텐서 변환 파이프라인.
        """

        dataset_cfg = meta.get("dataset", {})
        config = load_dataset_config(dataset_cfg)
        return build_val_transforms(config)

    def _resolve_checkpoint(self, run_dir: Path) -> Path:
        """사용할 체크포인트 경로를 반환한다.

        Args:
            run_dir: 실험 실행 디렉터리.

        Returns:
            체크포인트 파일 경로.
        """

        preferred = self._settings.vit_checkpoint_name or "best.pt"
        ckpt_path = run_dir / "checkpoints" / preferred
        if ckpt_path.is_file():
            return ckpt_path
        fallback = run_dir / "checkpoints" / "last.pt"
        if fallback.is_file():
            self._logger.warning(
                "선호한 체크포인트 %s 를 찾지 못해 last.pt를 사용합니다", ckpt_path.name
            )
            return fallback
        raise FileNotFoundError(
            f"체크포인트를 찾을 수 없습니다: {ckpt_path} 또는 {fallback}"
        )

    def _load_model(
        self, meta: dict[str, Any], checkpoint: Path, device: torch.device
    ) -> torch.nn.Module:
        """체크포인트에서 모델을 로드한다.

        Args:
            meta: 실험 메타데이터.
            checkpoint: 체크포인트 파일 경로.
            device: 로딩에 사용할 디바이스.

        Returns:
            로드된 torch 모델.
        """

        model_cfg = meta.get("model") or {}
        name = model_cfg.get("name")
        if not name:
            raise ValueError("meta.yaml에 model.name 항목이 없습니다.")
        params = model_cfg.get("params") or {}
        model = get_model(name, **params)

        state = torch.load(checkpoint, map_location="cpu")
        if isinstance(state, dict) and "model" in state:
            state = state["model"]
        if not isinstance(state, dict):
            raise RuntimeError(f"지원되지 않는 체크포인트 형식입니다: {checkpoint}")
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing:
            self._logger.warning("모델 로드시 누락된 파라미터: %s", missing)
        if unexpected:
            self._logger.warning("모델 로드시 예기치 않은 파라미터: %s", unexpected)
        model.to(device)
        model.eval()
        return model

    def _resolve_class_names(self, meta: dict[str, Any]) -> List[str]:
        """클래스 이름 목록을 반환한다.

        Args:
            meta: 실험 메타데이터.

        Returns:
            클래스 이름 리스트.
        """

        class_names = meta.get("dataset", {}).get("class_names")
        if isinstance(class_names, list):
            return [str(name) for name in class_names]
        if isinstance(class_names, tuple):
            return [str(name) for name in class_names]
        return []

    def _resolve_real_index(self, class_names: List[str]) -> int:
        """실제(REAL) 클래스의 인덱스를 식별한다.

        Args:
            class_names: 클래스 이름 리스트.

        Returns:
            REAL 클래스 인덱스.
        """

        lowered = [name.lower() for name in class_names]
        for candidate in ("nature", "real", "genuine"):
            if candidate in lowered:
                return lowered.index(candidate)
        return 0


__all__ = ["VitInferenceService", "VitPipeline"]
