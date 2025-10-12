"""추론 서비스 계층을 정의한다.

목적:
    ViT 기반 이미지 위조 판별 파이프라인을 초기화하고 FastAPI 및 워커에서
    재사용할 수 있도록 비동기 친화적인 서비스 클래스를 제공한다. 비동기
    배치 큐를 도입해 GPU 활용률을 높이고, 단계별 로깅을 강화한다.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import yaml
from PIL import Image

from core.datasets.base import load_dataset_config
from core.datasets.transforms import build_val_transforms
from core.models.multipatch import aggregate_scores, generate_patches
from core.models.registry import get_model
from core.utils.logger import get_logger, log_time

from api.config import RuntimeSettings
from api.services.batching import BatchInferenceQueue


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
    inference_mode: str
    inference_scales: Tuple[int, ...]
    inference_n_patches: int
    inference_aggregate: str
    uncertainty_band: Tuple[float, float]


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
        self._batch_queue = BatchInferenceQueue(
            max_batch_size=settings.vit_max_batch_size,
            max_wait_ms=settings.vit_max_batch_wait_ms,
            runner=self._predict_batch,
            queue_name="vit",
        )

    async def startup(self) -> None:
        """FastAPI 애플리케이션 시작 시 초기화를 수행한다."""

        await self.ensure_ready()
        await self._batch_queue.start()

    async def shutdown(self) -> None:
        """FastAPI 종료 시 자원을 정리한다."""

        await self._batch_queue.close()

    async def predict_image(self, image: Image.Image) -> List[float]:
        """이미지에 대한 위조 확률 분포를 반환한다."""

        probs, _ = await self.predict_image_with_metadata(image)
        return probs

    async def predict_image_with_metadata(
        self, image: Image.Image
    ) -> Tuple[List[float], Dict[str, Any]]:
        """이미지 추론과 함께 메타데이터를 반환한다."""

        pipeline = await self._ensure_pipeline()
        metadata: Dict[str, Any] = {
            "mode": pipeline.inference_mode,
            "n_patches": pipeline.inference_n_patches,
            "scales": list(pipeline.inference_scales),
            "aggregate": pipeline.inference_aggregate,
        }

        rgb_image = image if image.mode == "RGB" else image.convert("RGB")
        use_multipatch = self._should_use_multipatch(pipeline)

        if use_multipatch:
            patch_tensors = self._prepare_multipatch_tensors(rgb_image, pipeline)
            if len(patch_tensors) <= 1:
                use_multipatch = False
            else:
                metadata["mode"] = "multi"
                metadata["patch_count"] = len(patch_tensors)
                self._logger.info(
                    "멀티패치 추론 시작 - patches=%d scales=%s aggregate=%s",
                    len(patch_tensors),
                    list(pipeline.inference_scales),
                    pipeline.inference_aggregate,
                )
                patch_futures = [
                    self._batch_queue.enqueue(tensor) for tensor in patch_tensors
                ]
                patch_distributions = await asyncio.gather(*patch_futures)
                aggregated = self._aggregate_patch_probs(patch_distributions, pipeline)
                real_idx = pipeline.real_index if pipeline.real_index < len(aggregated) else 0
                ai_idx = 1 if real_idx == 0 else 0
                real_score = float(aggregated[real_idx]) if 0 <= real_idx < len(aggregated) else float("nan")
                ai_score = float(aggregated[ai_idx]) if 0 <= ai_idx < len(aggregated) else float("nan")
                self._logger.info(
                    "멀티패치 추론 완료 - patches=%d real=%.4f ai=%.4f",
                    len(patch_tensors),
                    real_score,
                    ai_score,
                )
                return aggregated, metadata

        tensor = self._prepare_tensor(rgb_image, pipeline)
        metadata["mode"] = "single"
        metadata["patch_count"] = 1
        probs = await self._batch_queue.enqueue(tensor)
        return probs, metadata

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

    @log_time
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
        inference_cfg = self._resolve_inference_config(meta)
        return VitPipeline(
            model=model,
            transform=transform,
            device=device,
            class_names=class_names,
            real_index=real_index,
            run_dir=run_dir,
            model_name=model_name,
            **inference_cfg,
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

    @log_time
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

    @log_time
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

    @log_time
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

    @log_time
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

    @log_time
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

    def _resolve_inference_config(self, meta: dict[str, Any]) -> Dict[str, Any]:
        """멀티패치/멀티스케일 추론 설정을 계산한다."""

        raw_cfg = meta.get("inference") or {}
        raw_scales = raw_cfg.get("multiscale") or raw_cfg.get("scales") or ()
        if isinstance(raw_scales, (int, float)):
            scales = (int(raw_scales),)
        elif isinstance(raw_scales, (list, tuple)):
            normalized: List[int] = []
            for value in raw_scales:
                try:
                    normalized.append(int(float(value)))
                except (TypeError, ValueError):
                    continue
            scales = tuple(sorted({val for val in normalized if val > 0}))
        else:
            scales = ()

        try:
            n_patches = int(raw_cfg.get("n_patches") or raw_cfg.get("patches") or 1)
        except (TypeError, ValueError):
            n_patches = 1
        if n_patches < 1:
            n_patches = 1

        aggregate = str(raw_cfg.get("aggregate") or "mean").lower()
        if aggregate not in {"mean", "max", "quality_weighted"}:
            aggregate = "mean"

        mode = "multi" if n_patches > 1 or len(scales) > 1 else "single"
        if not self._settings.vit_enable_multipatch:
            mode = "single"
            n_patches = 1

        if not scales:
            scales = (224,)

        uncertainty_band = (
            float(self._settings.uncertainty_band_low),
            float(self._settings.uncertainty_band_high),
        )

        return {
            "inference_mode": mode,
            "inference_scales": scales,
            "inference_n_patches": n_patches,
            "inference_aggregate": aggregate,
            "uncertainty_band": uncertainty_band,
        }

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

    def _prepare_tensor(
        self, image: Image.Image, pipeline: VitPipeline
    ) -> torch.Tensor:
        """이미지를 배치 추론에 맞게 텐서로 변환한다.

        Args:
            image: 입력 PIL 이미지.
            pipeline: 현재 활성화된 추론 파이프라인.

        Returns:
            배치 차원이 포함된 torch 텐서.
        """

        rgb = image if image.mode == "RGB" else image.convert("RGB")
        tensor = pipeline.transform(rgb).unsqueeze(0)
        return tensor.contiguous()

    def _should_use_multipatch(self, pipeline: VitPipeline) -> bool:
        """멀티패치 추론 사용 여부를 결정한다."""

        if not self._settings.vit_enable_multipatch:
            return False
        if pipeline.inference_mode != "multi":
            return False
        return pipeline.inference_n_patches > 1

    def _prepare_multipatch_tensors(
        self, image: Image.Image, pipeline: VitPipeline
    ) -> List[torch.Tensor]:
        """멀티패치 텐서 배치를 생성한다."""

        sizes: Sequence[int] = pipeline.inference_scales or (224,)
        patches = generate_patches(
            image, sizes=sizes, n_patches=pipeline.inference_n_patches
        )
        tensors: List[torch.Tensor] = []
        for patch in patches:
            rgb_patch = patch if patch.mode == "RGB" else patch.convert("RGB")
            tensor = pipeline.transform(rgb_patch).unsqueeze(0).contiguous()
            tensors.append(tensor)
        return tensors

    def _aggregate_patch_probs(
        self, patch_probs: Sequence[Sequence[float]], pipeline: VitPipeline
    ) -> List[float]:
        """패치별 확률을 하나의 분포로 병합한다."""

        if not patch_probs:
            raise ValueError("patch_probs is empty")
        first = patch_probs[0]
        num_classes = len(first)
        if num_classes == 0:
            raise ValueError("probability distribution must be non-empty")
        real_idx = pipeline.real_index if pipeline.real_index < num_classes else 0
        if num_classes != 2:
            self._logger.warning(
                "멀티패치 집계는 2-클래스 모델에서 최적화되어 있습니다 (classes=%d)",
                num_classes,
            )
        ai_idx = 1 if real_idx == 0 else 0

        patch_scores = []
        for dist in patch_probs:
            if len(dist) != num_classes:
                raise ValueError("inconsistent probability dimensions across patches")
            p_real = float(dist[real_idx])
            if num_classes >= 2 and ai_idx < num_classes:
                p_ai = float(dist[ai_idx])
            else:
                p_ai = float(max(0.0, 1.0 - p_real))
            patch_scores.append({"ai": p_ai, "real": p_real})

        aggregated = aggregate_scores(
            patch_scores, method=pipeline.inference_aggregate or "mean"
        )

        if num_classes >= 2 and ai_idx < num_classes:
            fused = [float(value) for value in first]
            fused[real_idx] = aggregated["real"]
            fused[ai_idx] = aggregated["ai"]
        else:
            fused = [aggregated["real"], aggregated["ai"]]

        total = sum(fused)
        if total > 0:
            fused = [float(value / total) for value in fused]
        return fused

    async def _predict_batch(self, batch: torch.Tensor) -> torch.Tensor:
        """비동기 배치 추론을 실행한다.

        Args:
            batch: (N, C, H, W) 형태의 입력 텐서.

        Returns:
            소프트맥스 확률 텐서.
        """

        pipeline = await self._ensure_pipeline()
        async with self._predict_lock:
            with torch.inference_mode():
                device_batch = batch.to(pipeline.device, non_blocking=True)
                logits = pipeline.model(device_batch)
                probs = torch.softmax(logits, dim=1)
        return probs.detach().cpu()


__all__ = ["VitInferenceService", "VitPipeline"]
