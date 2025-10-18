"""추론 서비스 계층을 정의한다.

목적:
    ViT 기반 이미지 위조 판별 파이프라인을 초기화하고 FastAPI 및 워커에서
    재사용할 수 있도록 비동기 친화적인 서비스 클래스를 제공한다. 비동기
    배치 큐를 도입해 GPU 활용률을 높이고, 단계별 로깅을 강화한다.
"""

from __future__ import annotations

import asyncio
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import yaml
from PIL import Image

from core.datasets.base import load_dataset_config
from core.datasets.transforms import build_val_transforms
from core.models.multipatch import (
    PatchSample,
    aggregate_scores,
    compute_patch_weights,
    generate_patches,
    estimate_priority_regions,
)
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
    inference_cell_sizes: Tuple[int, ...]
    inference_aggregate: str
    inference_overlap: Any
    inference_jitter: Any
    inference_max_patches: Optional[int]
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
            "cell_sizes": list(pipeline.inference_cell_sizes),
            "aggregate": pipeline.inference_aggregate,
            "overlap": pipeline.inference_overlap,
            "jitter": pipeline.inference_jitter,
            "max_patches": pipeline.inference_max_patches,
            "uncertainty_band": list(pipeline.uncertainty_band),
        }

        rgb_image = image if image.mode == "RGB" else image.convert("RGB")
        use_multipatch = self._should_use_multipatch(pipeline)

        priority_regions: Optional[List[Tuple[float, float, float, float]]] = None

        if use_multipatch:
            patch_entries, priority_regions = self._prepare_multipatch_inputs(
                rgb_image, pipeline
            )
            if len(patch_entries) <= 1:
                use_multipatch = False
            else:
                metadata["mode"] = "multi"
                metadata["n_patches"] = len(patch_entries)
                metadata["patch_count"] = len(patch_entries)
                if priority_regions:
                    metadata["priority_regions"] = priority_regions
                self._logger.info(
                    "멀티패치 추론 시작 - patches=%d scales=%s aggregate=%s",
                    len(patch_entries),
                    list(pipeline.inference_scales),
                    pipeline.inference_aggregate,
                )
                patch_futures = [
                    self._batch_queue.enqueue(tensor) for tensor, _ in patch_entries
                ]
                patch_distributions = await asyncio.gather(*patch_futures)
                (
                    aggregated,
                    patch_scores,
                    weights,
                    patch_stats,
                ) = self._aggregate_patch_probs(
                    patch_entries, patch_distributions, pipeline
                )
                patch_details = self._summarize_patch_outputs(
                    patch_entries, patch_distributions, patch_scores, weights
                )
                if patch_details:
                    metadata["patches"] = patch_details
                    metadata["grid"] = self._infer_patch_grid(patch_details)
                    metadata["heatmap"] = self._build_heatmap(patch_details)
                if patch_stats:
                    metadata["patch_stats"] = patch_stats
                    adjusted_band, partial_flag = self._adjust_uncertainty_band(
                        pipeline, aggregated, patch_stats
                    )
                    metadata["uncertainty_band_adjusted"] = list(adjusted_band)
                    metadata["partial_suspected"] = bool(partial_flag)
                    self._logger.debug(
                        "패치 통계 - count=%d best_ai=%.3f weighted_ai=%.3f partial=%s",
                        len(patch_entries),
                        patch_stats.get("best_ai", 0.0),
                        patch_stats.get("weighted_ai", 0.0),
                        str(partial_flag),
                    )
                real_idx = pipeline.real_index if pipeline.real_index < len(aggregated) else 0
                ai_idx = 1 if real_idx == 0 else 0
                real_score = float(aggregated[real_idx]) if 0 <= real_idx < len(aggregated) else float("nan")
                ai_score = float(aggregated[ai_idx]) if 0 <= ai_idx < len(aggregated) else float("nan")
                self._logger.info(
                    "멀티패치 추론 완료 - patches=%d real=%.4f ai=%.4f",
                    len(patch_entries),
                    real_score,
                    ai_score,
                )
                return aggregated, metadata

        tensor = self._prepare_tensor(rgb_image, pipeline)
        metadata["mode"] = "single"
        metadata["patch_count"] = 1
        if priority_regions:
            metadata.setdefault("priority_regions", priority_regions)
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
        preferred_run = root / "20251011_043932"
        if preferred_run.is_dir():
            return preferred_run
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

        raw_cell_sizes = (
            raw_cfg.get("cell_sizes")
            or raw_cfg.get("min_cell_sizes")
            or raw_cfg.get("min_cell_size")
            or raw_cfg.get("cell_size")
        )
        if isinstance(raw_cell_sizes, (list, tuple)):
            normalized_cells: List[int] = []
            for value in raw_cell_sizes:
                try:
                    normalized_cells.append(int(max(1, float(value))))
                except (TypeError, ValueError):
                    continue
            if len(normalized_cells) == 1 and len(scales) > 1:
                cell_sizes = tuple([normalized_cells[0]] * len(scales))
            elif len(normalized_cells) == len(scales):
                cell_sizes = tuple(normalized_cells)
            else:
                cell_sizes = tuple(max(1, scale) for scale in scales)
        elif isinstance(raw_cell_sizes, (int, float)):
            val = max(1, int(raw_cell_sizes))
            cell_sizes = tuple([val] * len(scales))
        else:
            cell_sizes = tuple(max(1, scale) for scale in scales)

        try:
            n_patches = int(raw_cfg.get("n_patches") or raw_cfg.get("patches") or 0)
        except (TypeError, ValueError):
            n_patches = 0
        if n_patches < 0:
            n_patches = 0

        aggregate = str(raw_cfg.get("aggregate") or "mean").lower()
        if aggregate not in {"mean", "max", "quality_weighted"}:
            aggregate = "mean"

        mode = "multi" if n_patches == 0 or n_patches > 1 or len(scales) > 1 else "single"
        if not self._settings.vit_enable_multipatch:
            mode = "single"
            n_patches = 1

        if not scales:
            scales = (224,)

        overlap_cfg = raw_cfg.get("patch_overlap") or raw_cfg.get("overlap")
        jitter_cfg = raw_cfg.get("patch_jitter") or raw_cfg.get("jitter")
        max_patches_cfg = raw_cfg.get("max_patches") or raw_cfg.get("patch_limit")

        def _normalize_scalar(value: Any) -> Any:
            if isinstance(value, (list, tuple)):
                return [
                    float(item) if item is not None else None for item in value
                ]
            if value in (None, "", False):
                return None
            try:
                return float(value)
            except (TypeError, ValueError):
                return None

        overlap_value = _normalize_scalar(overlap_cfg)
        jitter_value = _normalize_scalar(jitter_cfg)
        try:
            max_patches_value = (
                int(max_patches_cfg) if max_patches_cfg not in (None, "", False) else None
            )
        except (TypeError, ValueError):
            max_patches_value = None
        if max_patches_value is not None and max_patches_value <= 0:
            max_patches_value = None

        uncertainty_band = (
            float(self._settings.uncertainty_band_low),
            float(self._settings.uncertainty_band_high),
        )

        return {
            "inference_mode": mode,
            "inference_scales": scales,
            "inference_n_patches": n_patches,
            "inference_cell_sizes": cell_sizes,
            "inference_aggregate": aggregate,
            "inference_overlap": overlap_value,
            "inference_jitter": jitter_value,
            "inference_max_patches": max_patches_value,
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
        if pipeline.inference_n_patches == 0:
            return True
        return pipeline.inference_n_patches > 1

    def _prepare_multipatch_inputs(
        self, image: Image.Image, pipeline: VitPipeline
    ) -> Tuple[List[Tuple[torch.Tensor, PatchSample]], Optional[List[Tuple[float, float, float, float]]]]:
        """멀티패치 텐서와 메타데이터를 생성한다."""

        sizes: Sequence[int] = pipeline.inference_scales or (224,)
        priority_regions = self._detect_priority_regions(image, pipeline)
        patch_samples = generate_patches(
            image,
            sizes=sizes,
            n_patches=pipeline.inference_n_patches,
            min_cell_size=pipeline.inference_cell_sizes,
            overlap=pipeline.inference_overlap,
            jitter=pipeline.inference_jitter,
            max_patches=pipeline.inference_max_patches,
            priority_regions=priority_regions,
        )
        entries: List[Tuple[torch.Tensor, PatchSample]] = []
        for sample in patch_samples:
            rgb_patch = (
                sample.image
                if sample.image.mode == "RGB"
                else sample.image.convert("RGB")
            )
            tensor = pipeline.transform(rgb_patch).unsqueeze(0).contiguous()
            entries.append((tensor, sample))
        return entries, priority_regions

    def _detect_priority_regions(
        self, image: Image.Image, pipeline: VitPipeline
    ) -> Optional[List[Tuple[float, float, float, float]]]:
        """간단한 복잡도 히트맵 기반 우선순위 영역을 추정한다."""

        regions = estimate_priority_regions(image, max_regions=4)
        if regions:
            width, height = image.size
            self._logger.debug(
                "우선순위 영역 추정 - detected=%d width=%d height=%d",
                len(regions),
                width,
                height,
            )
        return regions or None

    def _adjust_uncertainty_band(
        self,
        pipeline: VitPipeline,
        aggregated: Sequence[float],
        stats: Dict[str, float],
    ) -> Tuple[Tuple[float, float], bool]:
        """패치 통계를 활용해 불확실 밴드를 미세 조정한다."""

        low, high = pipeline.uncertainty_band
        if not aggregated:
            return (low, high), False

        num_classes = len(aggregated)
        real_idx = pipeline.real_index if pipeline.real_index < num_classes else 0
        ai_idx = 1 if real_idx == 0 else 0
        ai_prob = float(aggregated[ai_idx]) if ai_idx < num_classes else 1.0 - float(aggregated[real_idx])
        real_prob = float(aggregated[real_idx]) if real_idx < num_classes else 1.0 - ai_prob

        best_ai = float(stats.get("best_ai", 0.0))
        best_real = float(stats.get("best_real", 0.0))
        weighted_ai = float(stats.get("weighted_ai", 0.0))

        adjusted_low, adjusted_high = float(low), float(high)
        partial_flag = False

        within_band = adjusted_low <= ai_prob <= adjusted_high
        if within_band and best_ai >= 0.7:
            delta = max(0.0, best_ai - ai_prob)
            tighten = min(0.05, delta * 0.25)
            adjusted_low = max(0.45, min(0.5, adjusted_low + tighten / 2.0))
            adjusted_high = min(0.55, max(0.5, adjusted_high - tighten / 2.0))
            partial_flag = True
        elif within_band and best_real >= 0.7:
            delta = max(0.0, best_real - real_prob)
            tighten = min(0.05, delta * 0.25)
            adjusted_low = max(0.45, min(0.5, adjusted_low + tighten / 2.0))
            adjusted_high = min(0.55, max(0.5, adjusted_high - tighten / 2.0))
            partial_flag = True
        elif weighted_ai >= 0.6 and ai_prob < adjusted_low:
            adjusted_low = max(0.45, adjusted_low - 0.01)
            partial_flag = True

        if adjusted_high < adjusted_low:
            mid = (adjusted_low + adjusted_high) / 2.0
            adjusted_low = mid - 0.01
            adjusted_high = mid + 0.01

        return (adjusted_low, adjusted_high), partial_flag

    def _aggregate_patch_probs(
        self,
        patch_entries: Sequence[Tuple[torch.Tensor, PatchSample]],
        patch_probs: Sequence[Sequence[float]],
        pipeline: VitPipeline,
    ) -> Tuple[
        List[float],
        List[Dict[str, float]],
        Optional[List[float]],
        Dict[str, float],
    ]:
        """패치별 확률을 병합하고 가중치/통계를 반환한다."""

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
        weights: Optional[List[float]] = None
        patch_samples: Optional[List[PatchSample]] = None
        try:
            patch_samples = [sample for (_, sample) in patch_entries]
        except Exception:
            patch_samples = None
        for dist in patch_probs:
            if len(dist) != num_classes:
                raise ValueError("inconsistent probability dimensions across patches")
            p_real = float(dist[real_idx])
            if num_classes >= 2 and ai_idx < num_classes:
                p_ai = float(dist[ai_idx])
            else:
                p_ai = float(max(0.0, 1.0 - p_real))
            patch_scores.append({"ai": p_ai, "real": p_real})

        scale_distribution: Dict[int, List[float]] = defaultdict(list) if patch_samples else {}
        if patch_samples:
            try:
                weights = compute_patch_weights(patch_samples)
            except Exception:
                weights = None
            for sample, score in zip(patch_samples, patch_scores):
                scale_distribution[int(sample.scale)].append(score["ai"])

        aggregated = aggregate_scores(
            patch_scores,
            method=pipeline.inference_aggregate or "mean",
            weights=weights,
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
        ai_values = [score["ai"] for score in patch_scores]
        real_values = [score["real"] for score in patch_scores]
        stats = {
            "best_ai": float(max(ai_values) if ai_values else 0.0),
            "best_real": float(max(real_values) if real_values else 0.0),
            "mean_ai": float(sum(ai_values) / len(ai_values)) if ai_values else 0.0,
            "mean_real": float(sum(real_values) / len(real_values)) if real_values else 0.0,
            "weighted_ai": float(
                sum(w * s for w, s in zip(weights, ai_values)) if weights else 0.0
            ),
        }
        if scale_distribution:
            stats["scale_ai_mean"] = {
                int(scale): float(sum(values) / len(values)) if values else 0.0
                for scale, values in scale_distribution.items()
            }
        return fused, patch_scores, weights, stats

    def _summarize_patch_outputs(
        self,
        patch_entries: Sequence[Tuple[torch.Tensor, PatchSample]],
        patch_distributions: Sequence[Sequence[float]],
        patch_scores: Sequence[Dict[str, float]],
        weights: Optional[Sequence[float]] = None,
    ) -> List[Dict[str, Any]]:
        """패치 추론 결과를 프론트엔드가 활용할 수 있는 형태로 정리한다."""

        summaries: List[Dict[str, Any]] = []
        for idx, ((_, sample), distribution, score) in enumerate(
            zip(patch_entries, patch_distributions, patch_scores)
        ):
            bbox = {
                "x1": float(sample.bbox[0]),
                "y1": float(sample.bbox[1]),
                "x2": float(sample.bbox[2]),
                "y2": float(sample.bbox[3]),
            }
            grid = {"row": int(sample.grid_index[0]), "col": int(sample.grid_index[1])}
            summary = {
                "index": int(sample.patch_index),
                "scale": int(sample.scale),
                "scale_index": int(sample.scale_index),
                "bbox": bbox,
                "grid": grid,
                "priority": bool(sample.priority),
                "complexity": float(sample.complexity),
                "source": str(sample.source),
                "weight": float(weights[idx]) if weights and idx < len(weights) else None,
                "scores": {
                    "ai": float(score["ai"]),
                    "real": float(score["real"]),
                    "distribution": [float(value) for value in distribution],
                },
            }
            summaries.append(summary)

        summaries.sort(key=lambda item: item["index"])
        return summaries

    @staticmethod
    def _infer_patch_grid(patch_details: Sequence[Dict[str, Any]]) -> Dict[str, int]:
        """패치 메타데이터에서 격자 크기를 추론한다."""

        if not patch_details:
            return {"rows": 0, "cols": 0}

        rows = [
            int(detail.get("grid", {}).get("row", -1))
            for detail in patch_details
            if int(detail.get("grid", {}).get("row", -1)) >= 0
        ]
        cols = [
            int(detail.get("grid", {}).get("col", -1))
            for detail in patch_details
            if int(detail.get("grid", {}).get("col", -1)) >= 0
        ]
        if not rows or not cols:
            return {"rows": 0, "cols": 0}

        max_row = max(rows)
        max_col = max(cols)
        return {"rows": int(max_row) + 1, "cols": int(max_col) + 1}

    @staticmethod
    def _build_heatmap(patch_details: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        """패치별 점수를 격자 형태의 히트맵으로 변환한다."""

        if not patch_details:
            return {"rows": 0, "cols": 0, "cells": []}

        valid_details = [
            detail
            for detail in patch_details
            if int(detail.get("grid", {}).get("row", -1)) >= 0
            and int(detail.get("grid", {}).get("col", -1)) >= 0
        ]
        if not valid_details:
            return {"rows": 0, "cols": 0, "cells": []}

        rows = max(detail["grid"]["row"] for detail in valid_details) + 1
        cols = max(detail["grid"]["col"] for detail in valid_details) + 1

        cells: Dict[Tuple[int, int], Dict[str, Any]] = {}
        for detail in valid_details:
            row = int(detail["grid"]["row"])
            col = int(detail["grid"]["col"])
            key = (row, col)
            cell = cells.setdefault(
                key,
                {
                    "row": row,
                    "col": col,
                    "ai_values": [],
                    "real_values": [],
                    "scales": [],
                },
            )

            ai_score = float(detail["scores"]["ai"])
            real_score = float(detail["scores"]["real"])
            cell["ai_values"].append(ai_score)
            cell["real_values"].append(real_score)

            cell["scales"].append(
                {
                    "scale": int(detail["scale"]),
                    "scale_index": int(detail["scale_index"]),
                    "bbox": {
                        "x1": float(detail["bbox"]["x1"]),
                        "y1": float(detail["bbox"]["y1"]),
                        "x2": float(detail["bbox"]["x2"]),
                        "y2": float(detail["bbox"]["y2"]),
                    },
                    "scores": {
                        "ai": ai_score,
                        "real": real_score,
                        "distribution": [
                            float(value) for value in detail["scores"]["distribution"]
                        ],
                    },
                }
            )

        cells_payload: List[Dict[str, Any]] = []
        for cell in cells.values():
            ai_values = cell.pop("ai_values")
            real_values = cell.pop("real_values")
            scales = cell["scales"]
            scales.sort(key=lambda item: (item["scale_index"], item["scale"]))
            ai_mean = sum(ai_values) / len(ai_values) if ai_values else 0.0
            real_mean = sum(real_values) / len(real_values) if real_values else 0.0
            best_scale = max(scales, key=lambda item: item["scores"]["ai"])
            cells_payload.append(
                {
                    "row": cell["row"],
                    "col": cell["col"],
                    "ai_mean": float(ai_mean),
                    "real_mean": float(real_mean),
                    "ai_max": float(max(ai_values) if ai_values else 0.0),
                    "real_max": float(max(real_values) if real_values else 0.0),
                    "best_scale_index": int(best_scale["scale_index"]),
                    "best_scale": int(best_scale["scale"]),
                    "scales": scales,
                }
            )

        cells_payload.sort(key=lambda item: (item["row"], item["col"]))
        return {"rows": rows, "cols": cols, "cells": cells_payload}

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
