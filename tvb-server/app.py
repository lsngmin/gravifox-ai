import sys
import time
import logging
import io
import json
from pathlib import Path
from datetime import datetime

import yaml
import torch
from PIL import Image
from fastapi import FastAPI, UploadFile, File, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import tempfile, cv2
import onnxruntime as ort

import os, uuid, datetime as dt, asyncio
from typing import Any
from typing import Optional

_THIS_DIR = Path(__file__).resolve().parent
_THIS_DIR_STR = str(_THIS_DIR)
if _THIS_DIR_STR not in sys.path:
    sys.path.insert(0, _THIS_DIR_STR)

PROJECT_ROOT = _THIS_DIR.parent
PROJECT_ROOT_STR = str(PROJECT_ROOT)
if PROJECT_ROOT_STR not in sys.path:
    sys.path.insert(0, PROJECT_ROOT_STR)

from core.datasets.base import load_dataset_config
from core.datasets.transforms import build_val_transforms
from core.models.registry import get_model
from detection.gf1.config import (
    create_onnx_session,
    DET_ONNX_PATH,
    CLS_ONNX_PATH,
    DET_ONNX_PROVIDERS,
    CLS_ONNX_PROVIDERS,
    VIDEO_PATH,
    CONF,
    FPS,
    CLIP_LEN,
    CLIP_STRIDE,
    ALIGN,
    LAYOUT,
    RGB,
    MEAN,
    STD,
    THRESHOLD,
    HIGH_CONF,
    SPECTRAL_R0,
    POSE_DELTA_OUTLIER,
)
from detection.gf1 import run_video
from model_registry import list_models, get_default_model
from settings import (
    ENABLE_MQ,
    TVB_MAX_CONCURRENCY,
    VIT_CHECKPOINT_NAME,
    VIT_DEVICE_NAME,
    VIT_RUN_DIR,
    VIT_RUN_ROOT,
)

# Unified file store root via env so FastAPI and worker see the same path
# Default: /tmp/uploads (matches worker.py default)
FILE_STORE_ROOT = Path(os.environ.get("FILE_STORE_ROOT", "/tmp/uploads"))
FILE_STORE_ROOT.mkdir(parents=True, exist_ok=True)

# Upload policy (can be tuned via env)
MAX_IMAGE_MB = float(os.environ.get("MAX_IMAGE_MB", 5))
MAX_VIDEO_MB = float(os.environ.get("MAX_VIDEO_MB", 50))
FILE_TTL_HOURS = float(os.environ.get("FILE_TTL_HOURS", 24))

ALLOWED_IMAGE_EXTS = {"jpg", "jpeg", "png", "webp"}
ALLOWED_VIDEO_EXTS = {"mp4", "mov", "webm"}

logger = logging.getLogger(__name__)


def _resolve_run_dir() -> Path:
    """환경 변수 기반으로 최신 실험 경로를 찾는다."""

    if VIT_RUN_DIR is not None:
        run_dir = VIT_RUN_DIR.expanduser().resolve()
        if not run_dir.is_dir():
            raise FileNotFoundError(f"VIT_RUN_DIR 경로를 찾을 수 없습니다: {run_dir}")
        return run_dir

    root = (
        VIT_RUN_ROOT.expanduser().resolve()
        if VIT_RUN_ROOT is not None
        else PROJECT_ROOT / "experiments" / "vit_residual_fusion"
    )

    if not root.is_dir():
        raise FileNotFoundError(f"실험 루트 디렉토리를 찾을 수 없습니다: {root}")

    candidates = sorted([path for path in root.iterdir() if path.is_dir()])
    if not candidates:
        raise FileNotFoundError(f"실험 루트에 실행 기록이 없습니다: {root}")

    return candidates[-1]


def _load_meta(run_dir: Path) -> dict[str, Any]:
    meta_path = run_dir / "meta.yaml"
    if not meta_path.is_file():
        raise FileNotFoundError(f"meta.yaml을 찾을 수 없습니다: {meta_path}")
    with meta_path.open("r", encoding="utf-8") as fp:
        return yaml.safe_load(fp) or {}


def _resolve_checkpoint(run_dir: Path) -> Path:
    preferred = VIT_CHECKPOINT_NAME or "best.pt"
    ckpt_path = run_dir / "checkpoints" / preferred
    if ckpt_path.is_file():
        return ckpt_path
    fallback = run_dir / "checkpoints" / "last.pt"
    if fallback.is_file():
        logger.warning(
            "선호한 체크포인트 %s 를 찾지 못해 last.pt를 사용합니다", ckpt_path.name
        )
        return fallback
    raise FileNotFoundError(
        f"체크포인트를 찾을 수 없습니다: {ckpt_path} 또는 {fallback}"
    )


def _resolve_real_index(class_names: Any) -> int:
    if isinstance(class_names, (list, tuple)):
        lowered = [str(name).lower() for name in class_names]
        for candidate in ("nature", "real", "genuine"):
            if candidate in lowered:
                return lowered.index(candidate)
        return 0
    return 0


def _build_transform(meta: dict[str, Any]):
    dataset_cfg = meta.get("dataset", {})
    config = load_dataset_config(dataset_cfg)
    return build_val_transforms(config)


def _load_torch_model(
    meta: dict[str, Any], checkpoint: Path, device: torch.device
) -> torch.nn.Module:
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
        logger.warning("모델 로드시 누락된 파라미터: %s", missing)
    if unexpected:
        logger.warning("모델 로드시 예기치 않은 파라미터: %s", unexpected)

    model.to(device)
    model.eval()
    return model


def _initialize_vit_pipeline():
    device_name = VIT_DEVICE_NAME
    if device_name and device_name != "auto":
        device = torch.device(device_name)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    run_dir = _resolve_run_dir()
    meta = _load_meta(run_dir)
    transform = _build_transform(meta)
    checkpoint = _resolve_checkpoint(run_dir)
    model = _load_torch_model(meta, checkpoint, device)
    model_cfg = meta.get("model") or {}
    class_names = meta.get("dataset", {}).get("class_names")
    if isinstance(class_names, tuple):
        class_names = list(class_names)
    elif not isinstance(class_names, list):
        class_names = []
    real_index = _resolve_real_index(class_names)

    logger.info(
        "VIT 추론 파이프라인 초기화 완료 - run=%s device=%s checkpoint=%s",
        run_dir,
        device,
        checkpoint.name,
    )
    return {
        "model": model,
        "transform": transform,
        "device": device,
        "class_names": class_names,
        "real_index": real_index,
        "run_dir": run_dir,
        "model_name": model_cfg.get("name", "unknown"),
    }


try:
    _vit_pipeline = _initialize_vit_pipeline()
except Exception as exc:  # pragma: no cover - 초기화 실패 시 런타임에 노출
    logger.error("VIT 추론 파이프라인 초기화 실패: %s", exc)
    _vit_pipeline = {"error": exc}
_vit_lock = asyncio.Lock()


def _infer_kind(filename: str, content_type: Optional[str]) -> str:
    ct = (content_type or "").lower()
    if ct.startswith("image/"):
        return "image"
    if ct.startswith("video/"):
        return "video"
    # fallback by extension
    ext = Path(filename or "").suffix.lower().lstrip(".")
    if ext in ALLOWED_IMAGE_EXTS:
        return "image"
    if ext in ALLOWED_VIDEO_EXTS:
        return "video"
    return "unknown"


app = FastAPI()

# Allow dev origins; tighten in production
app.add_middleware(
    CORSMiddleware,
    allow_origins=(
        os.environ.get("CORS_ALLOW_ORIGINS", "*").split(",")
        if os.environ.get("CORS_ALLOW_ORIGINS")
        else ["*"]
    ),
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


class Data(BaseModel):
    message: str


def _predict_with_vit(image: Image.Image) -> list[float]:
    if "error" in _vit_pipeline:
        raise RuntimeError(f"모델 초기화 실패: {_vit_pipeline['error']}")

    model: torch.nn.Module = _vit_pipeline["model"]
    transform = _vit_pipeline["transform"]
    device: torch.device = _vit_pipeline["device"]

    image_rgb = image.convert("RGB")
    tensor = transform(image_rgb).unsqueeze(0).to(device)
    with torch.inference_mode():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1)
    return probs.squeeze(0).detach().cpu().tolist()


def predict_image(image: Image.Image):
    return _predict_with_vit(image)


@app.post("/predeict/video/")
async def predict_video():
    det_sess = create_onnx_session("detector", DET_ONNX_PATH, DET_ONNX_PROVIDERS)
    # Prepare ONNX classifier session (single backend)
    cls_sess = create_onnx_session("classifier", CLS_ONNX_PATH, CLS_ONNX_PROVIDERS)

    result = run_video(
        video_path=VIDEO_PATH,
        det_sess=det_sess,
        cls_sess=cls_sess,
        conf_th=CONF,
        sample_fps=FPS,
        clip_len=CLIP_LEN,
        clip_stride=CLIP_STRIDE,
        align_size=ALIGN,
        layout=LAYOUT,
        rgb=RGB,
        mean=MEAN,
        std=STD,
        verdict_threshold=THRESHOLD,
        high_conf_threshold=HIGH_CONF,
        spectral_r0=SPECTRAL_R0,
        pose_delta_outlier=POSE_DELTA_OUTLIER,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


@app.post("/upload/")
async def upload_image(file: UploadFile = File(...)):

    image_data = await file.read()
    try:
        image = Image.open(io.BytesIO(image_data))
    except Exception:
        raise HTTPException(status_code=400, detail="invalid image file")

    start_time = time.time()
    probabilities: list[float] = []
    try:
        async with _vit_lock:
            probabilities = predict_image(image)
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    finally:
        image.close()
    end_time = time.time()

    real_index = int(_vit_pipeline.get("real_index", 0) or 0)
    if real_index >= len(probabilities):
        real_index = 0
    prob_real = float(probabilities[real_index]) if probabilities else 0.0

    return {
        "timestamp": datetime.now().timestamp(),
        "used_model": _vit_pipeline.get("model_name", "unknown"),
        "image_uuid": file.filename,
        "prediction_time": round(end_time - start_time, 2),
        "predicted_probability": round(prob_real, 4),
        "predicted_class": "Real" if prob_real > 0.5 else "Fake",
    }


# Generic media upload (image/video) → save to disk and return uploadId
# Note: distinct path from "/upload/" to avoid breaking existing image classifier.
UPLOAD_TOKEN = os.environ.get("UPLOAD_TOKEN")


@app.post("/upload")
async def upload_media(
    file: UploadFile = File(...),
    x_upload_token: Optional[str] = Header(default=None, alias="X-Upload-Token"),
):
    try:
        if UPLOAD_TOKEN and x_upload_token != UPLOAD_TOKEN:
            raise HTTPException(status_code=401, detail="invalid upload token")
        kind = _infer_kind(file.filename or "", getattr(file, "content_type", None))
        if kind == "unknown":
            raise HTTPException(status_code=415, detail="unsupported media type")

        max_bytes = int(
            (MAX_IMAGE_MB if kind == "image" else MAX_VIDEO_MB) * 1024 * 1024
        )

        now = dt.datetime.utcnow().strftime("%Y%m%dT%H%M%S")
        suffix = Path(file.filename or "media").suffix.lower()
        upload_id = f"{now}_{uuid.uuid4().hex}{suffix}"
        dest = FILE_STORE_ROOT / upload_id

        written = 0
        with open(dest, "wb") as f:
            while True:
                chunk = await file.read(1024 * 1024)
                if not chunk:
                    break
                written += len(chunk)
                if written > max_bytes:
                    try:
                        f.close()
                    finally:
                        dest.unlink(missing_ok=True)
                    raise HTTPException(status_code=413, detail="file too large")
                f.write(chunk)

        return {"uploadId": upload_id}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"upload failed: {e}")


@app.get("/models")
async def list_available_models():
    catalog = list_models()
    default_model = get_default_model()
    return {
        "defaultKey": default_model.key,
        "items": [
            {
                "key": m.key,
                "name": m.name,
                "version": m.version,
                "description": m.description,
                "type": m.type,
                "input": m.input,
                "threshold": m.threshold,
                "labels": list(m.labels),
            }
            for m in catalog
        ],
    }


@app.get("/")
async def index():
    return {"received_message": "Hello World"}


# ========== Background: simple TTL cleanup ==========
async def _ttl_cleanup_loop():
    if FILE_TTL_HOURS <= 0:
        return
    ttl = dt.timedelta(hours=FILE_TTL_HOURS)
    while True:
        try:
            now = dt.datetime.utcnow()
            for p in FILE_STORE_ROOT.glob("*"):
                try:
                    mtime = dt.datetime.utcfromtimestamp(p.stat().st_mtime)
                    if now - mtime > ttl:
                        p.unlink()
                except Exception:
                    pass
        except Exception:
            pass
        await asyncio.sleep(3600)


@app.on_event("startup")
async def _on_startup():
    # spawn TTL clean task
    asyncio.create_task(_ttl_cleanup_loop())
    # optional: start MQ consumer if configured
    if ENABLE_MQ:
        try:
            # Use module imports from the same directory (not package-relative)
            # because folder names contain dashes and cannot be used as package names.
            from mq import MQ
            from worker import run_analysis

            mq = MQ()
            await mq.connect()

            # bounded concurrency (default 1)
            sem = asyncio.Semaphore(max(1, TVB_MAX_CONCURRENCY))

            from typing import Dict

            async def handle_request(payload: Dict[str, Any]):
                job_id = payload.get("jobId")
                upload_id = payload.get("uploadId")
                params = payload.get("params")

                async def _task():
                    async with sem:
                        try:
                            await run_analysis(mq, job_id, upload_id, params)
                        except Exception as _e:
                            try:
                                from mq import publish_failed as _pf

                                await _pf(
                                    mq,
                                    job_id or "",
                                    f"analysis error: {_e}",
                                    reason_code="WORKER_EXCEPTION",
                                )
                            except Exception:
                                pass

                # schedule and return immediately (message is acked by consumer)
                asyncio.create_task(_task())

            asyncio.create_task(mq.consume_requests(handle_request))
        except Exception as e:
            # Log and continue without MQ
            print(f"[startup] MQ disabled due to error: {e}")
