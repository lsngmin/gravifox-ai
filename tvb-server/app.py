import sys
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent
_THIS_DIR_STR = str(_THIS_DIR)
if _THIS_DIR_STR not in sys.path:
    sys.path.insert(0, _THIS_DIR_STR)

import tf_keras
from pydantic import BaseModel
from fastapi.responses import JSONResponse
from tf_keras.src.utils import load_img, img_to_array
from fastapi import FastAPI, UploadFile, File, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
import tempfile, cv2
from PIL import Image
import io, json
import numpy as np
import tensorflow as tf
import time
from datetime import datetime
import onnxruntime as ort
# BNKPS_MODEL_PATH = "tvb-model/scrfd/bnkps.onnx"
# scrfd_sess = ort.InferenceSession(BNKPS_MODEL_PATH, providers=["CPUExecutionProvider"])
# detector = SCRFDDetector(session=scrfd_sess)

import os, uuid, datetime as dt, asyncio
from typing import Any

from settings import ENABLE_MQ, TVB_MAX_CONCURRENCY
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

from typing import Optional

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
    allow_origins=os.environ.get("CORS_ALLOW_ORIGINS", "*").split(",") if os.environ.get("CORS_ALLOW_ORIGINS") else ["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)
class Data(BaseModel):
    message: str

def predict_image(image: Image.Image):
    model = tf_keras.models.load_model("../tvb-model/Xception")

    img = image.resize((256, 256))  # 이미지 불러오기 및 크기 조정
    img_array = img_to_array(img)  # numpy 배열 변환
    img_array = np.expand_dims(img_array, axis=0)  # 배치 차원 추가 (1, 224, 224, 3)
    img_array = img_array / 255.0  # 정규화 (모델 학습 시 정규화했다면 필요)


    prediction = model.predict(img_array)
    return prediction

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
    image = Image.open(io.BytesIO(image_data))

    start_time = time.time()
    prediction = predict_image(image)
    end_time = time.time()

    analyzeResult = JSONResponse(content=
    {
    "timestamp": datetime.now().timestamp(),
    "used_model": "Xception",
    "image_uuid" : file.filename,
    "prediction_time": round(end_time - start_time, 2),
    "predicted_probability": round(prediction[0].tolist()[0], 4),
    "predicted_class": "Real" if prediction[0].tolist()[0]>0.5 else "Fake",
})
    return analyzeResult


# Generic media upload (image/video) → save to disk and return uploadId
# Note: distinct path from "/upload/" to avoid breaking existing image classifier.
UPLOAD_TOKEN = os.environ.get("UPLOAD_TOKEN")


@app.post("/upload")
async def upload_media(file: UploadFile = File(...), x_upload_token: Optional[str] = Header(default=None, alias="X-Upload-Token")):
    try:
        if UPLOAD_TOKEN and x_upload_token != UPLOAD_TOKEN:
            raise HTTPException(status_code=401, detail="invalid upload token")
        kind = _infer_kind(file.filename or "", getattr(file, "content_type", None))
        if kind == "unknown":
            raise HTTPException(status_code=415, detail="unsupported media type")

        max_bytes = int((MAX_IMAGE_MB if kind == "image" else MAX_VIDEO_MB) * 1024 * 1024)

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
@app.get("/")
async  def index():
    return {"received_message": "Hello World"}


# ========== Background: simple TTL cleanup ==========
async def _ttl_cleanup_loop():
    if FILE_TTL_HOURS <= 0:
        return
    ttl = dt.timedelta(hours=FILE_TTL_HOURS)
    while True:
        try:
            now = dt.datetime.utcnow()
            for p in FILE_STORE_ROOT.glob('*'):
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
                                await _pf(mq, job_id or "", f"analysis error: {_e}", reason_code="WORKER_EXCEPTION")
                            except Exception:
                                pass

                # schedule and return immediately (message is acked by consumer)
                asyncio.create_task(_task())

            asyncio.create_task(mq.consume_requests(handle_request))
        except Exception as e:
            # Log and continue without MQ
            print(f"[startup] MQ disabled due to error: {e}")
