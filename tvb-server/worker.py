from __future__ import annotations
import asyncio
from typing import Any, Optional, Dict

from mq import MQ, publish_progress, publish_result, publish_failed
from pathlib import Path
import os
import time

# Resolve upload dir (env, same default as FastAPI app)
FILE_STORE_ROOT = Path(os.environ.get("FILE_STORE_ROOT", "/tmp/uploads"))

# ---- Video analysis (ONNX) ----
import onnxruntime as ort
from scrfd.video_infer import run_media
from scrfd.pipeline.config import (
    DET_ONNX_PATH, CLS_ONNX_PATH, CONF, FPS, CLIP_LEN, CLIP_STRIDE,
    ALIGN, LAYOUT, RGB, MEAN, STD, THRESHOLD, HIGH_CONF, SPECTRAL_R0, POSE_DELTA_OUTLIER
)

_DET_SESS: Optional[ort.InferenceSession] = None
_CLS_SESS: Optional[ort.InferenceSession] = None

def _get_onnx_sessions():
    global _DET_SESS, _CLS_SESS
    if _DET_SESS is None:
        _DET_SESS = ort.InferenceSession(DET_ONNX_PATH, providers=["CPUExecutionProvider"])  # TODO: EP via env
    if _CLS_SESS is None:
        _CLS_SESS = ort.InferenceSession(CLS_ONNX_PATH, providers=["CPUExecutionProvider"])  # TODO: EP via env
    return _DET_SESS, _CLS_SESS

# ---- Image analysis (Keras) ----


VIDEO_EXTS = {".mp4", ".mov", ".webm", ".mkv", ".avi"}
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}

async def run_analysis(mq: MQ, job_id: str, upload_id: str, params: Optional[Dict[str, Any]] = None) -> None:
    print(f"[WORKER] start jobId={job_id} uploadId={upload_id}")
    try:
        media_path = FILE_STORE_ROOT / upload_id
        if not media_path.exists():
            print(f"[WORKER] media not found: {media_path}")
            await publish_failed(mq, job_id, "media not found", reason_code="MEDIA_NOT_FOUND")
            return
        else:
            try:
                size = media_path.stat().st_size
            except Exception:
                size = -1
            print(f"[WORKER] media path confirmed: {media_path} size={size}")

        # Determine kind by extension
        ext = media_path.suffix.lower()

        # ALIGN stage
        print(f"[WORKER] progress ALIGN 10%")
        await publish_progress(mq, job_id, {"status": "RUNNING", "stage": "ALIGN", "progress": 10, "etaSec": 20})
        await asyncio.sleep(0.6)

        # Unified infer stage with ONNX (image or video)
        det_sess, cls_sess = _get_onnx_sessions()
        print(f"[WORKER] progress INFER 50% ({'image' if ext in IMAGE_EXTS else 'video'})")
        await publish_progress(mq, job_id, {"status": "RUNNING", "stage": "INFER", "progress": 50, "etaSec": 6})
        # Run inference (blocking)
        result = run_media(
            path=str(media_path),
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

        # POST stage
        print(f"[WORKER] progress POST 95%")
        await publish_progress(mq, job_id, {"status": "RUNNING", "stage": "POST", "progress": 95, "etaSec": 2})
        await asyncio.sleep(0.5)

        import json as _json
        print(f"[WORKER] publish result jobId={job_id} json=\n{_json.dumps(result, ensure_ascii=False) }")
        await publish_result(mq, job_id, result)

        # optional: delete source media after result
        try:
            media_path.unlink(missing_ok=True)
            print(f"[WORKER] deleted source {media_path}")
        except Exception:
            pass
    except Exception as e:
        print(f"[WORKER] exception jobId={job_id} error={e}")
        await publish_failed(mq, job_id, f"analysis error: {e}", reason_code="WORKER_EXCEPTION")
