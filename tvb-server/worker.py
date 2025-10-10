from __future__ import annotations

import asyncio
import base64
import io
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

from mq import MQ, publish_failed, publish_progress, publish_result
from model_registry import ModelInfo, resolve_model

# Resolve upload dir (env, same default as FastAPI app)
FILE_STORE_ROOT = Path(os.environ.get("FILE_STORE_ROOT", "/tmp/uploads"))

# ---- Video analysis (ONNX) ----
import onnxruntime as ort
from detection.gf1 import run_media
from detection.gf1.config import (
    DET_ONNX_PATH, CLS_ONNX_PATH, DET_ONNX_PROVIDERS, CLS_ONNX_PROVIDERS,
    CONF, FPS, CLIP_LEN, CLIP_STRIDE,
    ALIGN, LAYOUT, RGB, MEAN, STD, THRESHOLD, HIGH_CONF, SPECTRAL_R0, POSE_DELTA_OUTLIER,
    create_onnx_session,
)

_DET_SESS: Optional[ort.InferenceSession] = None
_CLS_SESS: Optional[ort.InferenceSession] = None

def _get_onnx_sessions():
    global _DET_SESS, _CLS_SESS
    if _DET_SESS is None:
        _DET_SESS = create_onnx_session("detector", DET_ONNX_PATH, DET_ONNX_PROVIDERS)
    if _CLS_SESS is None:
        _CLS_SESS = create_onnx_session("classifier", CLS_ONNX_PATH, CLS_ONNX_PROVIDERS)
    return _DET_SESS, _CLS_SESS

# ---- Image analysis (Keras) ----


VIDEO_EXTS = {".mp4", ".mov", ".webm", ".mkv", ".avi"}
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}

_TORCH_MODELS: Dict[str, "TorchImageRunner"] = {}


class TorchImageRunner:
    """Simple TorchScript image classifier wrapper."""

    def __init__(self, info: ModelInfo):
        import torch

        self.torch = torch
        self.info = info
        device_pref = (os.environ.get("MODEL_DEVICE") or "").strip().lower()
        if device_pref == "cpu":
            self.device = torch.device("cpu")
        elif device_pref == "cuda":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model_path = info.file_path
        if not model_path.exists():
            raise FileNotFoundError(f"model checkpoint not found: {model_path}")

        self.model = self._load_model(model_path)
        self.model.to(self.device)
        self.model.eval()

        extras = info.extras or {}
        self.image_size = int(extras.get("imageSize") or extras.get("image_size") or 224)
        mean = extras.get("mean") or [0.485, 0.456, 0.406]
        std = extras.get("std") or [0.229, 0.224, 0.225]
        self.mean = np.asarray(mean, dtype="float32").reshape(1, 1, 3)
        self.std = np.asarray(std, dtype="float32").reshape(1, 1, 3)
        self.labels: Tuple[str, ...] = tuple(info.labels) if info.labels else ("REAL", "FAKE")
        if not self.labels:
            self.labels = ("REAL", "FAKE")
        labels_upper = [str(x).upper() for x in self.labels]
        self.fake_idx = labels_upper.index("FAKE") if "FAKE" in labels_upper else (1 if len(labels_upper) > 1 else 0)

    def _load_model(self, path: Path):
        # Expect TorchScript .pt; fall back to torch.load if needed.
        try:
            return self.torch.jit.load(str(path), map_location=self.device)
        except Exception as exc:
            try:
                obj = self.torch.load(str(path), map_location=self.device)
            except Exception as inner:
                raise RuntimeError(f"Failed to load Torch model {path}: {inner}") from inner
            if hasattr(obj, "eval"):
                return obj
            raise RuntimeError(f"Unsupported checkpoint format for model {path}: {exc}")

    def predict(self, media_path: Path) -> Tuple[List[float], Image.Image]:
        image = Image.open(media_path).convert("RGB")
        if self.image_size:
            image = image.resize((self.image_size, self.image_size))
        arr = np.asarray(image, dtype="float32") / 255.0
        arr = (arr - self.mean) / self.std
        arr = np.transpose(arr, (2, 0, 1))  # CHW
        tensor = self.torch.from_numpy(arr).unsqueeze(0).to(self.device)
        with self.torch.no_grad():
            output = self.model(tensor)
            if isinstance(output, (list, tuple)):
                output = output[0]
            if output.ndim == 2:
                output = output[0]
            if output.ndim == 1:
                if output.shape[0] >= 2:
                    probs = self.torch.softmax(output, dim=0)
                else:
                    prob_fake = float(self.torch.sigmoid(output).item())
                    probs = self.torch.tensor([1.0 - prob_fake, prob_fake], device=output.device)
            else:
                probs = self.torch.softmax(output.reshape(-1), dim=0)
            probs_list = probs.detach().cpu().numpy().tolist()
        return probs_list, image


def _get_torch_runner(info: ModelInfo) -> TorchImageRunner:
    runner = _TORCH_MODELS.get(info.key)
    if runner is not None:
        return runner
    runner = TorchImageRunner(info)
    _TORCH_MODELS[info.key] = runner
    return runner


def _serialize_preview(image: Image.Image) -> Optional[str]:
    try:
        buf = io.BytesIO()
        image.save(buf, format="JPEG", quality=85)
        return base64.b64encode(buf.getvalue()).decode("ascii")
    except Exception:
        return None


def _analyze_torch_image(model: ModelInfo, media_path: Path, started_at: float) -> Dict[str, Any]:
    runner = _get_torch_runner(model)
    probs, preview = runner.predict(media_path)
    probs_arr = np.asarray(probs, dtype="float32")
    fake_idx = runner.fake_idx if runner.fake_idx < len(probs_arr) else (len(probs_arr) - 1)
    fake_prob = float(probs_arr[fake_idx]) if probs_arr.size else 0.0
    threshold = float(model.threshold or 0.5)
    max_idx = int(np.argmax(probs_arr)) if probs_arr.size else fake_idx
    label = runner.labels[max_idx] if max_idx < len(runner.labels) else ("FAKE" if fake_prob >= threshold else "REAL")
    high_conf = float(1.0 if fake_prob >= threshold else 0.0)
    q50 = float(np.quantile(probs_arr, 0.5)) if probs_arr.size else fake_prob
    pmax = float(np.max(probs_arr)) if probs_arr.size else fake_prob
    sample_b64 = _serialize_preview(preview)
    latency = time.time() - started_at
    mean_p = fake_prob
    result = {
        "ok": True,
        "variant": "torch_image",
        "model": {
            "key": model.key,
            "name": model.name,
            "version": model.version,
            "type": model.type,
        },
        "prob_fake": round(mean_p, 4),
        "prob_std": 0.0,
        "high_conf_ratio": round(high_conf, 4),
        "threshold": threshold,
        "label": label,
        "agg": {"name": "mean", "score": round(mean_p, 4)},
        "latency_sec": round(latency, 2),
        "sample_fps": None,
        "clip_len": 1,
        "clip_stride": 1,
        "aligned_cnt": 1,
        "infer_cnt": 1,
        "frames_total": 1,
        "probs_timeline": [round(mean_p, 4)],
        "quantiles": {"p50": round(q50, 4), "max": round(pmax, 4)},
        "exemplars": [{"idx": 0, "prob": round(mean_p, 4)}],
        "segments": [{"start": 0, "end": 0, "max": mean_p}] if mean_p >= threshold else [],
        "stability": {},
        "spectral": {},
        "faces": {
            "size_mean": None,
            "det_score_mean": None,
            "samples": ([{"idx": 0, "image_jpg_base64": sample_b64}] if sample_b64 else []),
        },
        "runtime": {
            "backend": "torchscript",
            "device": str(getattr(runner, "device", "cpu")),
        },
        "probabilities": probs,
    }
    return result
async def run_analysis(mq: MQ, job_id: str, upload_id: str, params: Optional[Dict[str, Any]] = None) -> None:
    print(f"[WORKER] start jobId={job_id} uploadId={upload_id}")
    params = params or {}
    if not isinstance(params, dict):
        params = {}
    started_at = time.time()
    try:
        requested_key = params.get("modelKey")
        model_info = resolve_model(requested_key)
    except Exception as exc:
        msg = f"invalid model key: {requested_key} ({exc})"
        print(f"[WORKER] {msg}")
        await publish_failed(mq, job_id, msg, reason_code="MODEL_NOT_FOUND")
        return

    model_meta = {
        "model": {
            "key": model_info.key,
            "name": model_info.name,
            "version": model_info.version,
            "type": model_info.type,
        }
    }

    try:
        media_path = FILE_STORE_ROOT / upload_id
        if not media_path.exists():
            print(f"[WORKER] media not found: {media_path}")
            await publish_failed(mq, job_id, "media not found", reason_code="MEDIA_NOT_FOUND")
            return
        try:
            size = media_path.stat().st_size
        except Exception:
            size = -1
        print(f"[WORKER] media path confirmed: {media_path} size={size}")

        ext = media_path.suffix.lower()

        if model_info.type == "torch_image":
            if ext not in IMAGE_EXTS:
                msg = f"model {model_info.key} expects image input, got {ext}"
                print(f"[WORKER] {msg}")
                await publish_failed(mq, job_id, msg, reason_code="UNSUPPORTED_MEDIA")
                return
            await publish_progress(mq, job_id, {"status": "RUNNING", "stage": "ALIGN", "progress": 12, "etaSec": 8, **model_meta})
            await asyncio.sleep(0.2)
            await publish_progress(mq, job_id, {"status": "RUNNING", "stage": "INFER", "progress": 60, "etaSec": 4, **model_meta})
            result = _analyze_torch_image(model_info, media_path, started_at)
            await publish_progress(mq, job_id, {"status": "RUNNING", "stage": "POST", "progress": 95, "etaSec": 1, **model_meta})
        else:
            det_sess, cls_sess = _get_onnx_sessions()
            await publish_progress(mq, job_id, {"status": "RUNNING", "stage": "ALIGN", "progress": 10, "etaSec": 20, **model_meta})
            await asyncio.sleep(0.6)
            print(f"[WORKER] progress INFER 50% ({'image' if ext in IMAGE_EXTS else 'video'})")
            await publish_progress(mq, job_id, {"status": "RUNNING", "stage": "INFER", "progress": 50, "etaSec": 6, **model_meta})
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
            await publish_progress(mq, job_id, {"status": "RUNNING", "stage": "POST", "progress": 95, "etaSec": 2, **model_meta})
            await asyncio.sleep(0.5)

        result = dict(result or {})
        result.setdefault("model", model_meta["model"])
        params_out = result.get("params")
        if isinstance(params_out, dict):
            params_out.setdefault("modelKey", model_info.key)
        else:
            result["params"] = {"modelKey": model_info.key}

        import json as _json
        print(f"[WORKER] publish result jobId={job_id} json=\n{_json.dumps(result, ensure_ascii=False)}")
        await publish_result(mq, job_id, result)

        try:
            media_path.unlink(missing_ok=True)
            print(f"[WORKER] deleted source {media_path}")
        except Exception:
            pass
    except Exception as e:
        print(f"[WORKER] exception jobId={job_id} error={e}")
        await publish_failed(mq, job_id, f"analysis error: {e}", reason_code="WORKER_EXCEPTION")
