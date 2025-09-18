import os, sys, time, argparse, json, cv2, base64
import numpy as np
import onnxruntime as ort

from collections import deque
from typing import Deque, List, Tuple, Optional
from .runtime import SCRFDDetector

from .pipeline.detector_utils import warp_by_5pts, estimate_pose_5pts, lm_jitter, spectral_highfreq_ratio
from .pipeline.classifier_utils import softmax, preprocess_image, preprocess_frames
from .pipeline.config import VIDEO_PATH, DEFAULT_DETECTOR_ONNX
from pathlib import Path

def run_video(
    video_path: str,
    det_sess: ort.InferenceSession,
    cls_sess: ort.InferenceSession,
    conf_th: float = 0.5,
    sample_fps: float = 2.0,
    clip_len: int = 16,
    clip_stride: int = 16,
    align_size: int = 112,
    layout: str = "NCTHW",
    rgb: bool = True,
    mean: Optional[List[float]] = None,
    std: Optional[List[float]] = None,
    input_name: Optional[str] = None,
    verdict_threshold: float = 0.6,
    high_conf_threshold: float = 0.8,
    spectral_r0: float = 0.25,
    pose_delta_outlier: float = 10.0,
) -> dict:
    """Detect → align (in-memory) → frame or clip classify → aggregate & evidence.
    If clip_len <= 1, runs per-frame classification (image ONNX)."""
    detector = SCRFDDetector(session=det_sess)

    # Optional Torch backend for classifier
    torch_runner = None
    try:
        from .pipeline.config import (
            CLASSIFIER_BACKEND, TORCH_EXTRACTOR_CKPT, TORCH_MODEL_CKPT, TORCH_DEVICE,
            FAKE_IDX_IMAGE as _FAKE_IDX_IMAGE_CFG,
            FAKE_IDX_CLIP as _FAKE_IDX_CLIP_CFG,
        )
        if CLASSIFIER_BACKEND == 'torch':
            try:
                from mintime_runner import TorchClassifierRunner
                from pathlib import Path as _P
                _root = _P(__file__).resolve().parents[1]  # tvb-server root
                ext_ckpt = TORCH_EXTRACTOR_CKPT or str(_root / 'MINTIME_XC_Extractor_checkpoint30')
                cls_ckpt = TORCH_MODEL_CKPT or str(_root / 'MINTIME_XC_Model_checkpoint30')
                torch_runner = TorchClassifierRunner(
                    extractor_ckpt=ext_ckpt,
                    model_ckpt=cls_ckpt,
                    device=TORCH_DEVICE,
                    rgb=rgb,
                    mean=mean,
                    std=std,
                )
            except Exception as _e:  # fallback to ONNX
                print(f"[TorchBackend] disabled due to error: {_e}")
                torch_runner = None
    except Exception:
        torch_runner = None

    # If torch backend is requested but torch runner failed and no ONNX classifier session provided,
    # build an ONNX session as fallback to avoid NoneType errors.
    if torch_runner is None and cls_sess is None:
        try:
            from .pipeline.config import CLS_ONNX_PATH, CLS_ONNX_PROVIDERS, create_onnx_session as _mk
            cls_sess = _mk("classifier-fallback", CLS_ONNX_PATH, CLS_ONNX_PROVIDERS)
        except Exception as _e:
            print(f"[ONNX][classifier] fallback create failed: {_e}")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"failed to open video: {video_path}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    src_fps = float(cap.get(cv2.CAP_PROP_FPS)) or 30.0
    step = max(int(round(src_fps / sample_fps)), 1)

    buffer: Deque[np.ndarray] = deque(maxlen=clip_len)
    probs: List[float] = []            # per-frame/clip fake probs
    prob_series: List[float] = []      # timeline for report
    used_faces: List[np.ndarray] = []  # faces corresponding to each prob

    # evidence logs
    spectral_vals: List[float] = []
    lm_jitters: List[float] = []
    pose_deltas: List[Tuple[float, float, float]] = []
    det_scores: List[float] = []
    face_sizes: List[float] = []

    prev_kps: Optional[np.ndarray] = None
    prev_pose: Optional[Tuple[float, float, float]] = None

    frame_idx = 0
    sampled = 0
    t0 = time.time()

    # debug counters
    aligned_cnt = 0
    infer_cnt = 0

    # detection up-scale to improve recall on small faces
    scale_up = 1.25  # set to 1.0 to disable

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_idx % step != 0:
            frame_idx += 1
            continue

        # -------- 1) detect with optional up-scale --------
        det_frame = frame
        if scale_up != 1.0:
            det_frame = cv2.resize(frame, None, fx=scale_up, fy=scale_up)

        det, kps = detector.detect(det_frame, conf_th=conf_th, max_num=1)

        # map back to original scale
        if det is not None and len(det) > 0:
            if scale_up != 1.0:
                det = det.copy()
                det[:, :4] = det[:, :4] / scale_up
            if kps is not None and len(kps) > 0:
                kps = (np.asarray(kps)[0] / scale_up).reshape(5, 2)
            else:
                kps = None

        aligned: Optional[np.ndarray] = None
        bbox_size = None

        # -------- 2) build aligned face (kps → warp; else bbox crop) --------
        if det is not None and len(det) > 0:
            x1, y1, x2, y2, _ = det[0]
            try:
                det_scores.append(float(det[0,4]))
            except Exception:
                pass
            h, w = frame.shape[:2]
            x1 = int(max(0, min(w - 1, x1)))
            y1 = int(max(0, min(h - 1, y1)))
            x2 = int(max(0, min(w - 1, x2)))
            y2 = int(max(0, min(h - 1, y2)))
            bbox_size = float(max(1.0, min(x2 - x1, y2 - y1)))

            from .pipeline.config import CROP_MARGIN as _CROP_MARGIN, DISABLE_ALIGN_WARP as _DISABLE_WARP
            if kps is not None and not _DISABLE_WARP:
                aligned = warp_by_5pts(frame, kps, (align_size, align_size))
            else:
                pad = int(_CROP_MARGIN * max(y2 - y1, x2 - x1))
                xx1 = max(0, x1 - pad); yy1 = max(0, y1 - pad)
                xx2 = min(w, x2 + pad); yy2 = min(h, y2 + pad)
                crop = frame[yy1:yy2, xx1:xx2]
                if crop.size > 0:
                    aligned = cv2.resize(crop, (align_size, align_size))
            if bbox_size is not None:
                face_sizes.append(float(bbox_size))

        # -------- 2.5) evidence hooks (only when we have kps/aligned) --------
        if aligned is not None:
            aligned_cnt += 1
            buffer.append(aligned)

            # spectral
            spectral_vals.append(spectral_highfreq_ratio(aligned, r0_ratio=spectral_r0))

            # jitter (needs prev_kps)
            if kps is not None and prev_kps is not None and bbox_size is not None:
                val = lm_jitter(prev_kps, kps, bbox_size)
                if val is not None:
                    lm_jitters.append(val)

            # pose delta
            if kps is not None:
                pose = estimate_pose_5pts(kps, frame.shape)
                if pose is not None and prev_pose is not None:
                    d = tuple(abs(np.array(pose) - np.array(prev_pose)).tolist())
                    pose_deltas.append(d)
                prev_pose = pose

            prev_kps = kps if kps is not None else prev_kps

            # -------- 3) frame-mode (image ONNX): immediate infer --------
            # If Torch is not available, force frame-mode even when clip_len>1 (ONNX expects 4D input)
            if clip_len <= 1 or torch_runner is None:
                iname = input_name or cls_sess.get_inputs()[0].name
                if torch_runner is not None:
                    _, probs_all = torch_runner.infer_clip([aligned], size=align_size, layout=layout, rgb=rgb, mean=mean, std=std)
                    fake_prob = float(probs_all[_FAKE_IDX_IMAGE_CFG]) if len(probs_all) > _FAKE_IDX_IMAGE_CFG else float(probs_all[0])
                else:
                    inp = preprocess_image(
                        aligned,
                        size=align_size,
                        rgb=rgb,
                        mean=mean,
                        std=std,
                    )  # (1,3,H,W)
                    if inp.ndim == 5:
                        N, T, C, H, W = inp.shape
                        inp = inp.reshape(N * T, C, H, W)
                    logits = cls_sess.run(None, {iname: inp})[0]  # (1,2)
                    pr = softmax(logits, axis=-1)
                    fake_prob = float(pr[0, _FAKE_IDX_IMAGE_CFG])
                probs.append(fake_prob)
                prob_series.append(fake_prob)
                used_faces.append(aligned.copy())
                infer_cnt += 1
                sampled += 1
                frame_idx += 1
                continue

        # -------- 4) clip-mode (for temporal models) --------
        # Only run temporal clip-mode when the model expects 5D (clip_len > 1).
        # For image models (clip_len <= 1), skip this path to avoid feeding 5D to 4D inputs.
        if clip_len > 1 and torch_runner is not None and len(buffer) == clip_len and ((sampled % clip_stride) == 0):
            if torch_runner is not None:
                _, probs_all = torch_runner.infer_clip(list(buffer), size=align_size, layout=layout, rgb=rgb, mean=mean, std=std)
                fake_prob = float(probs_all[_FAKE_IDX_CLIP_CFG]) if len(probs_all) > _FAKE_IDX_CLIP_CFG else float(probs_all[0])
            else:
                inp = preprocess_frames(list(buffer), size=align_size, layout=layout, rgb=rgb, mean=mean, std=std)
                iname = input_name or cls_sess.get_inputs()[0].name
                out = cls_sess.run(None, {iname: inp})
                pr = softmax(out[0], axis=-1)
                fake_prob = float(pr[0, _FAKE_IDX_CLIP_CFG])
            probs.append(fake_prob)
            prob_series.append(fake_prob)
            try:
                center = list(buffer)[min(len(buffer)//2, len(buffer)-1)]
                used_faces.append(center.copy())
            except Exception:
                pass
            infer_cnt += 1

        sampled += 1
        frame_idx += 1

    cap.release()

    # ---- last resort: if frame-mode and nothing inferred, use last aligned ----
    if clip_len <= 1 and len(probs) == 0 and len(buffer) > 0:
        iname = input_name or cls_sess.get_inputs()[0].name
        inp = preprocess_image(buffer[-1], size=align_size, rgb=rgb, mean=mean, std=std)
        logits = cls_sess.run(None, {iname: inp})[0]
        pr = softmax(logits, axis=-1)
        probs.append(float(pr[0, 0]))
        prob_series.append(float(pr[0, 0]))
        infer_cnt += 1

    # aggregate
    if len(probs) == 0:
        return {
            "ok": False,
            "msg": "Not enough aligned frames to classify.",
            "clips": 0,
            "aligned_cnt": aligned_cnt,
            "infer_cnt": infer_cnt,
            "frames_total": total,
        }

    probs_arr = np.asarray(probs, dtype=np.float32)
    mean_p = float(np.mean(probs_arr))
    std_p = float(np.std(probs_arr))
    high_conf_ratio = float(np.mean(probs_arr >= high_conf_threshold))

    # Aggregation options
    from .pipeline.config import AGGREGATOR, TOPK_RATIO, SEG_THRESHOLD, SEG_MIN_LEN
    agg_name = AGGREGATOR.lower()
    agg_score = mean_p
    if agg_name == 'median':
        agg_score = float(np.median(probs_arr))
    elif agg_name == 'topk_mean':
        k = max(1, int(round(len(probs_arr) * TOPK_RATIO)))
        topk = np.sort(probs_arr)[-k:]
        agg_score = float(np.mean(topk))
    elif agg_name == 'p95':
        agg_score = float(np.quantile(probs_arr, 0.95))
    elif agg_name == 'hybrid':
        k = max(1, int(round(len(probs_arr) * TOPK_RATIO)))
        topk = np.sort(probs_arr)[-k:]
        agg_score = float(0.5 * mean_p + 0.5 * float(np.mean(topk)))

    label = "FAKE" if agg_score >= verdict_threshold else "REAL"

    # stability metrics
    pose_arr = np.asarray(pose_deltas, dtype=np.float32) if len(pose_deltas) else None
    pose_delta_mean = (
        dict(zip(["yaw", "pitch", "roll"], np.mean(pose_arr, axis=0).tolist())) if pose_arr is not None else None
    )
    pose_delta_outlier_ratio = (
        float(np.mean(np.linalg.norm(pose_arr, axis=1) > pose_delta_outlier)) if pose_arr is not None else None
    )

    lm_jitter_rms = (
        float(np.sqrt(np.mean(np.square(np.asarray([v for v in lm_jitters if v is not None], dtype=np.float32)))))
        if len(lm_jitters) else None
    )

    spectral_mean = float(np.mean(spectral_vals)) if len(spectral_vals) else None
    spectral_outlier_ratio = (
        float(np.mean(np.asarray(spectral_vals) > 0.4)) if len(spectral_vals) else None
    )

    # suspicious segments (consecutive samples >= SEG_THRESHOLD)
    segments = []
    if len(prob_series):
        curr_s = None
        for i, p in enumerate(prob_series):
            if p >= SEG_THRESHOLD:
                if curr_s is None:
                    curr_s = {"start": i, "max": p}
                else:
                    curr_s["max"] = max(curr_s["max"], p)
            else:
                if curr_s is not None:
                    curr_s["end"] = i - 1
                    if curr_s["end"] - curr_s["start"] + 1 >= SEG_MIN_LEN:
                        segments.append(curr_s)
                    curr_s = None
        if curr_s is not None:
            curr_s["end"] = len(prob_series) - 1
            if curr_s["end"] - curr_s["start"] + 1 >= SEG_MIN_LEN:
                segments.append(curr_s)

    # quantiles and exemplars
    q50 = float(np.quantile(probs_arr, 0.5))
    q75 = float(np.quantile(probs_arr, 0.75))
    q90 = float(np.quantile(probs_arr, 0.9))
    q95 = float(np.quantile(probs_arr, 0.95))
    top_n = min(10, len(prob_series))
    top_idx = np.argsort(probs_arr)[-top_n:][::-1].tolist()
    exemplars = [{"idx": int(i), "prob": float(probs_arr[i])} for i in top_idx]

    # attach face thumbnails (base64)
    from .pipeline.config import ATTACH_FACES as _ATTACH_FACES
    samples_b64 = []
    for i in top_idx[:max(0, _ATTACH_FACES)]:
        if 0 <= i < len(used_faces):
            try:
                ok, enc = cv2.imencode('.jpg', used_faces[i], [int(cv2.IMWRITE_JPEG_QUALITY), 85])
                if ok:
                    samples_b64.append({
                        "idx": int(i),
                        "image_jpg_base64": base64.b64encode(enc.tobytes()).decode('ascii')
                    })
            except Exception:
                pass

    # timing conversion for segments (seconds)
    if sample_fps and sample_fps > 0:
        for seg in segments:
            seg["start_sec"] = round(seg["start"] / sample_fps, 2)
            seg["end_sec"] = round(seg["end"] / sample_fps, 2)

    return {
        "ok": True,
        "clips": int(len(probs)),
        "prob_fake": round(mean_p, 4),
        "prob_std": round(std_p, 4),
        "high_conf_ratio": round(high_conf_ratio, 4),
        "threshold": verdict_threshold,
        "label": label,
        "agg": {"name": agg_name, "score": round(agg_score, 4)},
        "latency_sec": round(time.time() - t0, 2),
        "sample_fps": sample_fps,
        "clip_len": clip_len,
        "clip_stride": clip_stride,
        "aligned_cnt": aligned_cnt,
        "infer_cnt": infer_cnt,
        "frames_total": total,
        # timeline/evidence
        "probs_timeline": [round(float(x), 4) for x in prob_series],
        "quantiles": {"p50": round(q50, 4), "p75": round(q75, 4), "p90": round(q90, 4), "p95": round(q95, 4), "max": round(float(np.max(probs_arr)), 4)},
        "exemplars": exemplars,
        "segments": segments,
        "stability": {
            "lm_jitter_rms": lm_jitter_rms,
            "pose_delta_mean": pose_delta_mean,
            "pose_delta_outlier_ratio": pose_delta_outlier_ratio,
        },
        "spectral": {
            "highfreq_ratio_mean": spectral_mean,
            "outlier_ratio": spectral_outlier_ratio,
            "r0_ratio": spectral_r0,
        },
        "faces": {
            "size_mean": (float(np.mean(face_sizes)) if len(face_sizes) else None),
            "size_min": (float(np.min(face_sizes)) if len(face_sizes) else None),
            "size_max": (float(np.max(face_sizes)) if len(face_sizes) else None),
            "det_score_mean": (float(np.mean(det_scores)) if len(det_scores) else None),
            "samples": samples_b64,
        },
        "runtime": {
            "det_providers": getattr(det_sess, 'get_providers', lambda: [])(),
            "cls_providers": getattr(cls_sess, 'get_providers', lambda: [])(),
        }
    }

def run_image(
    image_path: str,
    det_sess: ort.InferenceSession,
    cls_sess: ort.InferenceSession,
    align_size: int = 112,
    layout: str = "NCTHW",  # unused for single frame
    rgb: bool = True,
    mean: Optional[List[float]] = None,
    std: Optional[List[float]] = None,
    input_name: Optional[str] = None,
    verdict_threshold: float = 0.6,
    high_conf_threshold: float = 0.8,
    spectral_r0: float = 0.25,
    pose_delta_outlier: float = 10.0,
    conf_th: float = 0.5,
    **kwargs,
) -> dict:
    """Single image flow that mirrors frame-mode of run_video and returns the same schema."""
    detector = SCRFDDetector(session=det_sess)
    frame = cv2.imread(image_path)
    if frame is None or frame.size == 0:
        raise RuntimeError(f"failed to open image: {image_path}")

    t0 = time.time()

    # up-scale detection for small faces
    scale_up = 1.25
    det_frame = frame
    if scale_up != 1.0:
        det_frame = cv2.resize(frame, None, fx=scale_up, fy=scale_up)

    det, kps = detector.detect(det_frame, conf_th=conf_th, max_num=1)
    if det is not None and len(det) > 0:
        if scale_up != 1.0:
            det = det.copy()
            det[:, :4] = det[:, :4] / scale_up
        if kps is not None and len(kps) > 0:
            kps = (np.asarray(kps)[0] / scale_up).reshape(5, 2)
        else:
            kps = None
    aligned = None
    bbox_size = None
    if det is not None and len(det) > 0:
        x1, y1, x2, y2, _ = det[0]
        h, w = frame.shape[:2]
        x1 = int(max(0, min(w - 1, x1)))
        y1 = int(max(0, min(h - 1, y1)))
        x2 = int(max(0, min(w - 1, x2)))
        y2 = int(max(0, min(h - 1, y2)))
        bbox_size = float(max(1.0, min(x2 - x1, y2 - y1)))
        from .pipeline.config import CROP_MARGIN as _CROP_MARGIN, DISABLE_ALIGN_WARP as _DISABLE_WARP
        if kps is not None and not _DISABLE_WARP:
            aligned = warp_by_5pts(frame, kps, (align_size, align_size))
        else:
            pad = int(_CROP_MARGIN * max(y2 - y1, x2 - x1))
            xx1 = max(0, x1 - pad); yy1 = max(0, y1 - pad)
            xx2 = min(w, x2 + pad); yy2 = min(h, y2 + pad)
            crop = frame[yy1:yy2, xx1:xx2]
            if crop.size > 0:
                aligned = cv2.resize(crop, (align_size, align_size))

    spectral_vals: List[float] = []
    lm_jitters: List[float] = []
    pose_deltas: List[Tuple[float, float, float]] = []
    lm_jitter_rms = None
    pose_delta_mean = None
    pose_delta_outlier_ratio = None

    if aligned is not None:
        spectral_vals.append(spectral_highfreq_ratio(aligned, r0_ratio=spectral_r0))
        # pose/jitter metrics require keypoints pairs; for single image we skip or set None

        # classify single aligned face
        iname = input_name or cls_sess.get_inputs()[0].name
        inp = preprocess_image(aligned, size=align_size, rgb=rgb, mean=mean, std=std)
        logits = cls_sess.run(None, {iname: inp})[0]
        pr = softmax(logits, axis=-1)
        fake_prob = float(pr[0, 0])  # image model: index 0 = fake assumption
    else:
        return {
            "ok": False,
            "msg": "No face/aligned region detected.",
            "clips": 0,
            "aligned_cnt": 0,
            "infer_cnt": 0,
            "frames_total": 1,
        }

    probs_arr = np.asarray([fake_prob], dtype=np.float32)
    mean_p = float(np.mean(probs_arr))
    std_p = float(np.std(probs_arr))
    high_conf_ratio = float(np.mean(probs_arr >= high_conf_threshold))
    label = "FAKE" if mean_p >= verdict_threshold else "REAL"
    spectral_mean = float(np.mean(spectral_vals)) if len(spectral_vals) else None
    spectral_outlier_ratio = (
        float(np.mean(np.asarray(spectral_vals) > 0.4)) if len(spectral_vals) else None
    )

    # simple quantiles/exemplars for single frame
    q50 = float(np.quantile(probs_arr, 0.5))
    exemplars = [{"idx": 0, "prob": float(fake_prob)}]

    # attach single-face sample (base64)
    sample_b64 = None
    try:
        ok, enc = cv2.imencode('.jpg', aligned, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        if ok:
            sample_b64 = base64.b64encode(enc.tobytes()).decode('ascii')
    except Exception:
        pass

    return {
        "ok": True,
        "clips": 1,
        "prob_fake": round(mean_p, 4),
        "prob_std": round(std_p, 4),
        "high_conf_ratio": round(high_conf_ratio, 4),
        "threshold": verdict_threshold,
        "label": label,
        "agg": {"name": "mean", "score": round(mean_p, 4)},
        "latency_sec": round(time.time() - t0, 2),
        "sample_fps": None,
        "clip_len": 1,
        "clip_stride": 1,
        "aligned_cnt": 1,
        "infer_cnt": 1,
        "frames_total": 1,
        "probs_timeline": [round(float(fake_prob), 4)],
        "quantiles": {"p50": round(q50, 4), "max": round(float(fake_prob), 4)},
        "exemplars": exemplars,
        "segments": [{"start": 0, "end": 0, "max": float(fake_prob)}] if fake_prob >= verdict_threshold else [],
        "stability": {
            "lm_jitter_rms": lm_jitter_rms,
            "pose_delta_mean": pose_delta_mean,
            "pose_delta_outlier_ratio": pose_delta_outlier_ratio,
        },
        "spectral": {
            "highfreq_ratio_mean": spectral_mean,
            "outlier_ratio": spectral_outlier_ratio,
            "r0_ratio": spectral_r0,
        },
        "faces": {
            "size_mean": (float(bbox_size) if bbox_size is not None else None),
            "det_score_mean": (float(det[0,4]) if det is not None and len(det)>0 else None),
            "samples": ([{"idx": 0, "image_jpg_base64": sample_b64}] if sample_b64 else []),
        },
        "runtime": {
            "det_providers": getattr(det_sess, 'get_providers', lambda: [])(),
            "cls_providers": getattr(cls_sess, 'get_providers', lambda: [])(),
        }
    }

def run_media(
    path: str,
    det_sess: ort.InferenceSession,
    cls_sess: ort.InferenceSession,
    **kwargs,
) -> dict:
    ext = Path(path).suffix.lower()
    if ext in {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}:
        return run_image(path, det_sess, cls_sess, **kwargs)
    # default as video
    return run_video(path, det_sess, cls_sess, **kwargs)

from .pipeline.config import *

def main():
    # sessions
    det_sess = create_onnx_session("detector", DET_ONNX_PATH, DET_ONNX_PROVIDERS)
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

if __name__ == "__main__":
    main()
