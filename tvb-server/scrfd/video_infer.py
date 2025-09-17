import os, sys, time, argparse, json, cv2
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
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"failed to open video: {video_path}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    src_fps = float(cap.get(cv2.CAP_PROP_FPS)) or 30.0
    step = max(int(round(src_fps / sample_fps)), 1)

    buffer: Deque[np.ndarray] = deque(maxlen=clip_len)
    probs: List[float] = []            # per-frame/clip fake probs
    prob_series: List[float] = []      # timeline for report

    # evidence logs
    spectral_vals: List[float] = []
    lm_jitters: List[float] = []
    pose_deltas: List[Tuple[float, float, float]] = []

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
            h, w = frame.shape[:2]
            x1 = int(max(0, min(w - 1, x1)))
            y1 = int(max(0, min(h - 1, y1)))
            x2 = int(max(0, min(w - 1, x2)))
            y2 = int(max(0, min(h - 1, y2)))
            bbox_size = float(max(1.0, min(x2 - x1, y2 - y1)))

            if kps is not None:
                aligned = warp_by_5pts(frame, kps, (align_size, align_size))
            else:
                pad = int(0.12 * max(y2 - y1, x2 - x1))
                xx1 = max(0, x1 - pad); yy1 = max(0, y1 - pad)
                xx2 = min(w, x2 + pad); yy2 = min(h, y2 + pad)
                crop = frame[yy1:yy2, xx1:xx2]
                if crop.size > 0:
                    aligned = cv2.resize(crop, (align_size, align_size))

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
            if clip_len <= 1:
                iname = input_name or cls_sess.get_inputs()[0].name
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
                # default mapping: {0:'Deepfake', 1:'Real'} → fake = index 0
                fake_prob = float(pr[0, 0])
                probs.append(fake_prob)
                prob_series.append(fake_prob)
                infer_cnt += 1
                sampled += 1
                frame_idx += 1
                continue

        # -------- 4) clip-mode (for temporal models) --------
        # Only run temporal clip-mode when the model expects 5D (clip_len > 1).
        # For image models (clip_len <= 1), skip this path to avoid feeding 5D to 4D inputs.
        if clip_len > 1 and len(buffer) == clip_len and ((sampled % clip_stride) == 0):
            inp = preprocess_frames(list(buffer), size=align_size, layout=layout, rgb=rgb, mean=mean, std=std)
            iname = input_name or cls_sess.get_inputs()[0].name
            out = cls_sess.run(None, {iname: inp})
            pr = softmax(out[0], axis=-1)
            # assumption for temporal model: index 1 = fake (adjust if needed)
            fake_prob = float(pr[0, 1])
            probs.append(fake_prob)
            prob_series.append(fake_prob)
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
    label = "FAKE" if mean_p >= verdict_threshold else "REAL"

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

    return {
        "ok": True,
        "clips": int(len(probs)),
        "prob_fake": round(mean_p, 4),
        "prob_std": round(std_p, 4),
        "high_conf_ratio": round(high_conf_ratio, 4),
        "threshold": verdict_threshold,
        "label": label,
        "latency_sec": round(time.time() - t0, 2),
        "sample_fps": sample_fps,
        "clip_len": clip_len,
        "clip_stride": clip_stride,
        "aligned_cnt": aligned_cnt,
        "infer_cnt": infer_cnt,
        "frames_total": total,
        # timeline/evidence
        "probs_timeline": [round(float(x), 4) for x in prob_series],
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
        if kps is not None:
            aligned = warp_by_5pts(frame, kps, (align_size, align_size))
        else:
            pad = int(0.12 * max(y2 - y1, x2 - x1))
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

    return {
        "ok": True,
        "clips": 1,
        "prob_fake": round(mean_p, 4),
        "prob_std": round(std_p, 4),
        "high_conf_ratio": round(high_conf_ratio, 4),
        "threshold": verdict_threshold,
        "label": label,
        "latency_sec": round(time.time() - t0, 2),
        "sample_fps": None,
        "clip_len": 1,
        "clip_stride": 1,
        "aligned_cnt": 1,
        "infer_cnt": 1,
        "frames_total": 1,
        "probs_timeline": [round(float(fake_prob), 4)],
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
    det_sess = ort.InferenceSession(DET_ONNX_PATH, providers=DET_ONNX_PROVIDERS)
    cls_sess = ort.InferenceSession(CLS_ONNX_PATH, providers=CLS_ONNX_PROVIDERS)

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
