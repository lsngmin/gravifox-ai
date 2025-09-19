import argparse
import json
import math
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import onnxruntime as ort

from detection.pipeline.config import (
    DET_ONNX_PATH,
    DET_ONNX_PROVIDERS,
    CLS_ONNX_PATH,
    CLS_ONNX_PROVIDERS,
    ALIGN,
    RGB,
    MEAN,
    STD,
    FAKE_IDX_IMAGE,
    CROP_MARGIN,
    create_onnx_session,
    TEMP_SCALE,
    ONNX_OUTPUT_PROBS,
    TTA_FLIP,
    FPS,
    CLIP_LEN,
    CLIP_STRIDE,
    THRESHOLD,
    HIGH_CONF,
    SPECTRAL_R0,
    POSE_DELTA_OUTLIER,
    LAYOUT,
)
from detection.runtime import SCRFDDetector
from detection.pipeline.classifier_utils import preprocess_image
from detection.video_infer import softmax, run_media


def to_probs(out: np.ndarray) -> np.ndarray:
    arr = np.asarray(out).reshape(-1)
    if ONNX_OUTPUT_PROBS:
        probs = arr.astype(np.float32)
    else:
        logits = arr.astype(np.float32)
        if TEMP_SCALE and TEMP_SCALE != 1.0:
            logits = logits / float(TEMP_SCALE)
        probs = softmax(logits, axis=-1)
    return probs


def classify_face(sess: ort.InferenceSession, face_bgr: np.ndarray) -> float:
    inp = preprocess_image(face_bgr, size=ALIGN, rgb=RGB, mean=MEAN, std=STD)
    out = sess.run(None, {sess.get_inputs()[0].name: inp})[0]
    probs = to_probs(out)
    fake_prob = float(probs[FAKE_IDX_IMAGE])
    if TTA_FLIP:
        flipped = cv2.flip(face_bgr, 1)
        inp_flipped = preprocess_image(flipped, size=ALIGN, rgb=RGB, mean=MEAN, std=STD)
        out_flipped = sess.run(None, {sess.get_inputs()[0].name: inp_flipped})[0]
        probs_flipped = to_probs(out_flipped)
        fake_prob = 0.5 * (fake_prob + float(probs_flipped[FAKE_IDX_IMAGE]))
    return fake_prob


def sample_indices(total_frames: int, sample_count: int) -> List[int]:
    if sample_count >= total_frames:
        return list(range(total_frames))
    step = total_frames / float(sample_count)
    return [min(total_frames - 1, int(round(step * i))) for i in range(sample_count)]


def center_crop(frame: np.ndarray, frac: float) -> np.ndarray:
    h, w = frame.shape[:2]
    side = int(min(h, w) * frac)
    y1 = max(0, (h - side) // 2)
    x1 = max(0, (w - side) // 2)
    return frame[y1:y1 + side, x1:x1 + side]


from typing import Optional


def detector_crop(detector: SCRFDDetector, frame: np.ndarray, conf_th: float, margin: float) -> Optional[np.ndarray]:
    det, _ = detector.detect(frame, conf_th=conf_th, max_num=1)
    if det is None or len(det) == 0:
        return None
    x1, y1, x2, y2, _ = det[0]
    h, w = frame.shape[:2]
    pad = int(margin * max(y2 - y1, x2 - x1))
    x1 = int(max(0, min(w - 1, x1 - pad)))
    y1 = int(max(0, min(h - 1, y1 - pad)))
    x2 = int(max(0, min(w, x2 + pad)))
    y2 = int(max(0, min(h, y2 + pad)))
    crop = frame[y1:y2, x1:x2]
    return crop if crop.size else None


def aggregate_metrics(probs: List[float]) -> dict:
    if not probs:
        return {"count": 0}
    arr = np.asarray(probs, dtype=np.float32)
    topn = max(1, int(math.ceil(len(arr) * 0.1)))
    topk = np.sort(arr)[-topn:]
    return {
        "count": int(len(arr)),
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "p95": float(np.quantile(arr, 0.95)) if len(arr) > 1 else float(arr.mean()),
        "top10_mean": float(topk.mean()),
        "max": float(arr.max()),
    }


def run_oracle(args) -> Tuple[dict, dict]:
    det_sess = create_onnx_session("oracle-det", DET_ONNX_PATH, DET_ONNX_PROVIDERS)
    cls_sess = create_onnx_session("oracle-cls", CLS_ONNX_PATH, CLS_ONNX_PROVIDERS)
    detector = SCRFDDetector(session=det_sess)

    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {args.video}")
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    indices = sample_indices(total_frames, args.frames)

    center_scores: List[float] = []
    detector_scores: List[float] = []

    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok:
            continue

        # Center crop
        cc = center_crop(frame, args.center_fraction)
        if cc.size:
            center_scores.append(classify_face(cls_sess, cc))

        # Detector crop with margin
        dc = detector_crop(detector, frame, args.conf, args.margin if args.margin is not None else CROP_MARGIN)
        if dc is not None:
            detector_scores.append(classify_face(cls_sess, dc))

    cap.release()

    return aggregate_metrics(center_scores), aggregate_metrics(detector_scores)


def main():
    ap = argparse.ArgumentParser(description="Pipeline vs Oracle A/B deepfake score analysis")
    ap.add_argument("video", type=Path, help="Path to video file")
    ap.add_argument("--frames", type=int, default=64, help="Number of frames to sample")
    ap.add_argument("--center-fraction", type=float, default=0.6, help="Central crop fraction for oracle center crop")
    ap.add_argument("--margin", type=float, default=None, help="Margin multiplier for detector crop (defaults to config CROP_MARGIN)")
    ap.add_argument("--conf", type=float, default=0.55, help="Detector confidence threshold for oracle detector crop")
    ap.add_argument("--output", type=Path, default=None, help="Optional JSON output path")
    args = ap.parse_args()

    # Pipeline run
    det_sess = create_onnx_session("pipeline-det", DET_ONNX_PATH, DET_ONNX_PROVIDERS)
    cls_sess = create_onnx_session("pipeline-cls", CLS_ONNX_PATH, CLS_ONNX_PROVIDERS)
    pipeline_result = run_media(
        path=str(args.video),
        det_sess=det_sess,
        cls_sess=cls_sess,
        conf_th=args.conf,
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

    center_metrics, detector_metrics = run_oracle(args)

    report = {
        "video": str(args.video),
        "pipeline": pipeline_result,
        "oracle_center": center_metrics,
        "oracle_detector_crop": detector_metrics,
    }

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
