import os
import sys
import time
import argparse
import json
from typing import Tuple, List

import cv2
import numpy as np
import onnxruntime as ort

# 사용자가 만든 런타임 래퍼 (당신의 모듈 이름에 맞춰 유지)
from runtime import SCRFDDetector

# ---- 모델 경로 (당신 환경 그대로 사용) ----
BNKPS_MODEL_PATH = \
    "/Users/sngmin/.cache/huggingface/hub/models--ykk648--face_lib/blobs/a3562ef62592bf387f6ef19151282ac127518e51c77696e62e0661bee95ba1ad"

# ---- 5점 랜드마크 기준 좌표 (112x112 기준, InsightFace commonly used) ----
REFERENCE_FIVE_POINTS = np.array([
    [38.2946, 51.6963],  # left eye
    [73.5318, 51.5014],  # right eye
    [56.0252, 71.7366],  # nose
    [41.5493, 92.3655],  # left mouth
    [70.7299, 92.2041],  # right mouth
], dtype=np.float32)


def warp_by_5pts(img: np.ndarray, kps: np.ndarray, output_size: Tuple[int, int] = (112, 112)) -> np.ndarray:
    """
    얼굴 정렬: 5점 랜드마크(src)를 기준 랜드마크(dst)에 정렬하여 회전/스케일 보정.
    img: 원본 프레임 (H,W,3)
    kps: shape = (5, 2)
    output_size: (W, H)
    """
    if kps is None or np.asarray(kps).shape != (5, 2):
        raise ValueError("kps must be shape (5,2)")

    src = np.array(kps, dtype=np.float32)
    dst = REFERENCE_FIVE_POINTS.copy()
    dst[:, 0] *= output_size[0] / 112.0
    dst[:, 1] *= output_size[1] / 112.0

    M, _ = cv2.estimateAffinePartial2D(src, dst, method=cv2.LMEDS)
    aligned = cv2.warpAffine(img, M, output_size, borderValue=0)
    return aligned

from typing import Optional

def draw_detections(
    frame: np.ndarray,
    det: np.ndarray,
    kps: Optional[np.ndarray] = None,
    box_color: Tuple[int, int, int] = (0, 255, 0),
    kp_color: Tuple[int, int, int] = (255, 0, 0),
) -> np.ndarray:
    """안전하게 bbox/kps 시각화 (프레임 경계를 넘어가지 않도록 clip)."""
    vis = frame.copy()
    h, w = vis.shape[:2]

    if det is not None and len(det) > 0:
        for i in range(det.shape[0]):
            x1, y1, x2, y2, score = det[i]
            x1 = int(max(0, min(w - 1, x1)))
            y1 = int(max(0, min(h - 1, y1)))
            x2 = int(max(0, min(w - 1, x2)))
            y2 = int(max(0, min(h - 1, y2)))
            if x2 > x1 and y2 > y1:
                cv2.rectangle(vis, (x1, y1), (x2, y2), box_color, 2)
                cv2.putText(
                    vis,
                    f"{float(score):.2f}",
                    (x1, max(0, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    box_color,
                    1,
                )

                if kps is not None and i < len(kps):
                    pts = np.asarray(kps[i]).reshape(-1, 2)
                    for (px, py) in pts:
                        px = int(max(0, min(w - 1, px)))
                        py = int(max(0, min(h - 1, py)))
                        cv2.circle(vis, (px, py), 2, kp_color, -1)

    return vis


def uniform_sampler(total_frames: int, src_fps: float, target_fps: float) -> List[int]:
    """전체 프레임 중 균일 간격으로 index 선택."""
    if target_fps <= 0 or src_fps <= 0:
        return list(range(total_frames))
    step = max(int(round(src_fps / target_fps)), 1)
    return list(range(0, total_frames, step))


def main():
    parser = argparse.ArgumentParser(description="SCRFD ONNX face detection video test (with 5pt alignment)")
    parser.add_argument("--video", "-v", required=True, help="Input video path")
    parser.add_argument("--out", "-o", default=None, help="Output annotated video path (default: <video>_det.mp4)")
    parser.add_argument("--conf", type=float, default=0.5, help="Confidence threshold")
    parser.add_argument("--fps", type=float, default=2.0, help="Target sampling fps (reduce compute)")
    parser.add_argument("--write-every", type=int, default=1, help="Write every Nth processed frame to output")
    parser.add_argument("--max-num", type=int, default=0, help="Max faces per frame (0 = no limit)")
    parser.add_argument(
        "--codec",
        default="avc1",
        choices=["avc1", "mp4v", "MJPG"],
        help="VideoWriter codec (QuickTime 호환 위해 avc1 권장)",
    )
    parser.add_argument("--align-save", action="store_true", help="정렬된 얼굴 crop 몇 장 저장")
    parser.add_argument("--align-size", type=int, default=112, help="정렬 얼굴 출력 크기 (정사각)")
    args = parser.parse_args()

    if not os.path.exists(args.video):
        print(f"[ERR] video not found: {args.video}")
        sys.exit(1)

    # ONNX 세션 & Detector
    print(f"[INFO] Loading model: {BNKPS_MODEL_PATH}")
    sess = ort.InferenceSession(BNKPS_MODEL_PATH, providers=["CPUExecutionProvider"])  # provider 옵션은 필요시 확장
    detector = SCRFDDetector(session=sess)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"[ERR] failed to open video: {args.video}")
        sys.exit(1)

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    src_fps = float(cap.get(cv2.CAP_PROP_FPS)) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # VideoWriter 설정 (코덱 선택)
    out_path = args.out or os.path.splitext(args.video)[0] + ("_det.avi" if args.codec == "MJPG" else "_det.mp4")
    fourcc = cv2.VideoWriter_fourcc(*args.codec)
    writer = cv2.VideoWriter(out_path, fourcc, max(args.fps, 1.0), (width, height))

    idxs = uniform_sampler(total, src_fps, args.fps)
    processed = 0
    faces_total = 0
    t0 = time.time()

    print(json.dumps({
        "video": args.video,
        "frames_total": total,
        "fps_src": src_fps,
        "fps_target": args.fps,
        "output": out_path,
        "codec": args.codec
    }, ensure_ascii=False, indent=2))

    aligned_saved = 0

    for i, fidx in enumerate(idxs):
        cap.set(cv2.CAP_PROP_POS_FRAMES, fidx)
        ok, frame = cap.read()
        if not ok:
            continue

        det, kps = detector.detect(frame, conf_th=args.conf, max_num=args.max_num)

        if det is not None and len(det) > 0:
            h, w = frame.shape[:2]
            print(f"[Frame {fidx}] detected {len(det)} faces")
            for (x1, y1, x2, y2, score) in det[:3]:  # 처음 3개만 확인
                print("  bbox:", x1, y1, x2, y2, "score:", score)
                if not (0 <= x1 < x2 <= w and 0 <= y1 < y2 <= h):
                    print("  ⚠️ bbox out of range!", (x1, y1, x2, y2), "frame size:", (w, h))

        faces_total += 0 if det is None else det.shape[0]

        # --- (옵션) 정렬 얼굴 저장 ---
        if args.align_save and (kps is not None) and len(kps) > 0 and aligned_saved < 6:
            # 감지된 몇 개 얼굴만 샘플로 저장
            num_to_save = min(len(kps), 2)  # 프레임당 최대 2장만 저장
            for j in range(num_to_save):
                try:
                    aligned = warp_by_5pts(
                        frame,
                        np.asarray(kps[j]).reshape(5, 2),
                        output_size=(args.align_size, args.align_size),
                    )
                    cv2.imwrite(f"aligned_face_{fidx}_{j}.png", aligned)
                    aligned_saved += 1
                    if aligned_saved >= 6:
                        break
                except Exception as e:
                    print("[WARN] alignment failed:", e)

        # --- 시각화 & 저장 ---
        vis = draw_detections(frame, det, kps)
        if i == 0:
            cv2.imwrite("dbg_firstframe.png", vis)
        writer.write(np.ascontiguousarray(vis))

        processed += 1
        if processed % 20 == 0:
            elapsed = time.time() - t0
            print(
                f"[INFO] processed={processed}/{len(idxs)} ({processed/len(idxs)*100:.1f}%), "
                f"faces_total={faces_total}, elapsed={elapsed:.1f}s"
            )

    print("VideoWriter size:", width, height)
    print("Last frame dtype/shape will match writer size; see dbg_firstframe.png for sanity check.")

    cap.release()
    writer.release()
    elapsed = time.time() - t0

    print(json.dumps({
        "ok": True,
        "frames_processed": processed,
        "faces_total": int(faces_total),
        "elapsed_sec": round(elapsed, 2),
        "output": out_path
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
