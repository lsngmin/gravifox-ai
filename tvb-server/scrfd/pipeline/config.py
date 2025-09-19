from pathlib import Path
from typing import List, Optional

# NOTE:
#   환경변수를 읽어 동적으로 값을 덮어쓰던 기존 로직을 제거했습니다.
#   필요한 값은 이 파일을 직접 수정해서 관리하세요.

ROOT_DIR = Path(__file__).resolve().parents[3]

# ---- ONNX 모델 경로 및 세션 설정 ----
DET_ONNX_PATH = str(ROOT_DIR / "tvb-server" / "scrfd" / "bnkps.onnx")
CLS_ONNX_PATH = str(ROOT_DIR / "tvb-server" / "scrfd" / "model_q4.onnx")

# onnxruntime provider 우선순위 (CUDA 사용 시 맨 앞에 배치)
DET_ONNX_PROVIDERS: List[str] = ["CUDAExecutionProvider", "CPUExecutionProvider"]
CLS_ONNX_PROVIDERS: List[str] = ["CUDAExecutionProvider", "CPUExecutionProvider"]

# 샘플 입력 경로 (테스트용)
VIDEO_PATH = str(ROOT_DIR / "sample.mp4")

# 추론 파이프라인 기본값
CONF: float = 0.6
FPS: int = 30
CLIP_LEN: int = 1
CLIP_STRIDE: int = 1
ALIGN: int = 224
LAYOUT: str = "NCTHW"
RGB: bool = True

MEAN = [0.5, 0.5, 0.5]
STD = [0.5, 0.5, 0.5]

THRESHOLD: float = 0.52
HIGH_CONF: float = 0.6
SPECTRAL_R0: float = 0.25
POSE_DELTA_OUTLIER: float = 10.0

FAKE_IDX_IMAGE: int = 0
FAKE_IDX_CLIP: int = 1

AGGREGATOR: str = "topk_mean"
TOPK_RATIO: float = 0.01
TRIM_RATIO: float = 0.2
SEG_THRESHOLD: float = THRESHOLD
SEG_MIN_LEN: int = 3

CROP_MARGIN: float = 0.30
DISABLE_ALIGN_WARP: bool = True
ATTACH_FACES: int = 3

EWMA_ALPHA: Optional[float] = None
MIN_FACE: float = 140.0
MIN_DET_SCORE: float = 0.7

TTA_FLIP: bool = True

TEMP_SCALE: float = 1.5
ONNX_OUTPUT_PROBS: bool = False
LOG_PREPROC: bool = True
LOG_MODEL_OUTPUT: bool = True


def log_session_info(label: str, path: str, requested: List[str], session: "ort.InferenceSession") -> None:
    """주요 세션 정보를 콘솔로 출력."""
    try:
        active = session.get_providers()
    except Exception as exc:  # pragma: no cover - debug helper
        active = f"error: {exc}"
    print(
        f"[ONNX][{label}] path={path} requested={requested} active={active}",
        flush=True,
    )


def create_onnx_session(label: str, path: str, providers: List[str]):
    """요청한 provider로 세션을 만들고 실패 시 CPU로 폴백."""
    import onnxruntime as ort

    requested = list(providers or []) or ["CPUExecutionProvider"]
    try:
        sess = ort.InferenceSession(path, providers=requested)
        log_session_info(label, path, requested, sess)
        return sess
    except Exception as exc:
        print(
            f"[ONNX][{label}] failed for providers={requested}: {exc}",
            flush=True,
        )
        if "CPUExecutionProvider" in requested and len(requested) == 1:
            raise
        fallback = [p for p in requested if p == "CPUExecutionProvider"]
        if not fallback:
            fallback = requested + ["CPUExecutionProvider"]
        try:
            sess = ort.InferenceSession(path, providers=fallback)
            log_session_info(f"{label}-fallback", path, fallback, sess)
            return sess
        except Exception:
            raise
