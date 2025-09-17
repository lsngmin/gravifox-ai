import os
from pathlib import Path
from typing import List


def _load_env_file():
    """
    활성화된 환경(LOCAL/PROD 등)에 맞춰 .env.{env} 파일을 읽어 환경 변수를 설정한다.
    우선순위: TVB_ENV_FILE > TVB_ENV > 기본(local).
    """

    root_dir = Path(__file__).resolve().parents[3]

    # 1) 명시적으로 파일 경로가 주어졌다면 그 파일을 사용
    explicit_path = os.environ.get("TVB_ENV_FILE")
    if explicit_path:
        env_paths = [Path(explicit_path)]
    else:
        # 2) 환경 이름 기반 파일 선택 (spring profile 유사)
        profile = os.environ.get("TVB_ENV", "local").lower()
        env_paths = [
            root_dir / f".env.{profile}",
            root_dir / ".env",  # fallback 공통 설정
        ]

    for env_path in env_paths:
        if not env_path.exists():
            continue
        try:
            for raw in env_path.read_text().splitlines():
                line = raw.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" not in line:
                    continue
                key, value = line.split("=", 1)
                key = key.strip()
                if not key:
                    continue
                value = value.strip().strip('"').strip("'")
                os.environ.setdefault(key, value)
        except Exception:
            pass


_load_env_file()


def _resolve_model_path(env_key: str, default: str) -> str:
    """환경 변수 우선, 없으면 기본 경로 사용."""
    candidate = os.environ.get(env_key, default)
    return str(Path(candidate).expanduser())


def _resolve_providers(specific_key: str, fallback_key: str, default: str) -> List[str]:
    """쉼표 구분 provider 문자열을 리스트로 변환."""
    raw = os.environ.get(specific_key) or os.environ.get(fallback_key) or default
    return [token.strip() for token in raw.split(',') if token.strip()]


# ---- 얼굴 검출/판별 모델 경로 ----
DEFAULT_DETECTOR_ONNX = \
    "/Users/sngmin/.cache/huggingface/hub/models--ykk648--face_lib/blobs/a3562ef62592bf387f6ef19151282ac127518e51c77696e62e0661bee95ba1ad"
DEFAULT_CLASSIFIER_ONNX = \
    "/Users/sngmin/.cache/huggingface/hub/models--prithivMLmods--Deepfake-Detection-Exp-02-22-ONNX/blobs/5b871f08a20f4543be3cec99eac74165821b6dc8f1b447c92391868e5d4f37b6"

DET_ONNX_PATH = _resolve_model_path("TVB_DETECTOR_ONNX", DEFAULT_DETECTOR_ONNX)
CLS_ONNX_PATH = _resolve_model_path("TVB_CLASSIFIER_ONNX", DEFAULT_CLASSIFIER_ONNX)

DET_ONNX_PROVIDERS = _resolve_providers(
    "TVB_DETECTOR_PROVIDERS", "TVB_ONNX_PROVIDERS", "CPUExecutionProvider"
)
CLS_ONNX_PROVIDERS = _resolve_providers(
    "TVB_CLASSIFIER_PROVIDERS", "TVB_ONNX_PROVIDERS", "CPUExecutionProvider"
)


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


VIDEO_PATH = os.environ.get('TVB_SAMPLE_VIDEO', '/Users/sngmin/gravifox/tvb-ai/sample.mp4')
CONF = 0.35
FPS = 30
CLIP_LEN = 1
CLIP_STRIDE = 1
ALIGN = 224
LAYOUT = 'NCTHW'
RGB= True
MEAN=[0.485, 0.456, 0.406]
STD=[0.229, 0.224, 0.225]
THRESHOLD=0.6
HIGH_CONF=0.8
SPECTRAL_R0=0.25
POSE_DELTA_OUTLIER=10
