"""Centralized settings for tvb-server components.

모든 런타임 설정을 여기서 직접 수정해서 사용하세요.
.env 파일을 통해서도 값을 덮어쓸 수 있도록 지원합니다.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

ROOT_DIR = Path(__file__).resolve().parents[1]
MODEL_CATALOG_PATH: str = str((ROOT_DIR / "tvb-server" / "models" / "catalog.json").resolve())


def _load_dotenv() -> Dict[str, str]:
    env_path = ROOT_DIR / ".env"
    if not env_path.is_file():
        return {}

    values: Dict[str, str] = {}
    for line in env_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        values[key.strip()] = value.strip()
    return values


_ENV_OVERRIDES = _load_dotenv()


def _env(name: str, default: Optional[str] = None) -> Optional[str]:
    """간단한 .env 기반 설정 조회."""

    return _ENV_OVERRIDES.get(name, default)

# ---- RabbitMQ / MQ settings ----
# 로컬 AI 서버의 MQ로 전환 (포트 5000, TLS 미사용)
# 필요 시 사용자/비밀번호를 포함한 형태로 교체하세요:
#   amqp://<user>:<pass>@117.17.149.66:5000
RABBITMQ_URL: str = "amqp://gravifox:!Tmdals017217@117.17.149.66:5000/"

# TLS 사용 여부(None이면 URL scheme에 따라 자동 결정)
RABBITMQ_USE_TLS: Optional[bool] = None  # amqp:// 이므로 자동으로 비TLS 적용
RABBITMQ_VERIFY_PEER: Optional[bool] = True
RABBITMQ_CA_FILE: Optional[str] = None
RABBITMQ_CERT_FILE: Optional[str] = None
RABBITMQ_KEY_FILE: Optional[str] = None

ANALYZE_EXCHANGE: str = "analyze.exchange"
REQUEST_QUEUE: str = "analyze.request.fastapi"
RABBITMQ_PREFETCH: int = 10

# MQ 소비자 실행 여부 및 동시 처리 개수
ENABLE_MQ: bool = True
TVB_MAX_CONCURRENCY: int = 1

# ---- Vision Transformer 추론 설정 ----
# 기본적으로 최신 vit_residual_fusion 실험을 따라가며,
# 필요하면 .env에서 아래 키로 덮어쓰세요.
_raw_run_dir = _env("TVB_VIT_RUN_DIR")
VIT_RUN_DIR: Optional[Path] = Path(_raw_run_dir).expanduser().resolve() if _raw_run_dir else None

_raw_run_root = _env("TVB_VIT_RUN_ROOT")
_default_run_root = ROOT_DIR / "experiments" / "vit_residual_fusion"
VIT_RUN_ROOT: Path = Path(_raw_run_root).expanduser().resolve() if _raw_run_root else _default_run_root

VIT_CHECKPOINT_NAME: str = _env("TVB_VIT_CHECKPOINT", "best.pt") or "best.pt"
VIT_DEVICE_NAME: str = (_env("TVB_VIT_DEVICE", "auto") or "auto").lower()
