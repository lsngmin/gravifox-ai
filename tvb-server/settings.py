"""Centralized settings for tvb-server components.

모든 런타임 설정을 여기서 직접 수정해서 사용하세요.
ENV 파일이나 외부 export 없이 동작하도록 구성했습니다.
"""
from pathlib import Path
from typing import Optional

ROOT_DIR = Path(__file__).resolve().parents[1]

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
