"""Centralized settings for tvb-server components.

모든 런타임 설정을 여기서 직접 수정해서 사용하세요.
ENV 파일이나 외부 export 없이 동작하도록 구성했습니다.
"""
from pathlib import Path
from typing import Optional

ROOT_DIR = Path(__file__).resolve().parents[1]

# ---- RabbitMQ / MQ settings ----
RABBITMQ_URL: str = (
    "amqps://gravifox:!Tmdals017217@"
    "b-d17504b0-9503-4739-950c-4f60999488bc.mq.us-east-2.on.aws:5671"
)

# TLS 사용 여부(None이면 URL scheme에 따라 자동 결정)
RABBITMQ_USE_TLS: Optional[bool] = None
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

