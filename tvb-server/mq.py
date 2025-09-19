from __future__ import annotations
import json
import ssl
from typing import Callable, Awaitable, Optional, Tuple

import aio_pika

from .settings import (
    RABBITMQ_URL,
    RABBITMQ_USE_TLS,
    RABBITMQ_VERIFY_PEER,
    RABBITMQ_CA_FILE,
    RABBITMQ_CERT_FILE,
    RABBITMQ_KEY_FILE,
    ANALYZE_EXCHANGE,
    REQUEST_QUEUE,
    RABBITMQ_PREFETCH,
)


def _should_use_tls(url: str) -> bool:
    if RABBITMQ_USE_TLS is not None:
        return bool(RABBITMQ_USE_TLS)
    return url.lower().startswith("amqps://")


def _resolve_tls(url: str) -> Tuple[bool, Optional[ssl.SSLContext]]:
    """Derive TLS usage and SSL context based on env overrides and URL scheme."""
    use_tls = _should_use_tls(url)
    if not use_tls:
        return False, None

    cafile = RABBITMQ_CA_FILE if RABBITMQ_CA_FILE else None
    context = ssl.create_default_context(cafile=cafile)

    # Optional client certs for mutual TLS
    certfile = RABBITMQ_CERT_FILE
    keyfile = RABBITMQ_KEY_FILE
    if certfile:
        context.load_cert_chain(certfile=certfile, keyfile=keyfile or None)

    if RABBITMQ_VERIFY_PEER is False:
        context.check_hostname = False
        context.verify_mode = ssl.CERT_NONE

    return True, context


def build_connect_kwargs() -> dict:
    use_tls, ssl_context = _resolve_tls(RABBITMQ_URL)
    kwargs = {}
    if use_tls:
        kwargs["ssl"] = True
        if ssl_context is not None:
            kwargs["ssl_options"] = ssl_context
    return kwargs


class MQ:
    def __init__(self) -> None:
        self._conn: Optional[aio_pika.RobustConnection] = None
        self._chan: Optional[aio_pika.RobustChannel] = None
        self._ex: Optional[aio_pika.RobustExchange] = None

    async def connect(self) -> None:
        self._conn = await aio_pika.connect_robust(RABBITMQ_URL, **build_connect_kwargs())
        self._chan = await self._conn.channel()
        await self._chan.set_qos(prefetch_count=RABBITMQ_PREFETCH)
        self._ex = await self._chan.declare_exchange(ANALYZE_EXCHANGE, aio_pika.ExchangeType.TOPIC, durable=True)

    async def close(self) -> None:
        if self._conn:
            await self._conn.close()

    async def publish_json(self, routing_key: str, payload: dict) -> None:
        assert self._ex is not None
        body = json.dumps(payload).encode("utf-8")
        msg = aio_pika.Message(body=body, content_type="application/json", delivery_mode=aio_pika.DeliveryMode.PERSISTENT)
        # debug print
        try:
            print(f"[MQ] publish rk={routing_key} bytes={len(body)}")
        except Exception:
            pass
        await self._ex.publish(msg, routing_key=routing_key)

    async def consume_requests(self, handler: Callable[[dict], Awaitable[None]]) -> None:
        assert self._chan is not None and self._ex is not None
        queue = await self._chan.declare_queue(REQUEST_QUEUE, durable=True)
        await queue.bind(self._ex, routing_key="analyze.request")

        async with queue.iterator() as qiter:
            async for message in qiter:
                async with message.process():
                    try:
                        payload = json.loads(message.body.decode("utf-8"))
                    except Exception:
                        continue
                    try:
                        print(f"[MQ] consume analyze.request bytes={len(message.body)}")
                    except Exception:
                        pass
                    await handler(payload)


async def publish_progress(mq: MQ, job_id: str, data: dict) -> None:
    await mq.publish_json(f"analyze.progress.{job_id}", {"jobId": job_id, **data})


async def publish_result(mq: MQ, job_id: str, result: dict) -> None:
    await mq.publish_json(f"analyze.result.{job_id}", {"jobId": job_id, "result": result})


async def publish_failed(mq: MQ, job_id: str, reason: str, reason_code: Optional[str] = None) -> None:
    payload = {"jobId": job_id, "status": "FAILED", "reason": reason}
    if reason_code:
        payload["reasonCode"] = reason_code
    await mq.publish_json(f"analyze.failed.{job_id}", payload)
