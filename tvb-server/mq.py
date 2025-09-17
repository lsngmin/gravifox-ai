from __future__ import annotations
import json
import os
from typing import Callable, Awaitable, Optional

import aio_pika


RABBITMQ_URL = os.environ.get("RABBITMQ_URL", "amqp://admin:admin@localhost:5672/")
EXCHANGE_NAME = os.environ.get("ANALYZE_EXCHANGE", "analyze.exchange")
REQUEST_QUEUE = os.environ.get("REQUEST_QUEUE", "analyze.request.fastapi")


class MQ:
    def __init__(self) -> None:
        self._conn: Optional[aio_pika.RobustConnection] = None
        self._chan: Optional[aio_pika.RobustChannel] = None
        self._ex: Optional[aio_pika.RobustExchange] = None

    async def connect(self) -> None:
        self._conn = await aio_pika.connect_robust(RABBITMQ_URL)
        self._chan = await self._conn.channel()
        await self._chan.set_qos(prefetch_count=10)
        self._ex = await self._chan.declare_exchange(EXCHANGE_NAME, aio_pika.ExchangeType.TOPIC, durable=True)

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
