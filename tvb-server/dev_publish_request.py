import asyncio
import json
import aio_pika

try:
    from .settings import RABBITMQ_URL, ANALYZE_EXCHANGE
    from .mq import build_connect_kwargs
except ImportError:
    # Allow execution as a standalone script (python tvb-server/dev_publish_request.py ...)
    import sys
    from pathlib import Path

    ROOT = Path(__file__).resolve().parent
    if str(ROOT) not in sys.path:
        sys.path.append(str(ROOT))

    from settings import RABBITMQ_URL, ANALYZE_EXCHANGE  # type: ignore
    from mq import build_connect_kwargs  # type: ignore


async def main(job_id: str, upload_id: str):
    conn = await aio_pika.connect_robust(RABBITMQ_URL, **build_connect_kwargs())
    chan = await conn.channel()
    ex = await chan.declare_exchange(ANALYZE_EXCHANGE, aio_pika.ExchangeType.TOPIC, durable=True)
    payload = {"jobId": job_id, "uploadId": upload_id}
    body = json.dumps(payload).encode("utf-8")
    msg = aio_pika.Message(body=body, content_type="application/json", delivery_mode=aio_pika.DeliveryMode.PERSISTENT)
    await ex.publish(msg, routing_key="analyze.request")
    await conn.close()


if __name__ == "__main__":
    import sys
    if len(sys.argv) != 3:
        print("usage: python dev_publish_request.py <jobId> <uploadId>")
        raise SystemExit(2)
    asyncio.run(main(sys.argv[1], sys.argv[2]))
