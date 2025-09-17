import asyncio
import json
import os
import aio_pika

RABBITMQ_URL = os.environ.get("RABBITMQ_URL", "amqp://admin:admin@localhost:5672/")
EXCHANGE_NAME = os.environ.get("ANALYZE_EXCHANGE", "analyze.exchange")


async def main(job_id: str, upload_id: str):
    conn = await aio_pika.connect_robust(RABBITMQ_URL)
    chan = await conn.channel()
    ex = await chan.declare_exchange(EXCHANGE_NAME, aio_pika.ExchangeType.TOPIC, durable=True)
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
