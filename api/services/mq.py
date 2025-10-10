"""MQ 연동 로직을 담당한다."""

from __future__ import annotations

import json
import ssl
from typing import Awaitable, Callable, Optional

import aio_pika

from core.utils.logger import get_logger

from api.config import RuntimeSettings


class MQService:
    """RabbitMQ 연동을 캡슐화하는 서비스."""

    def __init__(self, settings: RuntimeSettings) -> None:
        """서비스를 초기화한다.

        Args:
            settings: 런타임 환경 설정.
        """

        self._settings = settings
        self._logger = get_logger("api.services.mq")
        self._connection: Optional[aio_pika.RobustConnection] = None
        self._channel: Optional[aio_pika.RobustChannel] = None
        self._exchange: Optional[aio_pika.RobustExchange] = None

    async def connect(self) -> None:
        """RabbitMQ에 연결한다.

        Returns:
            없음.
        """

        kwargs = self._build_connect_kwargs()
        self._connection = await aio_pika.connect_robust(
            self._settings.rabbitmq_url, **kwargs
        )
        self._channel = await self._connection.channel()
        await self._channel.set_qos(prefetch_count=self._settings.rabbitmq_prefetch)
        self._exchange = await self._channel.declare_exchange(
            self._settings.analyze_exchange,
            aio_pika.ExchangeType.TOPIC,
            durable=True,
        )

    async def close(self) -> None:
        """열린 연결을 정리한다.

        Returns:
            없음.
        """

        if self._connection:
            await self._connection.close()
            self._connection = None
            self._channel = None
            self._exchange = None

    async def publish_json(self, routing_key: str, payload: dict) -> None:
        """JSON 메시지를 발행한다.

        Args:
            routing_key: 발행에 사용할 라우팅 키.
            payload: 직렬화할 JSON 데이터.

        Returns:
            없음.
        """

        if self._exchange is None:
            raise RuntimeError("exchange not initialized")
        body = json.dumps(payload).encode("utf-8")
        message = aio_pika.Message(
            body=body,
            content_type="application/json",
            delivery_mode=aio_pika.DeliveryMode.PERSISTENT,
        )
        self._logger.info("MQ publish rk=%s bytes=%d", routing_key, len(body))
        await self._exchange.publish(message, routing_key=routing_key)

    async def consume_requests(
        self, handler: Callable[[dict], Awaitable[None]]
    ) -> None:
        """요청 큐를 소비하면서 핸들러를 호출한다.

        Args:
            handler: 메시지를 처리할 비동기 콜백.

        Returns:
            없음.
        """

        if self._channel is None or self._exchange is None:
            raise RuntimeError("channel not initialized")
        queue = await self._channel.declare_queue(
            self._settings.request_queue, durable=True
        )
        await queue.bind(self._exchange, routing_key="analyze.request")
        async with queue.iterator() as iterator:
            async for message in iterator:
                async with message.process():
                    try:
                        payload = json.loads(message.body.decode("utf-8"))
                    except Exception:
                        self._logger.warning("MQ 메시지 파싱 실패")
                        continue
                    await handler(payload)

    def _build_connect_kwargs(self) -> dict:
        """연결 시 사용할 TLS 파라미터를 생성한다.

        Returns:
            aio_pika 연결 인자 딕셔너리.
        """

        use_tls, context = self._resolve_tls()
        if not use_tls:
            return {}
        kwargs = {"ssl": True}
        if context is not None:
            kwargs["ssl_options"] = context
        return kwargs

    def _resolve_tls(self) -> tuple[bool, Optional[ssl.SSLContext]]:
        """TLS 사용 여부와 컨텍스트를 계산한다.

        Returns:
            (TLS 사용 여부, SSL 컨텍스트) 튜플.
        """

        override = self._settings.rabbitmq_use_tls
        url = self._settings.rabbitmq_url.lower()
        use_tls = override if override is not None else url.startswith("amqps://")
        if not use_tls:
            return False, None
        cafile = (
            str(self._settings.rabbitmq_ca_file)
            if self._settings.rabbitmq_ca_file
            else None
        )
        context = ssl.create_default_context(cafile=cafile)
        certfile = self._settings.rabbitmq_cert_file
        keyfile = self._settings.rabbitmq_key_file
        if certfile:
            context.load_cert_chain(
                certfile=str(certfile), keyfile=str(keyfile) if keyfile else None
            )
        if self._settings.rabbitmq_verify_peer is False:
            context.check_hostname = False
            context.verify_mode = ssl.CERT_NONE
        return True, context


async def publish_progress(mq: MQService, job_id: str, data: dict) -> None:
    """분석 진행 상황을 발행한다.

    Args:
        mq: MQ 서비스 인스턴스.
        job_id: 작업 식별자.
        data: 추가 진행 정보.

    Returns:
        없음.
    """

    await mq.publish_json(f"analyze.progress.{job_id}", {"jobId": job_id, **data})


async def publish_result(mq: MQService, job_id: str, result: dict) -> None:
    """분석 결과를 발행한다.

    Args:
        mq: MQ 서비스 인스턴스.
        job_id: 작업 식별자.
        result: 결과 데이터.

    Returns:
        없음.
    """

    await mq.publish_json(
        f"analyze.result.{job_id}", {"jobId": job_id, "result": result}
    )


async def publish_failed(
    mq: MQService, job_id: str, reason: str, *, reason_code: Optional[str] = None
) -> None:
    """실패 상태를 MQ로 알린다.

    Args:
        mq: MQ 서비스 인스턴스.
        job_id: 작업 식별자.
        reason: 실패 원인 메시지.
        reason_code: 선택적 실패 코드.

    Returns:
        없음.
    """

    payload = {"jobId": job_id, "status": "FAILED", "reason": reason}
    if reason_code:
        payload["reasonCode"] = reason_code
    await mq.publish_json(f"analyze.failed.{job_id}", payload)


__all__ = ["MQService", "publish_progress", "publish_result", "publish_failed"]
