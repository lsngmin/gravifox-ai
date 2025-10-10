"""비동기 배치 추론 큐 구현."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Awaitable, Callable, List, Optional

import torch

from core.utils.logger import get_logger


@dataclass(slots=True)
class _QueueItem:
    """배치 큐에 적재되는 개별 요청 정보."""

    tensor: torch.Tensor
    future: asyncio.Future[List[float]]
    enqueue_time: float


class BatchInferenceQueue:
    """GPU 활용률을 높이기 위한 비동기 배치 처리 큐."""

    def __init__(
        self,
        *,
        max_batch_size: int,
        max_wait_ms: int,
        runner: Callable[[torch.Tensor], Awaitable[torch.Tensor]],
        queue_name: str = "vit",
    ) -> None:
        """큐를 초기화한다.

        Args:
            max_batch_size: 한 번에 묶어서 추론할 최대 요청 수.
            max_wait_ms: 최초 요청이 들어온 이후 대기할 최대 시간(ms).
            runner: 실제 배치 추론을 수행할 코루틴 콜백.
            queue_name: 로그 식별을 위한 큐 이름.
        """

        if max_batch_size <= 0:
            raise ValueError("max_batch_size는 1 이상이어야 합니다.")
        if max_wait_ms < 0:
            raise ValueError("max_wait_ms는 0 이상이어야 합니다.")

        self._max_batch_size = max_batch_size
        self._max_wait = max_wait_ms / 1000.0
        self._runner = runner
        self._queue: asyncio.Queue[Optional[_QueueItem]] = asyncio.Queue()
        self._logger = get_logger(f"api.services.batch_queue.{queue_name}")
        self._worker_task: Optional[asyncio.Task[None]] = None
        self._closed = False

    async def enqueue(self, tensor: torch.Tensor) -> List[float]:
        """요청을 큐에 추가하고 결과를 기다린다.

        Args:
            tensor: (1, C, H, W) 형태의 입력 텐서.

        Returns:
            소프트맥스 확률 분포 리스트.
        """

        if self._closed:
            raise RuntimeError("배치 큐가 이미 종료되었습니다.")
        self._ensure_worker()
        future: asyncio.Future[List[float]] = asyncio.get_running_loop().create_future()
        item = _QueueItem(
            tensor=tensor.cpu(), future=future, enqueue_time=time.perf_counter()
        )
        await self._queue.put(item)
        return await future

    async def start(self) -> None:
        """백그라운드 워커를 시작한다."""

        if self._closed:
            raise RuntimeError("배치 큐가 이미 종료되었습니다.")
        self._ensure_worker()

    async def close(self) -> None:
        """백그라운드 워커를 종료하고 대기 중인 요청을 정리한다."""

        if self._closed:
            return
        self._closed = True
        if self._worker_task is None:
            return
        await self._queue.put(None)
        try:
            await self._worker_task
        except asyncio.CancelledError:  # pragma: no cover - 종료 경로
            self._logger.warning("배치 워커가 취소되었습니다", exc_info=True)
        while not self._queue.empty():
            try:
                pending = self._queue.get_nowait()
            except asyncio.QueueEmpty:  # pragma: no cover - 동시성 보호
                break
            if pending is None:
                continue
            if not pending.future.done():
                pending.future.set_exception(RuntimeError("배치 큐가 종료되었습니다."))
        self._worker_task = None

    def _ensure_worker(self) -> None:
        """워커 태스크가 실행 중인지 확인한다."""

        if self._worker_task is not None and not self._worker_task.done():
            return
        loop = asyncio.get_running_loop()
        self._worker_task = loop.create_task(self._worker_loop())

    async def _worker_loop(self) -> None:
        """큐에 적재된 요청을 모아 배치 추론을 수행한다."""

        while True:
            item = await self._queue.get()
            if item is None:
                self._queue.task_done()
                break
            batch = [item]
            deadline = time.perf_counter() + self._max_wait
            while len(batch) < self._max_batch_size:
                timeout = deadline - time.perf_counter()
                if timeout <= 0:
                    break
                try:
                    next_item = await asyncio.wait_for(
                        self._queue.get(), timeout=timeout
                    )
                except asyncio.TimeoutError:
                    break
                if next_item is None:
                    self._queue.task_done()
                    await self._queue.put(None)
                    break
                batch.append(next_item)
            tensors = torch.cat([entry.tensor for entry in batch], dim=0)
            try:
                probs = await self._runner(tensors)
            except Exception as exc:  # pragma: no cover - 추론 실패 경로
                for entry in batch:
                    if not entry.future.done():
                        entry.future.set_exception(exc)
                self._logger.exception("배치 추론 중 예외가 발생했습니다")
            else:
                distributions = probs.cpu().tolist()
                for entry, dist in zip(batch, distributions):
                    if not entry.future.done():
                        entry.future.set_result([float(x) for x in dist])
                self._logger.debug(
                    "배치 추론 완료 - size=%d wait_ms=%.2f",
                    len(batch),
                    (time.perf_counter() - batch[0].enqueue_time) * 1000.0,
                )
            finally:
                for _ in batch:
                    self._queue.task_done()


__all__ = ["BatchInferenceQueue"]
