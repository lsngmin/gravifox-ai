"""중앙 로깅 유틸리티 모듈.

이 모듈은 core 하위 코드에서 일관된 로깅 포맷과 레벨을 사용하도록 돕는다.
컬러 출력(coloredlogs)을 한 번만 초기화하고, 각 모듈에서는 `get_logger`로
로거를 가져가서 사용하면 된다. 또한 `log_time` 데코레이터를 제공하여
핵심 함수의 실행 시간을 자동으로 INFO 로그로 남길 수 있게 했다.
"""

from __future__ import annotations

import functools
import logging
import time
from typing import Any, Callable, Optional, TypeVar, cast
from pathlib import Path

import coloredlogs


_ROOT_INITIALIZED: bool = False
_DEFAULT_FORMAT = (
    "%(asctime)s | %(levelname)-8s | %(process)d-%(threadName)s | "
    "%(name)s | %(filename)s:%(lineno)d | %(funcName)s | %(message)s"
)


def _initialize_root_logger(level: int = logging.INFO) -> None:
    """루트 로거를 한 번만 초기화한다.

    Args:
        level: 루트 로거 기본 레벨.

    Returns:
        없음. 내부 전역 상태를 변경한다.
    """

    global _ROOT_INITIALIZED
    if _ROOT_INITIALIZED:
        return

    # 기존 핸들러 제거 후 coloredlogs로 일관된 포맷 구성
    root_logger = logging.getLogger()
    for handler in list(root_logger.handlers):
        root_logger.removeHandler(handler)

    coloredlogs.install(level=level, fmt=_DEFAULT_FORMAT, logger=root_logger)
    root_logger.setLevel(level)
    logging.captureWarnings(True)
    _ROOT_INITIALIZED = True


def get_logger(name: str, level: Optional[int] = None) -> logging.Logger:
    """모듈별 로거를 반환한다.

    Args:
        name: 로거 이름(일반적으로 __name__).
        level: 개별 로거 레벨을 지정하고 싶을 때 사용.

    Returns:
        logging.Logger 인스턴스.
    """

    _initialize_root_logger(level or logging.INFO)
    logger = logging.getLogger(name)
    if level is not None:
        logger.setLevel(level)
    return logger

def add_file_handler(log_dir: Path, level: int = logging.INFO) -> None:
    """파일 핸들러를 추가해 로그를 저장한다.

        Args:
            log_dir: 로그 파일을 보관할 디렉터리.
            level: 로깅 레벨.

        Returns:
            None
        """
    log_dir.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(log_dir / "train.log", encoding="utf-8")
    handler.setFormatter(logging.Formatter(_DEFAULT_FORMAT))
    root = logging.getLogger()
    root.addHandler(handler)
    root.setLevel(level)

F = TypeVar("F", bound=Callable[..., Any])


def log_time(func: F) -> F:
    """함수 실행 시간을 INFO 레벨로 기록하는 데코레이터.

    Args:
        func: 데코레이팅할 함수.

    Returns:
        동일한 시그니처를 갖는 래핑 함수.
    """

    logger = get_logger(func.__module__)

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any):  # type: ignore[override]
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = (time.perf_counter() - start) * 1000.0
        logger.info("%s 실행 완료 - %.2fms", func.__qualname__, elapsed)
        return result

    return cast(F, wrapper)
