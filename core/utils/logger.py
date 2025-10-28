"""중앙 로깅 유틸리티 모듈.

이 모듈은 core 하위 코드에서 일관된 로깅 포맷과 레벨을 사용하도록 돕는다.
컬러 출력(coloredlogs)을 한 번만 초기화하고, 각 모듈에서는 `get_logger`로
시스템 로그를, `get_train_logger`로 학습 전용 로그를 받아 사용한다.
또한 `setup_experiment_loggers`를 통해 실험별 단일 로그 파일(옵션에 따라 JSON)
을 구성한다.
"""

from __future__ import annotations

import functools
import json
import logging
import time
from pathlib import Path
from typing import Any, Callable, Optional, TypeVar, cast

import coloredlogs


_ROOT_INITIALIZED: bool = False
_TRAIN_LOGGER_NAME = "gravifox.train"
_SYSTEM_LOGGER_NAME = "gravifox.system"
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


def _system_base_logger() -> logging.Logger:
    """시스템 로그의 루트 로거를 반환한다."""

    _initialize_root_logger(logging.INFO)
    return logging.getLogger(_SYSTEM_LOGGER_NAME)


def get_logger(name: str, level: Optional[int] = None) -> logging.Logger:
    """시스템 로그 계층에서 파생된 모듈별 로거를 반환한다."""

    base = _system_base_logger()
    logger = base.getChild(name)
    if level is not None:
        logger.setLevel(level)
    return logger


def get_system_logger() -> logging.Logger:
    """시스템 로그 베이스 로거를 반환한다."""

    return _system_base_logger()


def get_train_logger(level: Optional[int] = None) -> logging.Logger:
    """학습 전용 로그 로거를 반환한다."""

    _initialize_root_logger(level or logging.INFO)
    logger = logging.getLogger(_TRAIN_LOGGER_NAME)
    if level is not None:
        logger.setLevel(level)
    return logger


class _JsonFormatter(logging.Formatter):
    """JSONL 형식 출력을 위한 포매터."""

    def format(self, record: logging.LogRecord) -> str:  # type: ignore[override]
        payload = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(record.created)) + f".{int(record.msecs):03d}Z",
            "level": record.levelname,
            "name": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)


def _clear_file_handlers(logger: logging.Logger) -> None:
    for handler in list(logger.handlers):
        if isinstance(handler, logging.FileHandler):
            logger.removeHandler(handler)
            handler.close()


def setup_experiment_loggers(
    exp_dir: Path,
    *,
    level: int = logging.INFO,
    console: bool = True,
    log_path: str | Path | None = "logs/experiment.log",
    json_format: bool = True,
) -> None:
    """실험 디렉터리에 대해 단일 로그 파일을 설정한다."""

    _initialize_root_logger(level)
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    for handler in list(root_logger.handlers):
        if isinstance(handler, logging.FileHandler):
            root_logger.removeHandler(handler)
            handler.close()
        else:
            handler.setLevel(level if console else max(level + 10, logging.CRITICAL))

    def _resolve(pathlike: str | Path) -> Path:
        path = Path(pathlike)
        if not path.is_absolute():
            path = exp_dir / path
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    if log_path is not None:
        log_full = _resolve(log_path)
        file_handler = logging.FileHandler(log_full, encoding="utf-8")
        formatter: logging.Formatter = _JsonFormatter() if json_format else logging.Formatter(_DEFAULT_FORMAT)
        file_handler.setFormatter(formatter)
        file_handler.setLevel(level)
        root_logger.addHandler(file_handler)

    system_logger = get_system_logger()
    system_logger.setLevel(level)
    system_logger.propagate = True
    _clear_file_handlers(system_logger)

    train_logger = get_train_logger(level)
    train_logger.setLevel(level)
    train_logger.propagate = True
    _clear_file_handlers(train_logger)

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
