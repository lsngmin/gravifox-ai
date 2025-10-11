"""업로드 파일 저장 및 정리를 담당한다."""

from __future__ import annotations

import asyncio
import datetime as dt
import re
import uuid
from enum import Enum
from pathlib import Path
from typing import Optional

from fastapi import HTTPException, UploadFile

from core.utils.logger import get_logger

from api.config import RuntimeSettings


class MediaKind(str, Enum):
    """업로드 가능한 미디어 종류."""

    IMAGE = "image"
    VIDEO = "video"


class MediaStorageService:
    """업로드 파일 관리를 담당하는 서비스."""

    def __init__(self, settings: RuntimeSettings) -> None:
        """서비스를 초기화한다.

        Args:
            settings: 런타임 환경 설정.
        """

        self._settings = settings
        self._logger = get_logger("api.services.storage")
        self._root = settings.file_store_root
        self._root.mkdir(parents=True, exist_ok=True)

    def infer_media_kind(self, filename: str, content_type: Optional[str]) -> MediaKind:
        """파일 이름과 Content-Type을 기반으로 미디어 종류를 추론한다.

        Args:
            filename: 업로드 파일 이름.
            content_type: HTTP Content-Type 값.

        Returns:
            추론된 미디어 종류.
        """

        content = (content_type or "").lower()
        if content.startswith("image/"):
            return MediaKind.IMAGE
        if content.startswith("video/"):
            return MediaKind.VIDEO
        ext = Path(filename or "").suffix.lower().lstrip(".")
        if ext in {"jpg", "jpeg", "png", "webp"}:
            return MediaKind.IMAGE
        if ext in {"mp4", "mov", "webm"}:
            return MediaKind.VIDEO
        raise HTTPException(status_code=415, detail="unsupported media type")

    async def save_upload(
        self, file: UploadFile, kind: MediaKind, upload_id: Optional[str] = None
    ) -> str:
        """업로드 파일을 저장하고 식별자를 반환한다.

        Args:
            file: FastAPI 업로드 파일 객체.
            kind: 저장할 미디어 종류.

        Returns:
            저장된 파일의 업로드 식별자.
        """

        max_bytes = int(
            (
                self._settings.max_image_mb
                if kind is MediaKind.IMAGE
                else self._settings.max_video_mb
            )
            * 1024
            * 1024
        )
        candidate = self._resolve_upload_id(upload_id, file)
        destination = self._root / candidate
        written = 0
        try:
            with destination.open("wb") as fp:
                while True:
                    chunk = await file.read(1024 * 1024)
                    if not chunk:
                        break
                    written += len(chunk)
                    if written > max_bytes:
                        fp.close()
                        destination.unlink(missing_ok=True)
                        raise HTTPException(status_code=413, detail="file too large")
                    fp.write(chunk)
        except HTTPException:
            raise
        except Exception as exc:  # pragma: no cover - 파일 시스템 오류 대응
            destination.unlink(missing_ok=True)
            self._logger.error("파일 저장 실패: %s", exc)
            raise HTTPException(status_code=500, detail="upload failed") from exc
        return candidate

    def _resolve_upload_id(self, upload_id: Optional[str], file: UploadFile) -> str:
        suffix = Path(file.filename or "media").suffix.lower()
        if upload_id:
            candidate = upload_id.strip()
            if not candidate:
                raise HTTPException(status_code=400, detail="uploadId required")
            if re.search(r"[\\/]|\.\.", candidate):
                raise HTTPException(status_code=400, detail="invalid uploadId")
            if len(candidate) > 160:
                raise HTTPException(status_code=400, detail="uploadId too long")
            if "." not in Path(candidate).name and suffix:
                candidate = f"{candidate}{suffix}"
            return candidate
        now = dt.datetime.utcnow().strftime("%Y%m%dT%H%M%S")
        base = f"{now}_{uuid.uuid4().hex}"
        return f"{base}{suffix}"

    async def cleanup_expired(self) -> None:
        """TTL을 초과한 파일을 삭제한다.

        Returns:
            없음.
        """

        ttl = self._settings.file_ttl_hours
        if ttl <= 0:
            return
        expire_delta = dt.timedelta(hours=ttl)
        now = dt.datetime.utcnow()
        for path in self._root.glob("*"):
            try:
                mtime = dt.datetime.utcfromtimestamp(path.stat().st_mtime)
                if now - mtime > expire_delta:
                    path.unlink()
            except Exception:  # pragma: no cover - best effort cleanup
                self._logger.exception("파일 정리 중 오류 발생", exc_info=True)

    async def run_cleanup_loop(self) -> None:
        """백그라운드에서 주기적으로 TTL 정리를 수행한다.

        Returns:
            없음.
        """

        while True:
            await self.cleanup_expired()
            await asyncio.sleep(3600)


__all__ = ["MediaStorageService", "MediaKind"]
