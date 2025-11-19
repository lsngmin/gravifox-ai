"""객체 스토리지(S3 호환) 업로드 유틸리티."""

from __future__ import annotations

import asyncio
import datetime as dt
from pathlib import Path
from typing import Any, Dict, Optional

import boto3
from botocore.client import Config as BotoConfig
from botocore.exceptions import BotoCoreError, ClientError

from core.utils.logger import get_logger

from api.config import RuntimeSettings


class ObjectStorageClient:
    """S3 호환 객체 스토리지 업로드를 담당한다."""

    def __init__(self, settings: RuntimeSettings) -> None:
        self._logger = get_logger("api.services.object_storage")
        self._bucket = (settings.upload_s3_bucket or "").strip()
        self._enabled = bool(self._bucket)
        self._region = (settings.upload_s3_region or "").strip() or None
        self._prefix = (settings.upload_s3_prefix or "").strip().strip("/")
        self._base_url = (settings.upload_s3_base_url or "").strip().rstrip("/") or None
        self._endpoint = (settings.upload_s3_endpoint or "").strip() or None
        self._force_path = bool(settings.upload_s3_force_path_style)
        self._client = None

        if self._enabled:
            config_kwargs: Dict[str, Any] = {}
            if self._force_path:
                config_kwargs["s3"] = {"addressing_style": "path"}
            boto_config = BotoConfig(**config_kwargs) if config_kwargs else None
            client_kwargs: Dict[str, Any] = {}
            if self._region:
                client_kwargs["region_name"] = self._region
            if self._endpoint:
                client_kwargs["endpoint_url"] = self._endpoint
            if boto_config:
                client_kwargs["config"] = boto_config
            self._client = boto3.client("s3", **client_kwargs)
            self._logger.info(
                "[ObjectStorage] Initialized bucket=%s region=%s prefix=%s endpoint=%s",
                self._bucket,
                self._region or "<default>",
                self._prefix or "<root>",
                self._endpoint or "<aws-default>",
            )
        else:
            self._logger.info("[ObjectStorage] Disabled (UPLOAD_S3_BUCKET not configured)")

    @property
    def enabled(self) -> bool:
        """S3 업로드가 활성화되어 있는지 여부."""

        return self._enabled and self._client is not None

    def _build_key(self, upload_id: str) -> str:
        """버킷 내 오브젝트 키를 조합한다."""

        safe_upload_id = upload_id.lstrip("/")
        if self._prefix:
            return f"{self._prefix}/{safe_upload_id}"
        return safe_upload_id

    def _build_url(self, key: str) -> Optional[str]:
        if self._base_url:
            return f"{self._base_url}/{key}"
        if self._endpoint:
            base = self._endpoint.rstrip("/")
            return f"{base}/{self._bucket}/{key}" if self._force_path else f"{base}/{key}"
        if self._region:
            return f"https://{self._bucket}.s3.{self._region}.amazonaws.com/{key}"
        return f"https://{self._bucket}.s3.amazonaws.com/{key}"

    async def upload_file(
        self,
        source: Path,
        *,
        upload_id: str,
        content_type: Optional[str],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """로컬 파일을 S3에 업로드하고 메타데이터를 반환한다."""

        if not self.enabled:
            return {}
        if not source.exists():
            raise FileNotFoundError(f"Source file not found: {source}")

        key = self._build_key(upload_id)
        extra_args: Dict[str, Any] = {}
        if content_type:
            extra_args["ContentType"] = content_type
        if metadata:
            sanitized = {
                str(k): str(v)
                for k, v in metadata.items()
                if v is not None
            }
            if sanitized:
                extra_args["Metadata"] = sanitized

        upload_kwargs: Dict[str, Any] = {}
        if extra_args:
            upload_kwargs["ExtraArgs"] = extra_args

        try:
            await asyncio.to_thread(
                self._client.upload_file,
                str(source),
                self._bucket,
                key,
                **upload_kwargs,
            )
            head = await asyncio.to_thread(
                self._client.head_object,
                Bucket=self._bucket,
                Key=key,
            )
        except (BotoCoreError, ClientError) as exc:
            self._logger.error(
                "S3 upload failed bucket=%s key=%s: %s",
                self._bucket,
                key,
                exc,
            )
            raise

        etag = head.get("ETag")
        size = head.get("ContentLength")
        version_id = head.get("VersionId")
        uploaded_at = dt.datetime.utcnow().replace(tzinfo=dt.timezone.utc).isoformat()

        return {
            "storage": "s3",
            "bucket": self._bucket,
            "key": key,
            "region": self._region,
            "url": self._build_url(key),
            "etag": etag.strip('"') if isinstance(etag, str) else etag,
            "objectSize": size,
            "versionId": version_id,
            "uploadedAt": uploaded_at,
        }


__all__ = ["ObjectStorageClient"]
