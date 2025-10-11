"""업로드 토큰 검증 및 상태 보고 서비스."""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

import jwt
import requests
from fastapi import HTTPException

from api.config import RuntimeSettings


@dataclass(slots=True)
class UploadTokenClaims:
    """업로드 토큰에 포함된 클레임."""

    upload_id: str
    jti: str
    expires_at: datetime


@dataclass(slots=True)
class UploadTokenContext:
    """Spring 측 토큰 상태 추적용 식별자."""

    token_id: int
    upload_id: str
    jti: str


class UploadTokenVerifier:
    """Spring JWKS 기반 업로드 토큰 검증기."""

    def __init__(self, settings: RuntimeSettings) -> None:
        self._jwks_url = settings.upload_jwks_url
        self._cache_ttl = int(settings.upload_jwks_cache_seconds or 300)
        self._keys: dict[str, dict[str, str]] = {}
        self._fetched_at: float = 0.0

    async def verify(self, token: Optional[str]) -> UploadTokenClaims:
        """JWT 형식의 업로드 토큰을 검증한다."""

        if not token:
            raise HTTPException(status_code=401, detail="upload token missing")
        header = self._decode_header(token)
        kid = header.get("kid")
        if not kid:
            raise HTTPException(status_code=401, detail="upload token invalid")
        key = await self._get_public_key(kid)
        try:
            claims = jwt.decode(
                token,
                key=key,
                algorithms=["RS256"],
                options={"require": ["exp", "jti"], "verify_aud": False},
                issuer="gravifox-upload",
            )
        except jwt.ExpiredSignatureError as exc:
            raise HTTPException(status_code=401, detail="upload token expired") from exc
        except jwt.InvalidTokenError as exc:
            raise HTTPException(status_code=401, detail="upload token invalid") from exc

        upload_id = str(claims.get("uploadId") or "").strip()
        jti = str(claims.get("jti") or "").strip()
        if not upload_id or not jti:
            raise HTTPException(status_code=401, detail="upload token missing claims")
        exp_ts = claims.get("exp")
        expires_at = datetime.fromtimestamp(exp_ts, tz=timezone.utc)
        return UploadTokenClaims(upload_id=upload_id, jti=jti, expires_at=expires_at)

    def _decode_header(self, token: str) -> dict[str, str]:
        try:
            return jwt.get_unverified_header(token)
        except jwt.InvalidTokenError as exc:
            raise HTTPException(status_code=401, detail="upload token invalid") from exc

    async def _get_public_key(self, kid: str):
        await self._ensure_keys()
        key = self._keys.get(kid)
        if key is None:
            await self._ensure_keys(force=True)
            key = self._keys.get(kid)
        if key is None:
            raise HTTPException(status_code=401, detail="upload token key not found")
        return jwt.algorithms.RSAAlgorithm.from_jwk(json.dumps(key))

    async def _ensure_keys(self, force: bool = False) -> None:
        if not self._jwks_url:
            raise HTTPException(status_code=500, detail="upload jwks url not configured")
        if (
            not force
            and self._keys
            and (time.time() - self._fetched_at) < max(self._cache_ttl, 30)
        ):
            return
        try:
            response = await asyncio.to_thread(
                requests.get, self._jwks_url, timeout=5
            )
        except Exception as exc:  # pragma: no cover - 네트워크 오류
            raise HTTPException(status_code=503, detail="upload jwks fetch failed") from exc

        if response.status_code != 200:
            raise HTTPException(status_code=503, detail="upload jwks unavailable")

        try:
            payload = response.json()
        except ValueError as exc:
            raise HTTPException(status_code=503, detail="upload jwks invalid") from exc

        keys = payload.get("keys", [])
        self._keys = {
            str(key["kid"]): key for key in keys if isinstance(key, dict) and "kid" in key
        }
        self._fetched_at = time.time()


class UploadTokenRegistryClient:
    """Spring UploadTokenService와 상호작용해 단일 사용 상태를 추적한다."""

    def __init__(self, settings: RuntimeSettings) -> None:
        self._base_url = (settings.upload_token_api_base or "").rstrip("/")
        self._service_key = settings.upload_token_service_key

    async def authorize(self, token: str) -> Optional[UploadTokenContext]:
        """토큰 사용을 시작하며 상태를 IN_PROGRESS로 전환한다."""

        if not self._base_url:
            return None
        response = await self._post(
            "/authorize",
            headers=self._build_headers(token),
        )
        data = self._parse_response(response, expected=200)
        return UploadTokenContext(
            token_id=int(data["tokenId"]),
            upload_id=str(data["uploadId"]),
            jti=str(data["jti"]),
        )

    async def complete_success(self, context: Optional[UploadTokenContext]) -> None:
        """업로드 성공 시 토큰을 소비 처리한다."""

        if not self._base_url or context is None:
            return
        response = await self._post(
            "/success",
            json={
                "tokenId": context.token_id,
                "uploadId": context.upload_id,
                "jti": context.jti,
            },
        )
        self._ensure_success(response)

    async def complete_failure(
        self, context: Optional[UploadTokenContext], reason: Optional[str] = None
    ) -> None:
        """업로드 실패 시 토큰을 실패 처리한다."""

        if not self._base_url or context is None:
            return
        payload = {
            "tokenId": context.token_id,
            "uploadId": context.upload_id,
            "jti": context.jti,
        }
        if reason:
            payload["reason"] = reason
        response = await self._post("/failure", json=payload)
        self._ensure_success(response)

    async def _post(self, path: str, headers=None, json: Optional[dict] = None):
        url = f"{self._base_url}{path}"
        headers = headers or {}
        if self._service_key:
            headers.setdefault("X-Service-Key", self._service_key)
        try:
            return await asyncio.to_thread(
                requests.post, url, json=json, headers=headers, timeout=5
            )
        except Exception as exc:  # pragma: no cover - 네트워크 오류
            raise HTTPException(
                status_code=503, detail="upload token service unavailable"
            ) from exc

    def _parse_response(self, response, expected: int) -> dict:
        if response.status_code != expected:
            detail = self._extract_error(response)
            raise HTTPException(status_code=response.status_code, detail=detail)
        try:
            data = response.json()
        except ValueError as exc:
            raise HTTPException(status_code=502, detail="upload token service invalid") from exc
        return data

    @staticmethod
    def _ensure_success(response) -> None:
        if response.status_code != 200:
            detail = UploadTokenRegistryClient._extract_error(response)
            raise HTTPException(status_code=response.status_code, detail=detail)

    @staticmethod
    def _extract_error(response) -> str:
        try:
            payload = response.json()
            return str(payload.get("message") or payload.get("detail") or "upload token rejected")
        except ValueError:
            return response.text or "upload token rejected"

    def _build_headers(self, token: str) -> dict[str, str]:
        headers = {"Upload-Token": token}
        if self._service_key:
            headers["X-Service-Key"] = self._service_key
        return headers


__all__ = [
    "UploadTokenClaims",
    "UploadTokenContext",
    "UploadTokenVerifier",
    "UploadTokenRegistryClient",
]
