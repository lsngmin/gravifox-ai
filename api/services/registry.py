"""모델 카탈로그 조회 기능을 제공한다."""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from api.config import RuntimeSettings


@dataclass(frozen=True)
class ModelInfo:
    """모델 카탈로그 항목을 표현한다."""

    key: str
    name: str
    version: Optional[str]
    description: Optional[str]
    type: str
    path: str
    threshold: float = 0.5
    input: str = "image"
    labels: Tuple[str, ...] = ()
    extras: Dict[str, object] = field(default_factory=dict)
    catalog_dir: Path = field(default=Path("."), repr=False)

    @property
    def file_path(self) -> Path:
        """모델 파일 경로를 반환한다.

        Returns:
            모델 가중치 파일의 절대 경로.
        """

        path = Path(self.path)
        if path.is_absolute():
            return path
        return (self.catalog_dir / path).resolve()


class ModelRegistryService:
    """모델 카탈로그를 로드하고 제공하는 서비스."""

    def __init__(self, settings: RuntimeSettings) -> None:
        """서비스를 초기화한다.

        Args:
            settings: 런타임 환경 설정.
        """

        self._settings = settings
        self._lock = threading.Lock()
        self._cache: Optional[Tuple[float, Dict[str, object]]] = None

    def list_models(self) -> List[ModelInfo]:
        """카탈로그에 등록된 모델 목록을 반환한다.

        Returns:
            `ModelInfo` 객체 리스트.
        """

        raw = self._get_catalog()
        items = raw.get("items") or []
        models: List[ModelInfo] = []
        for item in items:
            if not isinstance(item, dict):
                raise ValueError("invalid model catalog entry")
            models.append(self._normalize_item(item))
        return models

    def get_default_model(self) -> ModelInfo:
        """카탈로그의 기본 모델을 반환한다.

        Returns:
            기본 모델 정보.
        """

        catalog = self._get_catalog()
        default_key = str(catalog.get("defaultKey") or "").strip()
        models = {model.key: model for model in self.list_models()}
        if default_key and default_key in models:
            return models[default_key]
        if not models:
            raise RuntimeError("model catalog has no entries")
        if default_key and default_key not in models:
            raise KeyError(f"default model '{default_key}' not found")
        return next(iter(models.values()))

    def resolve_model(self, key: Optional[str]) -> ModelInfo:
        """키에 해당하는 모델 정보를 반환한다.

        Args:
            key: 조회할 모델 키.

        Returns:
            일치하는 모델 정보.
        """

        if key:
            normalized = key.strip()
            if normalized:
                for model in self.list_models():
                    if model.key == normalized:
                        return model
                raise KeyError(f"unknown model key: {normalized}")
        return self.get_default_model()

    def _catalog_path(self) -> Path:
        """카탈로그 파일 경로를 반환한다.

        Returns:
            카탈로그 JSON 파일 경로.
        """

        path = self._settings.model_catalog_path
        if not path.is_absolute():
            path = path.resolve()
        return path

    def _load_catalog(self) -> Dict[str, object]:
        """카탈로그 JSON 파일을 로드한다.

        Returns:
            로드된 JSON 딕셔너리.
        """

        path = self._catalog_path()
        if not path.is_file():
            raise FileNotFoundError(f"model catalog not found: {path}")
        with path.open("r", encoding="utf-8") as fp:
            data = json.load(fp)
        if not isinstance(data, dict):
            raise ValueError(f"invalid model catalog format: {path}")
        return data

    def _get_catalog(self) -> Dict[str, object]:
        """캐시된 카탈로그 데이터를 반환한다.

        Returns:
            JSON 데이터 딕셔너리.
        """

        path = self._catalog_path()
        mtime = path.stat().st_mtime
        with self._lock:
            if self._cache and self._cache[0] == mtime:
                return self._cache[1]
            data = self._load_catalog()
            self._cache = (mtime, data)
            return data

    def _normalize_item(self, item: Dict[str, object]) -> ModelInfo:
        """JSON 항목을 `ModelInfo`로 변환한다.

        Args:
            item: 카탈로그 JSON 항목.

        Returns:
            파싱된 모델 정보.
        """

        key = str(item.get("key", "")).strip()
        if not key:
            raise ValueError("model entry missing key")
        path = str(item.get("path", "")).strip()
        if not path:
            raise ValueError(f"model {key} missing path")
        labels_raw = item.get("labels") or ()
        if isinstance(labels_raw, list):
            labels_tuple = tuple(str(x) for x in labels_raw)
        elif isinstance(labels_raw, tuple):
            labels_tuple = tuple(str(x) for x in labels_raw)
        else:
            labels_tuple = ()
        extras = {
            k: v
            for k, v in item.items()
            if k
            not in {
                "key",
                "name",
                "version",
                "description",
                "type",
                "path",
                "threshold",
                "input",
                "labels",
            }
        }
        catalog_dir = self._catalog_path().parent
        return ModelInfo(
            key=key,
            name=str(item.get("name") or key),
            version=item.get("version"),
            description=item.get("description"),
            type=str(item.get("type") or "torch_image"),
            path=path,
            threshold=float(item.get("threshold", 0.5)),
            input=str(item.get("input") or "image"),
            labels=labels_tuple,
            extras=extras,
            catalog_dir=catalog_dir,
        )


__all__ = ["ModelRegistryService", "ModelInfo"]
