from __future__ import annotations

import json
import os
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

_LOCK = threading.Lock()
_CACHE: Optional[Tuple[float, Dict[str, object]]] = None


def _catalog_path() -> Path:
    base = os.environ.get("MODEL_CATALOG_PATH")
    if base:
        return Path(base).expanduser().resolve()
    return (Path(__file__).resolve().parent / "models" / "catalog.json").resolve()


def _load_catalog() -> Dict[str, object]:
    path = _catalog_path()
    if not path.is_file():
        raise FileNotFoundError(f"Model catalog not found: {path}")
    with path.open("r", encoding="utf-8") as fp:
        data = json.load(fp)
    if not isinstance(data, dict):
        raise ValueError(f"Invalid model catalog format: {path}")
    return data


def _get_catalog() -> Dict[str, object]:
    global _CACHE
    path = _catalog_path()
    mtime = path.stat().st_mtime
    with _LOCK:
        if _CACHE and _CACHE[0] == mtime:
            return _CACHE[1]
        data = _load_catalog()
        _CACHE = (mtime, data)
        return data


@dataclass(frozen=True)
class ModelInfo:
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

    @property
    def file_path(self) -> Path:
        path = Path(self.path)
        if path.is_absolute():
            return path
        base = _catalog_path().parent
        return (base / path).resolve()


def _normalize_item(item: Dict[str, object]) -> ModelInfo:
    key = str(item.get("key", "")).strip()
    if not key:
        raise ValueError("model entry missing key")
    model_type = str(item.get("type", "")).strip() or "torch_image"
    path = str(item.get("path", "")).strip()
    if not path:
        raise ValueError(f"model {key} missing path")
    labels = item.get("labels") or ()
    if isinstance(labels, list):
        labels = tuple(str(x) for x in labels)
    extras = {
        k: v for k, v in item.items()
        if k not in {"key", "name", "version", "description", "type", "path", "threshold", "input", "labels"}
    }
    return ModelInfo(
        key=key,
        name=str(item.get("name") or key),
        version=item.get("version"),
        description=item.get("description"),
        type=model_type,
        path=path,
        threshold=float(item.get("threshold", 0.5)),
        input=str(item.get("input") or "image"),
        labels=labels,
        extras=extras,
    )


def list_models() -> List[ModelInfo]:
    raw = _get_catalog()
    items = raw.get("items") or []
    models = []
    for item in items:
        try:
            models.append(_normalize_item(item))
        except Exception as exc:
            raise ValueError(f"Failed to parse model entry {item!r}: {exc}") from exc
    return models


def get_default_model() -> ModelInfo:
    catalog = _get_catalog()
    default_key = str(catalog.get("defaultKey") or "").strip()
    models = {m.key: m for m in list_models()}
    if default_key and default_key in models:
        return models[default_key]
    if not models:
        raise RuntimeError("Model catalog has no items")
    if default_key and default_key not in models:
        raise KeyError(f"Default model key '{default_key}' not found in catalog")
    return models[next(iter(models))]


def resolve_model(key: Optional[str]) -> ModelInfo:
    if key:
        key_norm = key.strip()
        if key_norm:
            for model in list_models():
                if model.key == key_norm:
                    return model
            raise KeyError(f"Unknown model key: {key_norm}")
    return get_default_model()
