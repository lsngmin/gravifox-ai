"""SNS 특화 증강 프리셋 빌더."""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Callable, Mapping, Optional

from core.data.transforms_sns import AugConfig, generate_sns_augmentations


def _as_tuple(value: Any, default: tuple[int, int]) -> tuple[int, int]:
    if value is None:
        return default
    if isinstance(value, (list, tuple)) and len(value) == 2:
        return int(value[0]), int(value[1])
    raise ValueError(f"chain_depth must be length-2 sequence, got: {value}")


def build_sns_augment(spec: Optional[Mapping[str, Any]]) -> Optional[Callable]:
    """스펙 딕셔너리를 받아 SNS 증강 콜러블을 생성한다."""

    if not spec:
        return None

    spec_type = str(spec.get("type", "sns")).lower()
    if spec_type not in {"sns", "sns_augment", "snsaugment"}:
        raise ValueError(f"unsupported augment type: {spec_type}")

    chain_depth = _as_tuple(spec.get("chain_depth"), default=(1, 3))
    params = dict(spec.get("params", {}))

    base = AugConfig()
    cfg = replace(base, **params)

    return generate_sns_augmentations(chain_depth=chain_depth, config=cfg)
