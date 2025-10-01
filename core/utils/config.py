"""YAML 설정 파일 로딩 유틸리티."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import yaml


@dataclass
class Config:
    """YAML 설정을 래핑해 편리한 접근을 제공한다."""

    raw: Dict[str, Any]

    @staticmethod
    def load(path: str) -> "Config":
        """YAML 파일을 로드하여 Config 객체를 반환한다."""

        with open(path, "r", encoding="utf-8") as infile:
            data = yaml.safe_load(infile) or {}
        return Config(raw=data)

    def get(self, key: str, default: Any = None) -> Any:
        """dict.get과 동일한 동작을 제공한다."""

        return self.raw.get(key, default)

    def __getitem__(self, item: str) -> Any:
        """키 기반 접근을 지원한다."""

        return self.raw[item]
