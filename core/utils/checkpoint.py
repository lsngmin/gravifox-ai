"""체크포인트 저장 유틸리티."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import torch


def save_checkpoint(state: Dict[str, Any], ckpt_dir: str | Path, filename: str = "last.pt") -> str:
    """지정한 경로에 PyTorch 체크포인트를 저장한다."""

    out_dir = Path(ckpt_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / filename
    torch.save(state, path)
    return str(path)
