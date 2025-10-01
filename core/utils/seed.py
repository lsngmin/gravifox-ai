"""시드 고정 도우미 모듈."""

from __future__ import annotations

import os
import random
from typing import Optional

import numpy as np
import torch


def set_seed(seed: Optional[int]) -> None:
    """모든 주요 난수 발생기의 시드를 고정한다."""

    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
