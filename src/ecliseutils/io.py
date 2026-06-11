"""I/O, RNG, and memory-management helpers."""

from __future__ import annotations

import gc
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from . import settings
from .labeled_array import array_of


__all__ = ["torch_load", "empty_cache", "reset_seed", "model_size", "array_of"]


def torch_load(f: "Any", **kwargs) -> Any:
    return torch.load(f, map_location=settings.DEVICE, weights_only=False)


def empty_cache():
    gc.collect()
    torch.cuda.empty_cache()


def reset_seed():
    if settings.SEED is None:
        torch.seed()
        np.random.seed()
    else:
        torch.manual_seed(settings.SEED)
        np.random.seed(settings.SEED)


def model_size(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())
