"""Shared type aliases."""

from __future__ import annotations

from typing import TypeVar

import torch.nn as nn
from tensordict import TensorDict


__all__ = ["ModelPair", "T"]


# ``(reference_module, stacked_params)``: the on-the-wire representation of a
# vmapped module ensemble.
ModelPair = tuple[nn.Module, TensorDict]

# Generic type variable reused by container helpers.
T = TypeVar("T")
