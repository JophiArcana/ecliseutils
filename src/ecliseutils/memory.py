"""Tensor-memory introspection / debugging helpers."""

from __future__ import annotations

import collections
import gc
from collections import OrderedDict
from typing import Iterator, Tuple

import numpy as np
import torch


__all__ = [
    "get_tensors_in_memory",
    "print_tensors_in_memory",
    "get_tensors_in_memory_shape",
    "track_tensor_diff",
]


def get_tensors_in_memory(allowed_classes: Tuple[type] = (torch.Tensor,)) -> "OrderedDict[int, torch.Tensor]":
    gc.collect()
    tensors = [
        obj for obj in gc.get_objects()
        if (type(obj) in allowed_classes) and (torch.is_tensor(obj) or torch.is_tensor(getattr(obj, "data", None)))
    ]
    indices = np.argsort([t.numel() for t in tensors])[::-1]
    result = collections.OrderedDict()
    for idx in indices:
        t = tensors[idx]
        result[t.data_ptr()] = t
    return result


def print_tensors_in_memory(allowed_classes: Tuple[type] = (torch.Tensor,)) -> None:
    t_dict = get_tensors_in_memory(allowed_classes=allowed_classes)
    for t in t_dict.values():
        print(type(t), t.size(), t.numel())


def get_tensors_in_memory_shape(allowed_classes: Tuple[type] = (torch.Tensor,)) -> "OrderedDict[int, torch.Size]":
    tensors = get_tensors_in_memory(allowed_classes)
    out = {k: v.shape for k, v in tensors.items()}
    del tensors
    gc.collect()
    return out


def track_tensor_diff(
    allowed_classes: Tuple[type] = (torch.Tensor,),
) -> Iterator[tuple[list[tuple[int, torch.Size]], list[tuple[int, torch.Size]]]]:
    prev_tensors = get_tensors_in_memory_shape(allowed_classes)
    while True:
        tensors = get_tensors_in_memory_shape(allowed_classes)

        p = [(k, v) for k, v in tensors.items() if k not in prev_tensors]
        n = [(k, v) for k, v in prev_tensors.items() if k not in tensors]
        prev_tensors = tensors

        yield (p, n)
