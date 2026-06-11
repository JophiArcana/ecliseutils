"""Batched ODE integration helpers.

``batch_odeint`` wraps ``torchdiffeq`` with a time-rescaling augmentation so a
single integration call handles a batch of independent time grids. ``torchdiffeq``
is imported lazily so importing this module (and ``ecliseutils``) does not require
it unless ``batch_odeint`` is actually used.
"""

from __future__ import annotations

from typing import Callable

import torch
import torch.nn as nn

from .modules import multi_vmap


__all__ = ["linspace", "geomspace", "batch_odeint"]


def linspace(a: torch.Tensor, b: torch.Tensor, n: int) -> torch.Tensor:
    if not torch.is_tensor(a):
        a = torch.tensor(a)
    if not torch.is_tensor(b):
        b = torch.tensor(b)
    a, b = torch.broadcast_tensors(a, b)
    return multi_vmap(torch.linspace, a.ndim, in_dims=(0, 0, None,), out_dims=0)(a, b, n)


def geomspace(a: torch.Tensor, b: torch.Tensor, n: int) -> torch.Tensor:
    if not torch.is_tensor(a):
        a = torch.tensor(a)
    if not torch.is_tensor(b):
        b = torch.tensor(b)
    return torch.exp(linspace(torch.log(a), torch.log(b), n))


class _AugmentedModule(nn.Module):
    def __init__(self, t: torch.Tensor, module: nn.Module) -> None:
        nn.Module.__init__(self)
        self.time_scale = torch.diff(t, dim=-1)
        self.module = module
        self.bsz = t.shape[:-1]

    def forward(self, t: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        t_scale = self.time_scale[..., min(int(t.item()), self.time_scale.shape[-1] - 1)]
        vz = self.module(z[..., 0], z[..., 1:])
        return t_scale[..., None] * torch.cat((torch.ones(self.bsz + (1,)), vz,), dim=-1)


def batch_odeint(
    fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    y: torch.Tensor,
    t: torch.Tensor,
    adjoint: bool,
    **kwargs,
) -> torch.Tensor:
    import torchdiffeq

    augmented_fn = _AugmentedModule(t, fn)
    _y = torch.cat((t[..., 0, None], y,), dim=-1)
    _t = torch.arange(0, t.shape[-1], dtype=torch.float)

    if adjoint:
        odeint_func = torchdiffeq.odeint_adjoint
    else:
        odeint_func = torchdiffeq.odeint

    _out = odeint_func(augmented_fn, _y, _t, **kwargs)
    out = _out[..., 1:]

    return out
