"""Functional (pytree-in / pytree-out) optimizer steps.

These operate on arbitrary parameter pytrees (dicts of tensors, nested dicts,
etc.) without any in-place mutation, so they compose cleanly inside ``vmap`` /
``grad`` and inside :func:`ecliseutils.scan` loops (e.g. test-time training).
"""

from __future__ import annotations

from typing import Any

from torch.utils._pytree import tree_map


__all__ = ["sgd_step", "apply_updates"]


def apply_updates(params: Any, updates: Any) -> Any:
    """Add ``updates`` to ``params`` leaf-wise (the generic update seam).

    ``updates`` must share ``params``' pytree structure. This is the hook to
    build momentum / projected / quantized optimizers on top of: compute an
    update pytree however you like, then apply it here.
    """
    return tree_map(lambda p, u: p + u, params, updates)


def sgd_step(params: Any, grads: Any, lr: float) -> Any:
    """Plain gradient-descent step: ``params - lr * grads``, leaf-wise.

    ``grads`` must share ``params``' pytree structure (e.g. the output of
    ``torch.func.grad`` over a dict of parameters).
    """
    return tree_map(lambda p, g: p - lr * g, params, grads)
