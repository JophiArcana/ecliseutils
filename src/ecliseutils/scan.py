"""Eager, pytree-aware sequential scan (a ``jax.lax.scan`` analogue).

Unlike the parallel linear scans in :mod:`ecliseutils.fast_conv_scan`
(``conv_scan`` / ``dense_linear_scan``), this is a plain Python loop: it threads
an arbitrary ``carry`` pytree through ``fn`` step by step, which is exactly what
nonlinear / state-dependent recurrences (e.g. online weight updates, test-time
training) need. It is transparent to ``torch.func`` transforms (``vmap`` /
``grad``) because it only does tensor indexing, ``tree_map``, and ``torch.stack``.
"""

from __future__ import annotations

from typing import Any, Callable, Optional, Tuple

import torch
from torch.utils._pytree import tree_flatten, tree_map, tree_unflatten


__all__ = ["scan"]


def scan(
        fn: Callable[[Any, Any], Tuple[Any, Any]],
        init: Any,
        xs: Any = None,
        length: Optional[int] = None,
) -> Tuple[Any, Any]:
    """Apply ``fn`` over a leading ("time") axis, threading a carry.

    ``fn(carry, x) -> (carry, y)`` is called for each step. ``xs`` is any pytree
    whose leaves share a leading dimension of size ``length``; at step ``t`` each
    leaf is indexed at ``t`` to build ``x``. The collected ``y`` pytrees are
    stacked along a new leading dimension. Returns ``(final_carry, stacked_ys)``.

    If ``xs`` is ``None``, ``length`` must be given and ``x`` is ``None`` at every
    step. For an empty scan (``length == 0``) the stacked outputs are ``None``
    (there is nothing to infer an output structure from).
    """
    if xs is None and length is None:
        raise ValueError("scan requires either `xs` or `length` to be specified.")

    if xs is not None:
        leaves, _ = tree_flatten(xs)
        if len(leaves) > 0:
            inferred = leaves[0].shape[0]
            if length is not None and length != inferred:
                raise ValueError(
                    f"`length` ({length}) does not match the leading axis of `xs` ({inferred})."
                )
            length = inferred
        elif length is None:
            raise ValueError("scan received an empty `xs` and no `length`.")

    carry = init
    ys: list[Any] = []
    for t in range(length):
        x = None if xs is None else tree_map(lambda leaf: leaf[t], xs)
        carry, y = fn(carry, x)
        ys.append(y)

    if length == 0:
        return carry, None

    flat_ys = [tree_flatten(y)[0] for y in ys]
    _, out_spec = tree_flatten(ys[0])
    stacked = [torch.stack(components, dim=0) for components in zip(*flat_ys)]
    return carry, tree_unflatten(stacked, out_spec)
