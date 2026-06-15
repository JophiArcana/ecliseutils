"""Module / tensor / TensorDict batching utilities.

Stack numpy object-arrays of ``nn.Module`` into a single ``ModelPair``
(``(reference_module, stacked_params)``) and run them with ``vmap`` (falling
back to a serial loop), plus assorted TensorDict helpers.
"""

from __future__ import annotations

import warnings
from types import MappingProxyType
from typing import Any, Callable, Sequence, Union

import numpy as np
import torch
import torch.nn as nn
from tensordict import TensorDict
from torch.utils._pytree import tree_flatten, tree_unflatten

from . import settings
from .labeled_array import as_ndarray
from .types import ModelPair


__all__ = [
    "stack_tensor_arr",
    "stack_module_arr",
    "stack_module_arr_preserve_reference",
    "run_module_arr",
    "multi_vmap",
    "FunctionalMethod",
    "buffer_dict",
    "td_items",
    "td_get",
    "parameter_td",
    "mask_dataset_with_total_sequence_length",
    "broadcast_shapes",
    "get_all_hooks",
]


class FunctionalMethod(nn.Module):
    """Expose an arbitrary *method* of ``module`` as this adapter's ``forward`` so
    it can be driven by :func:`torch.func.functional_call` with substituted
    parameters, without writing a bespoke wrapper subclass.

    The parameter / buffer / submodule containers are shared *by reference* with
    ``module``, so (a) the names ``functional_call`` sees are identical to
    ``module``'s (no wrapper prefix), and (b) the substitution performed by
    ``functional_call`` reaches ``module`` itself for the duration of the call --
    which is what lets a bound method that reads ``module.<attr>`` observe the
    substituted (fast-)weights.

    The wrapped module is stored off the module registry (``object.__setattr__``)
    so it is not itself a submodule of the adapter (no recursion / duplicate
    parameters).
    """

    def __init__(self, module: nn.Module, method_name: str = "forward"):
        super().__init__()
        object.__setattr__(self, "_src_module", module)
        object.__setattr__(self, "_method_name", method_name)
        # Share state containers by reference (see class docstring).
        object.__setattr__(self, "_parameters", module._parameters)
        object.__setattr__(self, "_buffers", module._buffers)
        object.__setattr__(self, "_modules", module._modules)

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        return getattr(self._src_module, self._method_name)(*args, **kwargs)


def stack_tensor_arr(tensor_arr: "np.ndarray", dim: int = 0) -> Union[torch.Tensor, TensorDict]:
    base = as_ndarray(tensor_arr)
    tensor_list = [*base.ravel()]
    if isinstance(t := tensor_list[0], torch.Tensor):
        result = torch.stack(tensor_list, dim=dim)
    else:
        result = TensorDict.maybe_dense_stack(tensor_list, dim=dim)
    return result.reshape((*base.shape, *t.shape,))


def stack_module_arr(module_arr: "np.ndarray") -> ModelPair:
    params, buffers = torch.func.stack_module_state(module_arr.ravel().tolist())
    td = TensorDict({}, batch_size=module_arr.shape)

    def _unflatten(t: torch.Tensor, dim: int, shape: tuple[int, ...]):
        if len(shape) == 0:
            return t.squeeze(dim=dim)
        elif len(shape) == 1:
            return t
        else:
            return t.unflatten(dim, shape)

    for k, v in params.items():
        td[(*k.split("."),)] = nn.Parameter(_unflatten(v, 0, module_arr.shape), requires_grad=v.requires_grad)
    for k, v in buffers.items():
        td[(*k.split("."),)] = _unflatten(v, 0, module_arr.shape)

    return module_arr.ravel()[0].to(settings.DEVICE), td.to(settings.DEVICE)


def stack_module_arr_preserve_reference(module_arr: "np.ndarray") -> ModelPair:
    flattened_td = TensorDict.maybe_dense_stack([
        TensorDict({
            k: v
            for k in dir(module) if isinstance((v := getattr(module, k)), torch.Tensor)
        }, batch_size=())
        for module in module_arr.ravel()
    ], dim=0)
    td = flattened_td.reshape(module_arr.shape)
    return module_arr.ravel()[0], td.to(settings.DEVICE)


def run_module_arr(
        model_pair: ModelPair,
        args: Any,  # Note: a TensorDict is only checked for as the immediate argument and will not work inside a nested structure
        kwargs: dict[str, Any] = MappingProxyType(dict()),
        vmap: bool = True,
        method: str = "forward",
) -> Any:
    if "TensorDict" in type(args).__name__:
        args = args.to_dict()

    reference_module, module_td = model_pair
    # Drive an arbitrary method (not just ``forward``) over the stacked params by
    # wrapping the reference module so ``functional_call`` targets that method,
    # keeping the stacked-param keys identical (see ``FunctionalMethod``).
    target = reference_module if method == "forward" else FunctionalMethod(reference_module, method)
    module_td = TensorDict(td_items(module_td), batch_size=module_td.shape)
    n = int(np.prod(module_td.shape))

    # vmap requires more than one stacked module; a single module always uses the
    # explicit per-module loop below. When vmap is requested but fails at runtime
    # we fall back to the loop, but warn loudly so genuine model bugs are not hidden.
    if vmap and n > 1:
        try:
            def vmap_run(module_d, ags):
                return torch.func.functional_call(target, module_d, ags, kwargs)
            vmap_run = multi_vmap(vmap_run, module_td.ndim, randomness="different")
            return vmap_run(module_td.to_dict(), args)
        except RuntimeError as e:
            warnings.warn(
                f"vmap execution of {type(reference_module).__name__} failed and is "
                f"falling back to a serial per-module loop (this is slower and may hide "
                f"a model bug): {e}",
                RuntimeWarning,
            )

    # Serial per-module fallback (single module, or vmap disabled / failed above).
    flat_args, args_spec = tree_flatten(args)
    single_flat_args_list = [
        [t.view(n, *t.shape[module_td.ndim:])[idx] for t in flat_args]
        for idx in range(n)
    ]
    single_args_list = [tree_unflatten(single_flat_args, args_spec) for single_flat_args in single_flat_args_list]

    single_out_list = [
        torch.func.functional_call(target, module_td.view(n)[idx].to_dict(), single_args)
        for idx, single_args in enumerate(single_args_list)
    ]
    _, out_spec = tree_flatten(single_out_list[0])
    single_flat_out_list = [tree_flatten(single_out)[0] for single_out in single_out_list]
    flat_out = [
        torch.stack([*out_component_list], dim=0).view(*module_td.shape, *out_component_list[0].shape)
        for out_component_list in zip(*single_flat_out_list)
    ]
    return tree_unflatten(flat_out, out_spec)


def multi_vmap(func: Callable, n: int, **kwargs: Any) -> Callable:
    f = func
    for _ in range(n):
        f = torch.vmap(f, **kwargs)
    return f


def buffer_dict(td: TensorDict) -> nn.Module:
    def _buffer_dict(parent_module: nn.Module, td: TensorDict) -> nn.Module:
        for k, v in td.items(include_nested=False):
            if isinstance(v, torch.Tensor):
                parent_module.register_buffer(k, v)
            else:
                parent_module.register_module(k, _buffer_dict(nn.Module(), v))
        return parent_module
    return _buffer_dict(nn.Module(), td)


def td_items(td: TensorDict) -> dict[str, torch.Tensor]:
    return {
        k if isinstance(k, str) else ".".join(k): v
        for k, v in td.items(include_nested=True, leaves_only=True)
    }


def td_get(d: TensorDict, keys: Sequence) -> TensorDict:
    return TensorDict({k: d[k] for k in keys}, batch_size=d.shape)


def parameter_td(m: nn.Module) -> TensorDict:
    result = TensorDict({}, batch_size=())
    for k, v in m.named_parameters():
        k_ = (*k.split("."),)
        result[k_[0] if len(k_) == 1 else k_] = v
    return result


def mask_dataset_with_total_sequence_length(ds: TensorDict, total_sequence_length: int) -> TensorDict:
    batch_size, sequence_length = ds.shape[-2:]
    ds["mask"] = torch.Tensor(torch.arange(batch_size * sequence_length) < total_sequence_length).view(
        sequence_length, batch_size
    ).mT.expand(ds.shape)
    return ds


def broadcast_shapes(*shapes: tuple[int, ...]):
    def to_tuple(shape: tuple[int, ...]) -> tuple[int, ...]:
        return (*map(int, shape),)
    return to_tuple(torch.broadcast_shapes(*map(to_tuple, shapes)))


def get_all_hooks(module: nn.Module) -> dict[str, dict[int, Callable]]:
    """Retrieve all forward/backward (and pre-) hooks from ``module`` and submodules."""
    from collections import OrderedDict

    all_hooks: dict[str, dict[int, Callable]] = {}

    def _get_hooks(m: nn.Module, prefix=""):
        hooks = {}
        if hasattr(m, "_forward_hooks") and m._forward_hooks != OrderedDict():
            hooks.update({"forward_hooks": m._forward_hooks})
        if hasattr(m, "_forward_pre_hooks") and m._forward_pre_hooks != OrderedDict():
            hooks.update({"forward_pre_hooks": m._forward_pre_hooks})
        if hasattr(m, "_backward_hooks") and m._backward_hooks != OrderedDict():
            hooks.update({"backward_hooks": m._backward_hooks})
        if hasattr(m, "_full_backward_hooks") and m._full_backward_hooks != OrderedDict():
            hooks.update({"full_backward_hooks": m._full_backward_hooks})
        if hooks:
            all_hooks[prefix] = hooks
        for name, child in m.named_children():
            _get_hooks(child, prefix=f"{prefix}.{name}" if prefix else name)

    _get_hooks(module)
    return all_hooks
