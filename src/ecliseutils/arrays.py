"""NumPy object-array comprehensions and ``LabeledArray`` alignment helpers."""

from __future__ import annotations

from collections import OrderedDict
from typing import Any, Callable, Iterable, Iterator, Sequence

import numpy as np

from .labeled_array import LabeledArray, LabeledDataset, array_of, as_ndarray


__all__ = [
    "array_of",
    "multi_iter",
    "multi_enumerate",
    "multi_map",
    "multi_zip",
    "dim_array_like",
    "broadcast_dim_array_shapes",
    "broadcast_dim_arrays",
    "take_from_dim_array",
]


def multi_iter(arr: "np.ndarray | LabeledArray") -> Iterable[Any]:
    for x in np.nditer(as_ndarray(arr), flags=["refs_ok"]):
        yield x[()]


def multi_enumerate(arr: "np.ndarray | LabeledArray") -> Iterable[tuple[Sequence[int], Any]]:
    it = np.nditer(as_ndarray(arr), flags=["multi_index", "refs_ok"])
    for x in it:
        yield it.multi_index, x[()]


def multi_map(func: Callable[[Any], Any], arr: "np.ndarray | LabeledArray", dtype: type = None):
    base = as_ndarray(arr)
    # Apply ``func`` exactly once per element (the old dtype-inference path called it
    # twice on the first element).
    computed = [(idx, func(x)) for idx, x in multi_enumerate(base)]
    if dtype is None:
        dtype = type(computed[0][1])
    result = np.empty_like(base, dtype=dtype)
    for idx, value in computed:
        result[idx] = value
    return LabeledArray(result, arr.dims) if isinstance(arr, LabeledArray) else result


def multi_zip(*arrs: "np.ndarray | LabeledArray") -> np.ndarray:
    """Zip element-wise into an object array of tuples.

    Unlike a structured recarray, an object array of plain tuples never lets
    numpy introspect ``.dtype``/``.names`` on the elements, so it is safe to
    hold ``Tensor``/``TensorDict`` cells.
    """
    bases = [as_ndarray(arr) for arr in arrs]
    result = np.empty(bases[0].shape, dtype=object)
    for idx in np.ndindex(bases[0].shape):
        result[idx] = tuple(base[idx] for base in bases)
    return result


def dim_array_like(arr: LabeledArray, dtype: type) -> LabeledArray:
    empty_arr = np.full_like(as_ndarray(arr), None, dtype=dtype)
    return LabeledArray(empty_arr, arr.dims)


def broadcast_dim_array_shapes(*dim_arrs: Iterable[LabeledArray]) -> "OrderedDict[str, int]":
    dim_dict = OrderedDict()
    for dim_arr in dim_arrs:
        for dim_name, dim_len in zip(dim_arr.dims, dim_arr.shape):
            dim_dict.setdefault(dim_name, []).append(dim_len)
    return OrderedDict((k, np.broadcast_shapes(*v)[0]) for k, v in dim_dict.items())


def broadcast_dim_arrays(*dim_arrs: Iterable[np.ndarray]) -> Iterator[LabeledArray]:
    _dim_arrs = []
    for dim_arr in dim_arrs:
        if isinstance(dim_arr, LabeledArray):
            _dim_arrs.append(dim_arr)
        elif isinstance(dim_arr, np.ndarray):
            assert dim_arr.ndim == 0
            _dim_arrs.append(LabeledArray(dim_arr, ()))
        else:
            _dim_arrs.append(LabeledArray(array_of(dim_arr), ()))

    dim_dict = broadcast_dim_array_shapes(*_dim_arrs)
    return (dim_arr.broadcast(dim_dict) for dim_arr in _dim_arrs)


def take_from_dim_array(dim_arr: "LabeledArray | LabeledDataset", idx: dict[str, Any]):
    return dim_arr.take(indices=idx)
