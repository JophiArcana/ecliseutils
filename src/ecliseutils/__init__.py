"""ecliseutils: shared general-purpose research utilities.

Submodules group the utilities by concern (``modules``, ``arrays``, ``linalg``,
``are``, ``ode``, ``labeled_array``, ``ensemble``, ``fast_conv_scan``,
``recursive``, ``dicts``, ``io``, ``memory``, ``timing``, ``plotting``,
``settings``, ``types``). The most commonly used names are re-exported here for
convenience (``import ecliseutils as eu; eu.stack_module_arr(...)``).

Projects should call :func:`ecliseutils.configure` early (typically from their
own ``settings`` module) to set device/dtype/precision/seed.
"""

from __future__ import annotations

from . import (
    arrays,
    are,
    dicts,
    ensemble,
    fast_conv_scan,
    io,
    labeled_array,
    linalg,
    memory,
    modules,
    ode,
    optim,
    plotting,
    recursive,
    settings,
    timing,
    types,
)

from .settings import configure, set_debug, register_safe_globals, default_dtype
from .types import ModelPair

from .labeled_array import LabeledArray, LabeledDataset, array_of, put_object, as_ndarray
from .modules import (
    stack_tensor_arr,
    stack_module_arr,
    stack_module_arr_preserve_reference,
    run_module_arr,
    multi_vmap,
    FunctionalMethod,
    buffer_dict,
    td_items,
    td_get,
    parameter_td,
    mask_dataset_with_total_sequence_length,
    broadcast_shapes,
    get_all_hooks,
)
from .arrays import (
    multi_iter,
    multi_enumerate,
    multi_map,
    multi_zip,
    dim_array_like,
    broadcast_dim_array_shapes,
    broadcast_dim_arrays,
    take_from_dim_array,
)
from .linalg import (
    pow_series,
    batch_trace,
    kl_div,
    sqrtm,
    complex,
    ceildiv,
    ceil,
    T,
    hadamard_conjugation,
    hadamard_conjugation_diff_order1,
    hadamard_conjugation_diff_order2,
    inverse,
    eig_some,
)
from .are import (
    solve_discrete_are,
    solve_continuous_are,
    test_discrete_are,
    test_continuous_are,
)
from .ode import linspace, geomspace, batch_odeint
from .ensemble import EnsembleModule, DEFAULT_SPLIT_SIZE
from .fast_conv_scan import ConvScanFn, conv_scan, DenseLinearScanFn, dense_linear_scan
from .scan import scan
from .optim import sgd_step, apply_updates
from .recursive import rgetattr, rsetattr, rhasattr, rgetitem, rsetitem
from .dicts import flatten_nested_dict, map_dict, nested_type, print_dict, call_func_with_kwargs, hash_hex
from .io import torch_load, empty_cache, reset_seed, model_size
from .memory import (
    get_tensors_in_memory,
    print_tensors_in_memory,
    get_tensors_in_memory_shape,
    track_tensor_diff,
)
from .timing import Timer, identity, PTR, print_disabled, print_enabled, track_calls
from .plotting import color, confidence_ellipse


__version__ = "0.1.2"
