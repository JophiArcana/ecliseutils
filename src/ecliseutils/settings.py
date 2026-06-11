"""Configurable runtime settings shared across projects.

Unlike a project-local ``settings`` module, this one does **not** hardcode
project paths and does **not** mutate torch global defaults purely on import.
Instead, each consuming project calls :func:`configure` early (e.g. from its own
thin ``settings.py``) with its preferred device/dtype/precision/seed.

Module-level globals (``DEVICE``, ``DTYPE``, ``PRECISION``, ``SEED``, ``DEBUG``)
hold the currently-active configuration. Other ``ecliseutils`` modules read
these dynamically as ``settings.<NAME>`` so that a later :func:`configure` call
is reflected everywhere.
"""

from __future__ import annotations

import os
from typing import Optional, Union

import torch


__all__ = [
    "DEVICE",
    "DTYPE",
    "PRECISION",
    "SEED",
    "DEBUG",
    "configure",
    "set_debug",
    "register_safe_globals",
    "default_dtype",
]


# Neutral defaults. Projects override these via ``configure``.
DEVICE: str = "cpu"
DTYPE: torch.dtype = torch.get_default_dtype()
PRECISION: int = 8
SEED: Optional[int] = None
DEBUG: bool = False


def configure(
        device: Optional[str] = None,
        dtype: Optional[torch.dtype] = None,
        precision: Optional[int] = None,
        seed: Optional[int] = None,
        debug: Optional[bool] = None,
        set_torch_defaults: bool = True,
        safe_globals: bool = True,
) -> None:
    """Apply shared runtime configuration.

    Any argument left as ``None`` leaves the corresponding global unchanged.
    When ``set_torch_defaults`` is true, torch/numpy/pandas global state
    (default device/dtype, print options, autograd anomaly) is updated to match.
    """
    global DEVICE, DTYPE, PRECISION, SEED, DEBUG

    if device is not None:
        DEVICE = device
    if dtype is not None:
        DTYPE = dtype
    if precision is not None:
        PRECISION = precision
    if seed is not None:
        SEED = seed
    if debug is not None:
        DEBUG = bool(debug)

    if set_torch_defaults:
        import numpy as np
        np.set_printoptions(precision=PRECISION)
        try:
            import pandas as pd
            pd.set_option("display.precision", PRECISION)
        except Exception:
            pass
        torch.set_printoptions(precision=PRECISION, sci_mode=False, linewidth=400)

        if DEVICE.startswith("cuda"):
            cuda_num = DEVICE.split(":", 1)[1] if ":" in DEVICE else "0"
            os.environ["CUDA_VISIBLE_DEVICES"] = cuda_num
        torch.set_default_device(DEVICE)
        torch.set_default_dtype(DTYPE)
        torch.autograd.set_detect_anomaly(DEBUG)

    if safe_globals:
        register_safe_globals()


def set_debug(flag: bool) -> None:
    """Toggle debug-only global side effects (e.g. autograd anomaly detection)."""
    global DEBUG
    DEBUG = bool(flag)
    torch.autograd.set_detect_anomaly(DEBUG)


def register_safe_globals() -> None:
    """Allow-list the numpy reconstruction primitives for ``weights_only`` loads.

    Works across numpy 1.x (``numpy.core``) and numpy 2.x (``numpy._core``).
    """
    add = getattr(torch.serialization, "add_safe_globals", None)
    if add is None:
        return
    import numpy

    candidates = []
    for mod_name in ("numpy._core.multiarray", "numpy.core.multiarray"):
        try:
            mod = __import__(mod_name, fromlist=["_reconstruct"])
            candidates.append(mod._reconstruct)
            break
        except Exception:
            continue
    candidates.extend([numpy.dtype, numpy.ndarray])
    try:
        add(candidates)
    except Exception:
        pass


class default_dtype:
    """Context manager that temporarily sets torch's default dtype."""

    def __init__(self, dtype: Union[torch.dtype, None]):
        self.dtype = dtype

    def __enter__(self):
        if self.dtype is not None:
            self._original_dtype = torch.get_default_dtype()
            torch.set_default_dtype(self.dtype)

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.dtype is not None:
            torch.set_default_dtype(self._original_dtype)
