"""Timing, stdout-suppression, and lightweight call-tracking helpers."""

from __future__ import annotations

import os
import sys
import time
import inspect
from functools import wraps
from typing import Any


__all__ = ["Timer", "identity", "PTR", "print_disabled", "print_enabled", "track_calls"]


class Timer:
    def __init__(self):
        self.t = time.perf_counter()

    def reset(self):
        t = time.perf_counter()
        out = t - self.t
        self.t = t
        return out


def identity(x: Any) -> Any:
    return x


class PTR(object):
    """A trivial single-element container (yields its wrapped object once)."""

    def __init__(self, obj: object) -> None:
        self.obj = obj

    def __iter__(self):
        yield self.obj


class print_disabled:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, "w")

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


class print_enabled:
    def __init__(self, enabled: bool):
        self.enabled = enabled

    def __enter__(self):
        if not self.enabled:
            self._original_stdout = sys.stdout
            sys.stdout = open(os.devnull, "w")

    def __exit__(self, exc_type, exc_val, exc_tb):
        if not self.enabled:
            sys.stdout.close()
            sys.stdout = self._original_stdout


def track_calls(desc=None, unit="call", log_args=None):
    """Decorator that wraps a function with a ``tqdm`` progress bar counting calls."""
    from tqdm import tqdm

    def decorator(func):
        sig = inspect.signature(func)
        params = list(sig.parameters.keys())
        pbar = tqdm(desc=desc or func.__name__, unit=unit)

        @wraps(func)
        def wrapper(*args, **kwargs):
            result = func(*args, **kwargs)
            pbar.update(1)

            if log_args:
                postfix = {}
                for arg in log_args:
                    if isinstance(arg, int):
                        postfix[f"arg[{arg}]"] = args[arg] if arg < len(args) else "?"
                    elif arg in kwargs:
                        postfix[arg] = kwargs[arg]
                    elif arg in params:
                        idx = params.index(arg)
                        postfix[arg] = args[idx] if idx < len(args) else "?"
                pbar.set_postfix(postfix)

            return result

        wrapper.close = pbar.close
        return wrapper
    return decorator
