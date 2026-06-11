"""Dict / nested-structure and function-call helpers."""

from __future__ import annotations

import hashlib
import inspect
from typing import Any, Callable


__all__ = [
    "flatten_nested_dict",
    "map_dict",
    "nested_type",
    "print_dict",
    "call_func_with_kwargs",
    "hash_hex",
]


def hash_hex(s: str) -> str:
    """Short (8 hex char) SHA-256 digest of a string. Handy for config-hash IDs."""
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:8]


def flatten_nested_dict(d: dict[str, Any]) -> dict[str, Any]:
    result = {}

    def _flatten_nested_dict(s: tuple[str, ...], d: dict[str, Any]) -> None:
        for k, v in d.items():
            if isinstance(v, dict):
                _flatten_nested_dict((*s, k), v)
            else:
                result[".".join((*s, k))] = v
    _flatten_nested_dict((), d)
    return result


def map_dict(d: dict[str, Any], func: Callable[[Any], Any]) -> dict[str, Any]:
    return {
        k: map_dict(v, func) if hasattr(v, "items") else func(v)
        for k, v in d.items()
    }


def nested_type(o: object) -> object:
    if type(o) in [list, tuple]:
        return type(o)(map(nested_type, o))
    elif type(o) == dict:
        return {k: nested_type(v) for k, v in o.items()}
    else:
        return type(o)


def print_dict(d: "dict[str, Any] | object", n: int = 0, indent: int = 4) -> None:
    if isinstance(d, dict):
        for k, v in d.items():
            print(" " * (n * indent) + k)
            print_dict(v, n=n + 1, indent=indent)
    else:
        to_print = str(d)
        print("\n".join([" " * (n * indent) + s for s in to_print.split("\n")]))


def call_func_with_kwargs(func: Callable, args: tuple[Any, ...], kwargs: dict[str, Any]):
    params = inspect.signature(func).parameters
    required_args = [
        kwargs[k] if k in kwargs else args[i] for i, (k, v) in enumerate(params.items())
        if v.kind is inspect.Parameter.POSITIONAL_OR_KEYWORD and v.default is inspect.Parameter.empty
    ]
    additional_args = args[len(required_args):]

    allow_var_keywords = any(v.kind is inspect.Parameter.VAR_KEYWORD for v in params.values())
    valid_kwargs = {
        k: v for k, v in kwargs.items()
        if ((params[k].default is not inspect.Parameter.empty) if k in params else allow_var_keywords)
    }
    return func(*required_args, *additional_args, **valid_kwargs)
