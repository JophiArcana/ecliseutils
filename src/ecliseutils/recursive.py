"""Recursive (dotted-path) attribute and dict item access helpers."""

from __future__ import annotations

import functools
from types import SimpleNamespace
from typing import Any


__all__ = ["rgetattr", "rsetattr", "rhasattr", "rgetitem", "rsetitem"]


def rgetattr(obj: object, attr: str, *args):
    def _getattr(obj: object, attr: str) -> Any:
        return getattr(obj, attr, *args)
    return functools.reduce(_getattr, [obj] + attr.split("."))


def rsetattr(obj: object, attr: str, value: Any) -> None:
    def _rsetattr(obj: object, attrs: list[str], value: Any) -> None:
        if len(attrs) == 1:
            setattr(obj, attrs[0], value)
        else:
            _rsetattr(next_obj := getattr(obj, attrs[0], SimpleNamespace()), attrs[1:], value)
            setattr(obj, attrs[0], next_obj)
    _rsetattr(obj, attr.split("."), value)


def rhasattr(obj: object, attr: str) -> bool:
    try:
        rgetattr(obj, attr)
        return True
    except AttributeError:
        return False


def rgetitem(obj: dict[str, Any], item: str, *args):
    def _getitem(obj: dict[str, Any], item: str) -> Any:
        return obj.get(item, *args)
    return functools.reduce(_getitem, [obj] + item.split("."))


def rsetitem(obj: dict[str, Any], item: str, value: Any) -> None:
    def _rsetitem(obj: dict[str, Any], items: list[str], value: Any) -> None:
        if len(items) == 1:
            obj[items[0]] = value
        else:
            _rsetitem(next_obj := obj.get(items[0], {}), items[1:], value)
            obj[items[0]] = next_obj
    _rsetitem(obj, item.split("."), value)
