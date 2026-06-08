"""A tiny string->object registry used for backends, tasks, and proposers.

Lets configs refer to components by name (e.g. backend: "mock") without the
config layer importing every implementation directly.
"""

from __future__ import annotations

from typing import Callable, Dict, Generic, TypeVar

T = TypeVar("T")


class Registry(Generic[T]):
    def __init__(self, name: str) -> None:
        self.name = name
        self._items: Dict[str, T] = {}

    def register(self, key: str) -> Callable[[T], T]:
        def decorator(obj: T) -> T:
            if key in self._items:
                raise KeyError(f"{self.name!r} already has an entry for {key!r}")
            self._items[key] = obj
            return obj

        return decorator

    def get(self, key: str) -> T:
        if key not in self._items:
            raise KeyError(
                f"{self.name!r} has no entry {key!r}. Known: {sorted(self._items)}"
            )
        return self._items[key]

    def keys(self) -> list[str]:
        return sorted(self._items)
