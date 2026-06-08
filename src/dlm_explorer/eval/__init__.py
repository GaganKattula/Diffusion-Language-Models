"""Evaluation tasks and metrics.

A Task supplies examples (prompt + ground-truth) and a scorer. Each example is a
single decode + score, so an evaluation is cheap (inference only) — which is the
whole reason the agent can afford hundreds of search iterations.
"""

from __future__ import annotations

from ..registry import Registry

# Registry must exist before tasks.py imports it (avoids a circular import).
TASKS: Registry = Registry("tasks")


def build_task(name: str, **kwargs):
    return TASKS.get(name)(**kwargs)


from .tasks import Example, Task  # noqa: E402  (self-registration + re-export)

__all__ = ["Task", "Example", "TASKS", "build_task"]
