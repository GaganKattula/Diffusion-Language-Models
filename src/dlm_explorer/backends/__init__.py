"""Model backends: anything that can score a (partially masked) token sequence.

A backend is the *only* thing that knows about model weights. The decoding
strategy (dlm_explorer.decoding.sampler) drives the iterative denoising loop on
top of whatever logits the backend returns, which is exactly what makes the
decoding space searchable independently of the model.
"""

from __future__ import annotations

from ..registry import Registry
from .base import MaskedDiffusionBackend

BACKENDS: Registry[type[MaskedDiffusionBackend]] = Registry("backends")


def build_backend(name: str, **kwargs) -> MaskedDiffusionBackend:
    return BACKENDS.get(name)(**kwargs)


# Import implementations so they self-register. The LLaDA backend imports torch
# lazily inside __init__, so importing this module never requires torch.
from . import mock  # noqa: E402,F401
from . import llada  # noqa: E402,F401

__all__ = ["MaskedDiffusionBackend", "BACKENDS", "build_backend"]
