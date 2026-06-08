"""Real masked-diffusion backend backed by an open LLaDA checkpoint.

GATED: this backend requires `torch` + `transformers`, a downloaded checkpoint,
and (realistically) a GPU. It is wired so the search machinery is identical
whether you run the mock or the real model — only the `--backend` changes — but
it is NOT exercised until Phase 2 (see docs/05_roadmap.md). Importing this module
does not import torch; the heavy imports happen inside __init__.

Reference checkpoints (Apache-2.0 / open):
  * GSAI-ML/LLaDA-8B-Base
  * GSAI-ML/LLaDA-8B-Instruct
LLaDA uses a dedicated mask token id (126336 at time of writing). Confirm against
the checkpoint's config before a real run.

This wrapper deliberately exposes ONLY `logits(input_ids, prompt_len)`. The
denoising loop — steps, unmask order, remasking, blocks — is owned by
dlm_explorer.decoding.sampler, so the exact same search applies to mock and
LLaDA alike.
"""

from __future__ import annotations

import numpy as np

from . import BACKENDS
from .base import MaskedDiffusionBackend

DEFAULT_MODEL = "GSAI-ML/LLaDA-8B-Base"
DEFAULT_MASK_TOKEN_ID = 126336


@BACKENDS.register("llada")
class LladaBackend(MaskedDiffusionBackend):
    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        mask_token_id: int = DEFAULT_MASK_TOKEN_ID,
        device: str = "cuda",
        dtype: str = "bfloat16",
    ) -> None:
        try:
            import torch  # noqa: F401
            from transformers import AutoModel, AutoTokenizer
        except ImportError as e:  # pragma: no cover - environment dependent
            raise ImportError(
                "The LLaDA backend needs `torch` and `transformers`.\n"
                "  pip install torch transformers\n"
                "and a CUDA-capable GPU. Use --backend mock for offline work."
            ) from e

        import torch

        self._torch = torch
        self.model_name = model_name
        self.mask_token_id = mask_token_id
        self.device = device
        self._dtype = getattr(torch, dtype)

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(
            model_name, trust_remote_code=True, torch_dtype=self._dtype
        ).to(device).eval()
        self.vocab_size = int(self.model.config.vocab_size)

    def encode(self, text: str) -> list[int]:
        return self.tokenizer(text)["input_ids"]

    def decode(self, ids: list[int]) -> str:
        return self.tokenizer.decode(
            [i for i in ids if i != self.mask_token_id], skip_special_tokens=True
        )

    def logits(self, input_ids: np.ndarray, prompt_len: int) -> np.ndarray:
        torch = self._torch
        with torch.no_grad():
            t = torch.tensor(input_ids[None, :], dtype=torch.long, device=self.device)
            out = self.model(t).logits  # (1, L, V)
            return out[0].float().cpu().numpy()

    def describe(self) -> str:
        return f"LladaBackend(model={self.model_name}, mask={self.mask_token_id})"
