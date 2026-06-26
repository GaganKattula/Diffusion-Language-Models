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
from .base import MaskedDiffusionBackend, StepPrediction

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
        # Token used to right-pad batched sequences (attention-masked out anyway).
        self.pad_token_id = (
            self.tokenizer.pad_token_id
            if self.tokenizer.pad_token_id is not None
            else (self.tokenizer.eos_token_id or 0)
        )

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

    def predict_batch(self, input_ids, prompt_lens, valid_lens, temperature=0.0, seed=0, step=0):
        """Batched forward with the reduction done on-device.

        One forward over (B, L); softmax/top-2/entropy/gather run on the GPU and
        only the (B, L) summaries cross to host — so the (B, L, V) logits never
        leave the device. Right-padding preserves real-token positions; pad
        columns are attention-masked out.
        """
        torch = self._torch
        B, L = input_ids.shape
        with torch.no_grad():
            t = torch.tensor(input_ids, dtype=torch.long, device=self.device)
            attn = torch.zeros((B, L), dtype=torch.long, device=self.device)
            for b in range(B):
                attn[b, : valid_lens[b]] = 1
            logits = self.model(t, attention_mask=attn).logits.float()  # (B, L, V)
            logp = torch.log_softmax(logits, dim=-1)
            p = logp.exp()
            top2 = torch.topk(p, 2, dim=-1).values
            conf = top2[..., 0]
            margin = top2[..., 0] - top2[..., 1]
            neg_ent = (p * logp).sum(-1)  # sum p log p  ( = -entropy )
            cur = p.gather(-1, t.unsqueeze(-1)).squeeze(-1)
            if temperature > 0:
                g = torch.Generator(device=self.device)
                g.manual_seed(int(seed) * 100003 + int(step))
                pt = torch.softmax(logits / temperature, dim=-1)
                tok = torch.multinomial(
                    pt.view(-1, pt.shape[-1]), 1, generator=g
                ).view(B, L)
            else:
                tok = logits.argmax(-1)
            return StepPrediction(
                tok.cpu().numpy().astype(np.int64),
                conf.cpu().numpy().astype(np.float32),
                margin.cpu().numpy().astype(np.float32),
                neg_ent.cpu().numpy().astype(np.float32),
                cur.cpu().numpy().astype(np.float32),
            )

    def describe(self) -> str:
        return f"LladaBackend(model={self.model_name}, mask={self.mask_token_id})"
