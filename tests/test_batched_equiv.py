"""Equivalence: batched decoding must exactly match the serial sampler on the
mock backend for deterministic configs. This is the offline correctness proof
for the batched path before it runs on a GPU."""

import itertools

import pytest

from dlm_explorer.backends import build_backend
from dlm_explorer.decoding import DecodingParams, MaskedDiffusionSampler
from dlm_explorer.decoding.batched import BatchedSampler

DETERMINISTIC_ORDERS = ["confidence", "margin", "entropy", "left_to_right"]
CONFIGS = [
    DecodingParams(num_steps=ns, unmask_order=o, remask=rm, block_size=bs)
    for ns, o, rm, bs in itertools.product(
        [4, 8], DETERMINISTIC_ORDERS, [False, True], [None, 5]
    )
]


def _prompts_equal_len():
    # equal-length prompts => no padding in the batch
    return [[2, 3, 4, 5, 6, 7, 8, 9, 10, 11][:10] for _ in range(5)], 10


def _prompts_varying_len():
    return [[2, 3, 4, 5], [2, 3, 4, 5, 6, 7], [2, 3, 4, 5, 6, 7, 8, 9]], 6


@pytest.mark.parametrize("params", CONFIGS)
def test_batched_matches_serial_equal_len(params):
    backend = build_backend("mock")
    prompts, gen_len = _prompts_equal_len()
    serial = MaskedDiffusionSampler(backend)
    batched = BatchedSampler(backend)

    ser = [serial.generate(p, gen_len, params) for p in prompts]
    bat = batched.generate_batch(prompts, gen_len, params)

    for s, b in zip(ser, bat):
        assert s.gen_tokens == b.gen_tokens, params
        assert s.num_model_calls == b.num_model_calls, params
        assert s.num_self_corrections == b.num_self_corrections, params


@pytest.mark.parametrize("params", CONFIGS)
def test_batched_matches_serial_varying_len(params):
    # Right-padding must preserve each row's result even with mixed prompt lengths.
    backend = build_backend("mock")
    prompts, gen_len = _prompts_varying_len()
    serial = MaskedDiffusionSampler(backend)
    batched = BatchedSampler(backend)

    ser = [serial.generate(p, gen_len, params) for p in prompts]
    bat = batched.generate_batch(prompts, gen_len, params)

    for s, b in zip(ser, bat):
        assert s.gen_tokens == b.gen_tokens, params
        assert s.num_self_corrections == b.num_self_corrections, params
