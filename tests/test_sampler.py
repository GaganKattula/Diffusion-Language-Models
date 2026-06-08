"""Behavioural tests: the mock pipeline must reproduce the qualitative effects
that make the decoding search meaningful."""

from statistics import mean

from dlm_explorer.backends import build_backend
from dlm_explorer.decoding import DecodingParams, MaskedDiffusionSampler
from dlm_explorer.eval import build_task


def _run(params, n=24):
    backend = build_backend("mock")
    task = build_task("copy", n_examples=n, seq_len=12)
    sampler = MaskedDiffusionSampler(backend)
    accs, calls, corr = [], [], []
    for ex in task.examples():
        r = sampler.generate(ex.prompt_ids, ex.gen_len, params)
        accs.append(task.score(r.gen_tokens, ex)["accuracy"])
        calls.append(r.num_model_calls)
        corr.append(r.num_self_corrections)
    return mean(accs), mean(calls), mean(corr)


def test_more_steps_helps():
    acc1, calls1, _ = _run(DecodingParams(num_steps=1, unmask_order="confidence"))
    acc16, calls16, _ = _run(DecodingParams(num_steps=16, unmask_order="confidence"))
    assert acc16 > acc1
    assert calls16 > calls1  # more steps => more compute (the Pareto tradeoff)


def test_confidence_order_beats_random():
    acc_conf, _, _ = _run(DecodingParams(num_steps=8, unmask_order="confidence"))
    acc_rand, _, _ = _run(DecodingParams(num_steps=8, unmask_order="random"))
    assert acc_conf >= acc_rand


def test_self_correction_helps_and_is_counted():
    # With a weak unmasking order (left_to_right commits early positions before
    # their right-context exists), self-correction recovers accuracy — the
    # structural move autoregression cannot make. A strong order (confidence)
    # avoids the mistakes in the first place, so it has nothing to revise; that
    # is why we probe the weak-order regime here.
    acc_off, _, corr_off = _run(DecodingParams(num_steps=6, unmask_order="left_to_right", remask=False))
    acc_on, _, corr_on = _run(DecodingParams(num_steps=6, unmask_order="left_to_right", remask=True))
    assert corr_off == 0.0                 # no remask => no self-corrections
    assert corr_on > 0.0                   # remask => positions get revised
    assert acc_on > acc_off                # and the revisions improve accuracy


def test_no_masks_remain():
    backend = build_backend("mock")
    sampler = MaskedDiffusionSampler(backend)
    r = sampler.generate([2, 3, 4, 5, 6], gen_len=5, params=DecodingParams(num_steps=3, remask=True))
    assert backend.mask_token_id not in r.gen_tokens
