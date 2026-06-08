import numpy as np

from dlm_explorer.backends import build_backend


def test_mock_backend_shapes_and_determinism():
    b = build_backend("mock", vocab_size=32)
    seq = np.array([2, 3, 4, 5, b.mask_token_id, b.mask_token_id, b.mask_token_id, b.mask_token_id])
    l1 = b.logits(seq, prompt_len=4)
    l2 = b.logits(seq, prompt_len=4)
    assert l1.shape == (8, 32)
    assert np.allclose(l1, l2), "mock backend must be deterministic"


def test_mock_prefers_copy_target_with_context():
    # With both neighbours committed correctly, the target should dominate.
    b = build_backend("mock")
    # prompt = [2,3,4]; gen positions copy prompt; commit neighbours of middle pos.
    seq = np.array([2, 3, 4, 2, b.mask_token_id, 4])  # gen middle (idx4) masked, neighbours correct
    logits = b.logits(seq, prompt_len=3)
    assert int(np.argmax(logits[4])) == 3  # target for gen[1] is prompt[1] == 3
