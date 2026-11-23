"""Independent numerical tests for attention masks and softmax attention."""

from __future__ import annotations

import numpy as np
import pytest

from applied_ml.attention import (
    attention,
    attention_weights,
    causal_mask,
    scaled_dot_product_scores,
    softmax,
)


def test_causal_mask_is_lower_triangular():
    M = causal_mask(4)
    assert M.shape == (4, 4)
    assert np.array_equal(M, np.tril(np.ones((4, 4), dtype=int)))
    assert np.all(np.triu(M, k=1) == 0)


def test_causal_mask_rejects_nonpositive_length():
    with pytest.raises(ValueError):
        causal_mask(0)


def test_softmax_rows_sum_to_one():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(5, 7))
    S = softmax(X)
    assert S.shape == X.shape
    assert np.allclose(S.sum(axis=1), 1.0)
    assert np.all(S >= 0.0)


def test_scaled_scores_match_manual_formula():
    Q = np.array([[1.0, 0.0], [0.0, 1.0]])
    K = np.array([[1.0, 0.0], [1.0, 1.0]])
    scores = scaled_dot_product_scores(Q, K)
    expected = (Q @ K.T) / np.sqrt(2.0)
    assert np.allclose(scores, expected)


def test_attention_weights_unmasked_rows_sum_to_one():
    rng = np.random.default_rng(1)
    Q = rng.normal(size=(3, 4))
    K = rng.normal(size=(5, 4))
    W = attention_weights(Q, K, mask=None)
    assert W.shape == (3, 5)
    assert np.allclose(W.sum(axis=1), 1.0)


def test_causal_mask_zeros_future_attention():
    rng = np.random.default_rng(2)
    n, d = 5, 3
    Q = rng.normal(size=(n, d))
    K = rng.normal(size=(n, d))
    mask = causal_mask(n)
    W = attention_weights(Q, K, mask=mask)
    # Strictly upper triangle must be zero under a causal mask.
    assert np.allclose(np.triu(W, k=1), 0.0)
    assert np.allclose(W.sum(axis=1), 1.0)
    # Each query may only put mass on keys ≤ its index.
    for i in range(n):
        assert np.allclose(W[i, i + 1 :], 0.0)
        assert W[i, : i + 1].sum() == pytest.approx(1.0)


def test_attention_output_shape_and_mask_consistency():
    rng = np.random.default_rng(3)
    Q = rng.normal(size=(4, 2))
    K = rng.normal(size=(4, 2))
    V = rng.normal(size=(4, 3))
    out_full = attention(Q, K, V, mask=None)
    out_causal = attention(Q, K, V, mask=causal_mask(4))
    assert out_full.shape == (4, 3)
    assert out_causal.shape == (4, 3)
    # First row is identical: only key 0 is allowed under causal, same as
    # putting all mass on key 0 if scores differ — not always equal to unmasked.
    # But future keys must not contribute under causal: reconstruct from weights.
    W = attention_weights(Q, K, mask=causal_mask(4))
    assert np.allclose(out_causal, W @ V)


def test_mask_rejects_all_blocked_query():
    Q = np.eye(2)
    K = np.eye(2)
    V = np.eye(2)
    bad_mask = np.zeros((2, 2))
    with pytest.raises(ValueError, match="allowed key"):
        attention(Q, K, V, mask=bad_mask)
