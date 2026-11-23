"""Scaled dot-product attention primitives (NumPy).

Generalized from MATH 412 mini-transformer coursework scaffolding
(Jason Aubrey / University of Arizona), adapted for this portfolio.
"""

from __future__ import annotations

import numpy as np


def causal_mask(length: int) -> np.ndarray:
    """Return a lower-triangular 0/1 causal mask of shape ``(length, length)``."""
    if not isinstance(length, (int, np.integer)) or length <= 0:
        raise ValueError("length must be a positive integer")
    return np.tril(np.ones((length, length), dtype=int))


def scaled_dot_product_scores(Q: np.ndarray, K: np.ndarray) -> np.ndarray:
    """Return ``(Q K^T) / sqrt(d_k)`` for 2-D query/key matrices."""
    Q = np.asarray(Q, dtype=float)
    K = np.asarray(K, dtype=float)
    if Q.ndim != 2 or K.ndim != 2 or Q.shape[1] != K.shape[1]:
        raise ValueError("Q and K must be 2-D with matching feature dimension")
    return (Q @ K.T) / np.sqrt(Q.shape[1])


def softmax(X: np.ndarray) -> np.ndarray:
    """Row-wise numerically stable softmax."""
    X = np.asarray(X, dtype=float)
    shifted = X - np.max(X, axis=-1, keepdims=True)
    exponentials = np.exp(shifted)
    return exponentials / np.sum(exponentials, axis=-1, keepdims=True)


def _apply_mask(scores: np.ndarray, mask: np.ndarray | None) -> np.ndarray:
    if mask is None:
        return scores
    mask = np.asarray(mask)
    if mask.shape != scores.shape:
        raise ValueError("mask must have the same shape as the score matrix")
    allowed = mask.astype(bool)
    if not np.all(allowed.any(axis=1)):
        raise ValueError("every query must have at least one allowed key")
    return np.where(allowed, scores, -np.inf)


def attention_weights(
    Q: np.ndarray,
    K: np.ndarray,
    mask: np.ndarray | None = None,
) -> np.ndarray:
    """Return the softmax attention weight matrix (rows sum to 1)."""
    scores = _apply_mask(scaled_dot_product_scores(Q, K), mask)
    return softmax(scores)


def attention(
    Q: np.ndarray,
    K: np.ndarray,
    V: np.ndarray,
    mask: np.ndarray | None = None,
) -> np.ndarray:
    """Masked scaled dot-product attention: ``softmax(QK^T/√d) V``."""
    scores = scaled_dot_product_scores(Q, K)
    V = np.asarray(V, dtype=float)
    if V.ndim != 2 or V.shape[0] != scores.shape[1]:
        raise ValueError("V must be 2-D with one row per key")
    weights = softmax(_apply_mask(scores, mask))
    return weights @ V


def attention_head(
    X: np.ndarray,
    W_Q: np.ndarray,
    W_K: np.ndarray,
    W_V: np.ndarray,
    mask: np.ndarray | None = None,
) -> np.ndarray:
    """Project ``X`` to Q, K, V and return one attention-head output."""
    X = np.asarray(X, dtype=float)
    return attention(X @ W_Q, X @ W_K, X @ W_V, mask=mask)


def attention_head_weights(
    X: np.ndarray,
    W_Q: np.ndarray,
    W_K: np.ndarray,
    mask: np.ndarray | None = None,
) -> np.ndarray:
    """Attention weights for one head (for visualization)."""
    X = np.asarray(X, dtype=float)
    return attention_weights(X @ W_Q, X @ W_K, mask=mask)
