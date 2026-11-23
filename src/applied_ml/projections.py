"""Orthogonal projections, low-rank weights, and a tiny transformer block.

Components generalized from MATH 412 mini-transformer coursework scaffolding
(Jason Aubrey / University of Arizona) for portfolio experiments.
"""

from __future__ import annotations

import numpy as np

from .attention import attention_head


def orthogonal_matrix(
    rows: int,
    cols: int | None = None,
    seed: int = 0,
) -> np.ndarray:
    """Return a reproducible ``rows × cols`` matrix with orthonormal short side."""
    if cols is None:
        cols = rows
    if not isinstance(rows, (int, np.integer)) or not isinstance(cols, (int, np.integer)):
        raise TypeError("rows and cols must be integers")
    if rows <= 0 or cols <= 0:
        raise ValueError("rows and cols must be positive")
    rng = np.random.default_rng(seed)
    if rows >= cols:
        Q, _ = np.linalg.qr(rng.standard_normal((rows, cols)), mode="reduced")
        return Q
    Q, _ = np.linalg.qr(rng.standard_normal((cols, rows)), mode="reduced")
    return Q.T


def orthogonal_projection_matrices(
    d_model: int,
    d_head: int,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return distinct orthogonal ``(W_Q, W_K, W_V)`` weight matrices."""
    if not isinstance(d_model, (int, np.integer)) or not isinstance(d_head, (int, np.integer)):
        raise TypeError("d_model and d_head must be integers")
    if d_model <= 0 or d_head <= 0:
        raise ValueError("d_model and d_head must be positive")
    return tuple(
        orthogonal_matrix(d_model, d_head, seed=seed + offset) for offset in range(3)
    )


def relu(Z: np.ndarray) -> np.ndarray:
    """Apply ``max(0, z)`` entrywise."""
    return np.maximum(np.asarray(Z, dtype=float), 0.0)


def _validated_feed_forward_inputs(
    X: np.ndarray,
    W1: np.ndarray,
    b1: np.ndarray,
    W2: np.ndarray,
    b2: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    X = np.asarray(X, dtype=float)
    W1 = np.asarray(W1, dtype=float)
    b1 = np.asarray(b1, dtype=float)
    W2 = np.asarray(W2, dtype=float)
    b2 = np.asarray(b2, dtype=float)
    if X.ndim != 2 or W1.ndim != 2 or W2.ndim != 2 or b1.ndim != 1 or b2.ndim != 1:
        raise ValueError("X, W1, W2 must be matrices and b1, b2 must be vectors")
    _, d_model = X.shape
    if W1.shape[0] != d_model:
        raise ValueError("W1 must have one row per model feature")
    d_hidden = W1.shape[1]
    if b1.shape != (d_hidden,) or W2.shape != (d_hidden, d_model) or b2.shape != (d_model,):
        raise ValueError("incompatible feed-forward shapes")
    return X, W1, b1, W2, b2


def feed_forward(
    X: np.ndarray,
    W1: np.ndarray,
    b1: np.ndarray,
    W2: np.ndarray,
    b2: np.ndarray,
) -> np.ndarray:
    """Two-layer position-wise MLP with ReLU hidden activation."""
    X, W1, b1, W2, b2 = _validated_feed_forward_inputs(X, W1, b1, W2, b2)
    hidden = relu(X @ W1 + b1)
    return hidden @ W2 + b2


def transformer_block(
    X: np.ndarray,
    W_Q: np.ndarray,
    W_K: np.ndarray,
    W_V: np.ndarray,
    W1: np.ndarray,
    b1: np.ndarray,
    W2: np.ndarray,
    b2: np.ndarray,
    mask: np.ndarray | None = None,
) -> np.ndarray:
    """One residual attention head followed by a residual feed-forward block."""
    X = np.asarray(X, dtype=float)
    attended = attention_head(X, W_Q, W_K, W_V, mask=mask)
    if attended.shape != X.shape:
        raise ValueError("attention output must match X for a residual addition")
    after_attention = X + attended
    return after_attention + feed_forward(after_attention, W1, b1, W2, b2)


def _validated_projection_inputs(
    X: np.ndarray,
    Q: np.ndarray,
    atol: float = 1e-10,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert arrays and verify that the columns of ``Q`` are orthonormal."""
    X = np.asarray(X, dtype=float)
    Q = np.asarray(Q, dtype=float)
    if X.ndim != 2 or Q.ndim != 2:
        raise ValueError("X and Q must both be matrices")
    if Q.shape[0] != X.shape[1] or Q.shape[1] == 0 or Q.shape[1] > Q.shape[0]:
        raise ValueError("Q must have one row per feature and at least one column")
    if not np.allclose(Q.T @ Q, np.eye(Q.shape[1]), atol=atol, rtol=0):
        raise ValueError("the columns of Q must be orthonormal")
    return X, Q


def project_features(X: np.ndarray, Q: np.ndarray, atol: float = 1e-10) -> np.ndarray:
    """Project each token row of ``X`` onto the feature subspace spanned by ``Q``."""
    X, Q = _validated_projection_inputs(X, Q, atol=atol)
    return X @ Q @ Q.T


def low_rank_weight(W: np.ndarray, k: int) -> np.ndarray:
    """Return the best rank-``k`` approximation of ``W`` in Frobenius norm."""
    W = np.asarray(W, dtype=float)
    if W.ndim != 2 or 0 in W.shape:
        raise ValueError("W must be a nonempty matrix")
    if not isinstance(k, (int, np.integer)):
        raise TypeError("k must be an integer")
    if k < 0 or k > min(W.shape):
        raise ValueError("k must satisfy 0 <= k <= min(W.shape)")
    U, singular_values, Vt = np.linalg.svd(W, full_matrices=False)
    return (U[:, :k] * singular_values[:k]) @ Vt[:k, :]


def spectral_normalize(W: np.ndarray, target: float = 1.0) -> np.ndarray:
    """Return a copy of ``W`` scaled to have spectral norm ``target``."""
    W = np.asarray(W, dtype=float)
    if W.ndim != 2 or 0 in W.shape:
        raise ValueError("W must be a nonempty matrix")
    if not np.isscalar(target) or isinstance(target, (str, bytes)):
        raise TypeError("target must be a positive finite scalar")
    target = float(target)
    if not np.isfinite(target) or target <= 0:
        raise ValueError("target must be a positive finite scalar")
    sigma_max = np.linalg.norm(W, ord=2)
    if sigma_max == 0:
        raise ValueError("the zero matrix cannot be spectrally normalized")
    return (target / sigma_max) * W
