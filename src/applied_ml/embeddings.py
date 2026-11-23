"""Random token embeddings and lookup helpers.

These embeddings are **random** (not learned). They provide a fixed map from
token IDs to vectors for pedagogical forward passes.
"""

from __future__ import annotations

import numpy as np

# Course / portfolio default seed for reproducible demos.
SEED = 412


def set_seed(seed: int = SEED) -> None:
    """Set the legacy NumPy global RNG seed for reproducibility."""
    np.random.seed(seed)


def make_embedding_matrix(
    vocab_size: int,
    d_model: int,
    seed: int = SEED,
    scale: float = 0.02,
) -> np.ndarray:
    """Create a random token embedding matrix of shape ``(vocab_size, d_model)``.

    Row ``i`` is the embedding for token ``i``. Values are i.i.d. Gaussian
    with standard deviation ``scale`` — **not** trained / learned.
    """
    if vocab_size <= 0 or d_model <= 0:
        raise ValueError("vocab_size and d_model must be positive")
    rng = np.random.default_rng(seed)
    return rng.standard_normal((vocab_size, d_model)) * scale


def embed_tokens(token_ids: np.ndarray | list[int], E: np.ndarray) -> np.ndarray:
    """Look up embeddings for token IDs; return shape ``(n_tokens, d_model)``."""
    E = np.asarray(E, dtype=float)
    ids = np.asarray(token_ids, dtype=int)
    if E.ndim != 2:
        raise ValueError("E must be a 2-D embedding matrix")
    if ids.ndim != 1:
        raise ValueError("token_ids must be a 1-D sequence")
    if np.any(ids < 0) or np.any(ids >= E.shape[0]):
        raise ValueError("token_ids out of vocabulary range")
    return E[ids]
