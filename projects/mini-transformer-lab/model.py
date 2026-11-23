"""Small NumPy mini-transformer used by the lab experiments.

**Disclaimer:** This is a pedagogical forward pass with **random** (not learned)
token embeddings and fixed orthogonal projection weights. It is **not** a
pretrained language model.

Core primitives live in ``src/applied_ml``; this module wires a reproducible
default configuration for the lab.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from applied_ml.attention import attention_head_weights, causal_mask
from applied_ml.embeddings import SEED, embed_tokens, make_embedding_matrix
from applied_ml.projections import (
    orthogonal_projection_matrices,
    transformer_block,
)


@dataclass(frozen=True)
class MiniTransformerConfig:
    vocab_size: int = 32
    d_model: int = 16
    d_head: int = 16
    d_hidden: int = 32
    seed: int = SEED


@dataclass
class MiniTransformer:
    """One-block transformer with random embeddings and fixed weights."""

    config: MiniTransformerConfig
    E: np.ndarray
    W_Q: np.ndarray
    W_K: np.ndarray
    W_V: np.ndarray
    W1: np.ndarray
    b1: np.ndarray
    W2: np.ndarray
    b2: np.ndarray

    @classmethod
    def create(cls, config: MiniTransformerConfig | None = None) -> MiniTransformer:
        cfg = config or MiniTransformerConfig()
        rng = np.random.default_rng(cfg.seed + 100)
        E = make_embedding_matrix(cfg.vocab_size, cfg.d_model, seed=cfg.seed)
        W_Q, W_K, W_V = orthogonal_projection_matrices(
            cfg.d_model, cfg.d_head, seed=cfg.seed + 1
        )
        # When d_head == d_model, attention residual matches token width.
        W1 = rng.standard_normal((cfg.d_model, cfg.d_hidden)) * 0.1
        b1 = np.zeros(cfg.d_hidden)
        W2 = rng.standard_normal((cfg.d_hidden, cfg.d_model)) * 0.1
        b2 = np.zeros(cfg.d_model)
        return cls(cfg, E, W_Q, W_K, W_V, W1, b1, W2, b2)

    def embed(self, token_ids: np.ndarray | list[int]) -> np.ndarray:
        return embed_tokens(token_ids, self.E)

    def forward(
        self,
        token_ids: np.ndarray | list[int],
        *,
        causal: bool = False,
        W_Q: np.ndarray | None = None,
        W_K: np.ndarray | None = None,
        W_V: np.ndarray | None = None,
        X: np.ndarray | None = None,
    ) -> np.ndarray:
        """Run one transformer block; optionally override projections or inputs."""
        if X is None:
            X = self.embed(token_ids)
        else:
            X = np.asarray(X, dtype=float)
        n = X.shape[0]
        mask = causal_mask(n) if causal else None
        return transformer_block(
            X,
            W_Q if W_Q is not None else self.W_Q,
            W_K if W_K is not None else self.W_K,
            W_V if W_V is not None else self.W_V,
            self.W1,
            self.b1,
            self.W2,
            self.b2,
            mask=mask,
        )

    def attention_matrix(
        self,
        token_ids: np.ndarray | list[int],
        *,
        causal: bool = False,
        W_Q: np.ndarray | None = None,
        W_K: np.ndarray | None = None,
        X: np.ndarray | None = None,
    ) -> np.ndarray:
        """Softmax attention weights for visualization."""
        if X is None:
            X = self.embed(token_ids)
        else:
            X = np.asarray(X, dtype=float)
        mask = causal_mask(X.shape[0]) if causal else None
        return attention_head_weights(
            X,
            W_Q if W_Q is not None else self.W_Q,
            W_K if W_K is not None else self.W_K,
            mask=mask,
        )
