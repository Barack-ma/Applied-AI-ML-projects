"""Shared numerical helpers for the Applied ML portfolio.

Module map
----------
- ``embeddings`` — random (not learned) token embeddings
- ``attention`` — causal masks, scaled scores, softmax, attention heads
- ``projections`` — orthogonal weights, feed-forward / transformer block
- ``sparsity`` — soft-thresholding, OLS / Ridge / LASSO
- ``kmeans`` — Lloyd k-means + inertia
- ``spectral`` — Gaussian graphs, Laplacians, spectral embedding

``spectral`` depends on ``kmeans``. ``projections.transformer_block`` depends
on ``attention``. Other modules are independent NumPy utilities.
"""

from .attention import (
    attention,
    attention_head,
    attention_head_weights,
    attention_weights,
    causal_mask,
    scaled_dot_product_scores,
    softmax,
)
from .embeddings import SEED, embed_tokens, make_embedding_matrix, set_seed
from .kmeans import compute_inertia, kmeans_lloyd
from .projections import (
    feed_forward,
    low_rank_weight,
    orthogonal_matrix,
    orthogonal_projection_matrices,
    project_features,
    relu,
    spectral_normalize,
    transformer_block,
)
from .sparsity import lasso_cd, mse, ols, ridge, soft_threshold
from .spectral import (
    gaussian_similarity,
    knn_sparsify,
    normalized_laplacian,
    spectral_embedding,
    unnormalized_laplacian,
)

__version__ = "0.1.0"

__all__ = [
    "SEED",
    "attention",
    "attention_head",
    "attention_head_weights",
    "attention_weights",
    "causal_mask",
    "compute_inertia",
    "embed_tokens",
    "feed_forward",
    "gaussian_similarity",
    "kmeans_lloyd",
    "knn_sparsify",
    "lasso_cd",
    "low_rank_weight",
    "make_embedding_matrix",
    "mse",
    "normalized_laplacian",
    "ols",
    "orthogonal_matrix",
    "orthogonal_projection_matrices",
    "project_features",
    "relu",
    "ridge",
    "scaled_dot_product_scores",
    "set_seed",
    "softmax",
    "soft_threshold",
    "spectral_embedding",
    "spectral_normalize",
    "transformer_block",
    "unnormalized_laplacian",
]
