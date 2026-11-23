"""Shared numerical helpers for the Applied ML portfolio."""

from .kmeans import compute_inertia, kmeans_lloyd
from .spectral import (
    gaussian_similarity,
    knn_sparsify,
    normalized_laplacian,
    spectral_embedding,
    unnormalized_laplacian,
)

__version__ = "0.1.0"

__all__ = [
    "compute_inertia",
    "gaussian_similarity",
    "kmeans_lloyd",
    "knn_sparsify",
    "normalized_laplacian",
    "spectral_embedding",
    "unnormalized_laplacian",
]
