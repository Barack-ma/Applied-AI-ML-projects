"""Graph Laplacian and spectral embedding utilities."""

from __future__ import annotations

import numpy as np
from scipy import linalg

from .kmeans import kmeans_lloyd


def pairwise_squared_distances(X: np.ndarray) -> np.ndarray:
    """Return the dense pairwise squared Euclidean distance matrix."""
    X = np.asarray(X, dtype=float)
    # ||x - y||^2 = ||x||^2 + ||y||^2 - 2 x·y
    sq = np.sum(X * X, axis=1, keepdims=True)
    dist2 = sq + sq.T - 2.0 * (X @ X.T)
    np.maximum(dist2, 0.0, out=dist2)
    np.fill_diagonal(dist2, 0.0)
    return dist2


def gaussian_similarity(X: np.ndarray, sigma: float) -> np.ndarray:
    """Build a dense Gaussian (RBF) similarity graph.

    ``W_ij = exp(-||x_i - x_j||^2 / (2 σ^2))`` with zeros on the diagonal.
    """
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    dist2 = pairwise_squared_distances(X)
    W = np.exp(-dist2 / (2.0 * sigma * sigma))
    np.fill_diagonal(W, 0.0)
    return W


def knn_sparsify(W: np.ndarray, n_neighbors: int) -> np.ndarray:
    """Retain a mutual k-NN sparsification of a symmetric affinity matrix.

    For each row we keep the ``n_neighbors`` largest off-diagonal affinities,
    then form the mutual (symmetrized) graph ``(A + A.T) / 2``.
    """
    W = np.asarray(W, dtype=float)
    n = W.shape[0]
    if W.shape != (n, n):
        raise ValueError("W must be square")
    if n_neighbors < 1:
        raise ValueError("n_neighbors must be at least 1")
    k = min(n_neighbors, n - 1)

    A = np.zeros_like(W)
    # Ignore self-similarity when selecting neighbors.
    masked = W.copy()
    np.fill_diagonal(masked, -np.inf)
    neighbor_idx = np.argpartition(masked, -k, axis=1)[:, -k:]
    rows = np.repeat(np.arange(n), k)
    A[rows, neighbor_idx.ravel()] = W[rows, neighbor_idx.ravel()]
    return 0.5 * (A + A.T)


def degree_vector(W: np.ndarray) -> np.ndarray:
    """Return node degrees ``d_i = sum_j W_ij``."""
    return np.asarray(W, dtype=float).sum(axis=1)


def unnormalized_laplacian(W: np.ndarray) -> np.ndarray:
    """Return the combinatorial Laplacian ``L = D - W``."""
    W = np.asarray(W, dtype=float)
    d = degree_vector(W)
    return np.diag(d) - W


def normalized_laplacian(W: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Return the symmetric normalized Laplacian ``L_sym = I - D^{-1/2} W D^{-1/2}``.

    Isolated nodes (degree ≈ 0) are regularized by ``eps`` so the expression
    remains well-defined for sparse graphs.
    """
    W = np.asarray(W, dtype=float)
    d = degree_vector(W)
    d_inv_sqrt = 1.0 / np.sqrt(np.maximum(d, eps))
    D_inv_sqrt = np.diag(d_inv_sqrt)
    I = np.eye(W.shape[0])
    return I - D_inv_sqrt @ W @ D_inv_sqrt


def laplacian_eigh(L: np.ndarray, n_components: int) -> tuple[np.ndarray, np.ndarray]:
    """Return the ``n_components`` smallest eigenpairs of a symmetric Laplacian."""
    L = np.asarray(L, dtype=float)
    if n_components < 1 or n_components > L.shape[0]:
        raise ValueError("n_components must be in [1, n]")
    # Dense eigh is fine for the synthetic study sizes used here.
    eigenvalues, eigenvectors = linalg.eigh(L, subset_by_index=[0, n_components - 1])
    return eigenvalues, eigenvectors


def spectral_embedding(
    X: np.ndarray,
    n_clusters: int,
    *,
    sigma: float = 1.0,
    n_neighbors: int | None = 10,
    normalized: bool = True,
    n_eigenvalues: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Embed points via the smallest nontrivial Laplacian eigenvectors.

    Returns
    -------
    embedding :
        Matrix of shape ``(n_samples, n_clusters)`` used for Euclidean k-means.
    eigenvalues :
        The smallest Laplacian eigenvalues (length ``n_eigenvalues`` or
        ``n_clusters``). The first ``n_clusters`` columns of the associated
        eigenvectors form the embedding before optional row normalization.
    affinity :
        The (possibly sparsified) similarity matrix used to build the Laplacian.
    """
    W = gaussian_similarity(X, sigma)
    if n_neighbors is not None:
        W = knn_sparsify(W, n_neighbors)
    L = normalized_laplacian(W) if normalized else unnormalized_laplacian(W)
    n_eigs = n_clusters if n_eigenvalues is None else n_eigenvalues
    n_eigs = max(n_eigs, n_clusters)
    n_eigs = min(n_eigs, L.shape[0])
    eigenvalues, eigenvectors = laplacian_eigh(L, n_eigs)
    embedding = eigenvectors[:, :n_clusters].copy()
    # Row-normalize for the Ng–Jordan–Weiss normalized spectral clustering step.
    if normalized:
        norms = np.linalg.norm(embedding, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-12)
        embedding = embedding / norms
    return embedding, eigenvalues, W


def spectral_cluster(
    X: np.ndarray,
    n_clusters: int,
    *,
    sigma: float = 1.0,
    n_neighbors: int | None = 10,
    normalized: bool = True,
    random_state: int | None = 0,
    n_eigenvalues: int | None = 12,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Spectral clustering: Laplacian embedding followed by Lloyd k-means.

    Returns labels, Laplacian eigenvalues, affinity matrix, and k-means inertia
    in the embedding space.
    """
    embedding, eigenvalues, W = spectral_embedding(
        X,
        n_clusters,
        sigma=sigma,
        n_neighbors=n_neighbors,
        normalized=normalized,
        n_eigenvalues=n_eigenvalues,
    )
    labels, _, inertia = kmeans_lloyd(
        embedding, n_clusters, random_state=random_state
    )
    return labels, eigenvalues, W, inertia
