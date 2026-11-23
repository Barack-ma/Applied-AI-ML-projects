"""Lloyd k-means helpers used across portfolio projects."""

from __future__ import annotations

import numpy as np


def compute_inertia(
    X: np.ndarray,
    labels: np.ndarray,
    centroids: np.ndarray | None = None,
) -> float:
    """Return the within-cluster sum of squared Euclidean distances.

    Parameters
    ----------
    X :
        Data matrix of shape ``(n_samples, n_features)``.
    labels :
        Integer cluster assignments of shape ``(n_samples,)``.
    centroids :
        Optional cluster centers of shape ``(n_clusters, n_features)``.
        When omitted, centers are the empirical means of each labeled group.
    """
    X = np.asarray(X, dtype=float)
    labels = np.asarray(labels, dtype=int)
    if X.ndim != 2:
        raise ValueError("X must be a 2-D array")
    if labels.shape[0] != X.shape[0]:
        raise ValueError("labels length must match number of rows in X")

    if centroids is None:
        unique = np.unique(labels)
        centroids = np.vstack([X[labels == k].mean(axis=0) for k in unique])
        # Map labels to contiguous [0, K) for indexing when unique is sparse.
        remap = {int(k): i for i, k in enumerate(unique)}
        idx = np.array([remap[int(lab)] for lab in labels], dtype=int)
    else:
        centroids = np.asarray(centroids, dtype=float)
        if centroids.ndim != 2 or centroids.shape[1] != X.shape[1]:
            raise ValueError("centroids must have shape (n_clusters, n_features)")
        idx = labels

    diffs = X - centroids[idx]
    return float(np.sum(diffs * diffs))


def _init_centroids(X: np.ndarray, n_clusters: int, rng: np.random.Generator) -> np.ndarray:
    """Choose distinct random rows of ``X`` as initial centroids."""
    n = X.shape[0]
    if n_clusters > n:
        raise ValueError("n_clusters cannot exceed n_samples")
    indices = rng.choice(n, size=n_clusters, replace=False)
    return X[indices].copy()


def _lloyd_once(
    X: np.ndarray,
    n_clusters: int,
    rng: np.random.Generator,
    *,
    max_iter: int,
    tol: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    centroids = _init_centroids(X, n_clusters, rng)
    labels = np.zeros(X.shape[0], dtype=int)

    for _ in range(max_iter):
        distances = ((X[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2)
        new_labels = distances.argmin(axis=1)

        new_centroids = centroids.copy()
        for k in range(n_clusters):
            members = X[new_labels == k]
            if len(members) == 0:
                new_centroids[k] = X[rng.integers(0, X.shape[0])]
            else:
                new_centroids[k] = members.mean(axis=0)

        shift = np.linalg.norm(new_centroids - centroids)
        centroids = new_centroids
        labels = new_labels
        if shift <= tol:
            break

    inertia = compute_inertia(X, labels, centroids)
    return labels, centroids, inertia


def kmeans_lloyd(
    X: np.ndarray,
    n_clusters: int,
    *,
    max_iter: int = 300,
    tol: float = 1e-4,
    n_init: int = 10,
    random_state: int | None = 0,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Run Lloyd k-means with ``n_init`` random restarts; keep lowest inertia.

    Returns
    -------
    labels :
        Hard assignments of shape ``(n_samples,)``.
    centroids :
        Final centers of shape ``(n_clusters, n_features)``.
    inertia :
        Within-cluster sum of squares for the returned partition.
    """
    X = np.asarray(X, dtype=float)
    if n_init < 1:
        raise ValueError("n_init must be at least 1")
    rng = np.random.default_rng(random_state)

    best_labels: np.ndarray | None = None
    best_centroids: np.ndarray | None = None
    best_inertia = np.inf
    for _ in range(n_init):
        # Independent sub-stream per restart for reproducibility.
        child = np.random.default_rng(rng.integers(0, 2**31 - 1))
        labels, centroids, inertia = _lloyd_once(
            X, n_clusters, child, max_iter=max_iter, tol=tol
        )
        if inertia < best_inertia:
            best_inertia = inertia
            best_labels = labels
            best_centroids = centroids

    assert best_labels is not None and best_centroids is not None
    return best_labels, best_centroids, float(best_inertia)
