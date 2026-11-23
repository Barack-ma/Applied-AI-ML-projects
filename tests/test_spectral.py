"""Tests for Gaussian affinities and Laplacian spectral properties."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from applied_ml.spectral import (
    degree_vector,
    gaussian_similarity,
    knn_sparsify,
    laplacian_eigh,
    normalized_laplacian,
    pairwise_squared_distances,
    spectral_embedding,
    unnormalized_laplacian,
)


def test_pairwise_squared_distances_known_values():
    X = np.array([[0.0, 0.0], [3.0, 4.0]])
    dist2 = pairwise_squared_distances(X)
    assert dist2.shape == (2, 2)
    assert dist2[0, 1] == pytest.approx(25.0)
    assert dist2[0, 0] == pytest.approx(0.0)


def test_gaussian_similarity_symmetric_with_zero_diagonal():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(12, 2))
    W = gaussian_similarity(X, sigma=0.8)
    assert W.shape == (12, 12)
    assert np.allclose(W, W.T)
    assert np.allclose(np.diag(W), 0.0)
    assert np.all(W >= 0.0)
    assert np.all(W <= 1.0 + 1e-12)


def test_gaussian_similarity_rejects_nonpositive_sigma():
    X = np.zeros((3, 2))
    with pytest.raises(ValueError):
        gaussian_similarity(X, sigma=0.0)


def test_knn_sparsify_is_symmetric_and_sparse():
    rng = np.random.default_rng(2)
    X = rng.normal(size=(20, 2))
    W = gaussian_similarity(X, sigma=1.0)
    A = knn_sparsify(W, n_neighbors=3)
    assert np.allclose(A, A.T)
    # Symmetrized k-NN has at most n*k undirected edge endpoints from the
    # directed top-k selection (in-degree can concentrate on hubs).
    n_edges = np.count_nonzero(np.triu(A, k=1) > 0)
    assert n_edges <= 20 * 3
    assert np.count_nonzero(A > 0, axis=1).min() >= 1


def test_unnormalized_laplacian_psd_and_nullspace():
    """Combinatorial Laplacian is PSD and annihilates the constant vector."""
    rng = np.random.default_rng(3)
    X = rng.normal(size=(15, 2))
    W = gaussian_similarity(X, sigma=1.2)
    L = unnormalized_laplacian(W)
    ones = np.ones(W.shape[0])
    assert np.allclose(L @ ones, 0.0, atol=1e-10)
    evals = np.linalg.eigvalsh(L)
    assert evals[0] == pytest.approx(0.0, abs=1e-9)
    assert np.all(evals >= -1e-9)


def test_normalized_laplacian_spectrum_in_unit_interval():
    rng = np.random.default_rng(4)
    X = rng.normal(size=(18, 2))
    W = knn_sparsify(gaussian_similarity(X, sigma=1.0), n_neighbors=5)
    L = normalized_laplacian(W)
    evals = np.linalg.eigvalsh(L)
    assert evals[0] == pytest.approx(0.0, abs=1e-8)
    assert evals[-1] <= 2.0 + 1e-8
    assert np.all(evals >= -1e-8)


def test_degree_vector_matches_row_sums():
    W = np.array([[0.0, 0.5, 0.5], [0.5, 0.0, 0.0], [0.5, 0.0, 0.0]])
    assert np.allclose(degree_vector(W), [1.0, 0.5, 0.5])


def test_laplacian_eigh_returns_ordered_smallest():
    rng = np.random.default_rng(5)
    X = rng.normal(size=(25, 3))
    L = normalized_laplacian(gaussian_similarity(X, sigma=0.9))
    evals, evecs = laplacian_eigh(L, n_components=4)
    assert evals.shape == (4,)
    assert evecs.shape == (25, 4)
    assert np.all(np.diff(evals) >= -1e-10)
    # Rayleigh quotients should recover the eigenvalues.
    for i in range(4):
        v = evecs[:, i]
        rayleigh = (v @ L @ v) / (v @ v)
        assert rayleigh == pytest.approx(evals[i], abs=1e-8)


def test_spectral_embedding_separates_two_blobs():
    rng = np.random.default_rng(6)
    a = rng.normal(loc=[-4.0, 0.0], scale=0.2, size=(30, 2))
    b = rng.normal(loc=[4.0, 0.0], scale=0.2, size=(30, 2))
    X = np.vstack([a, b])
    embedding, eigenvalues, W = spectral_embedding(
        X, n_clusters=2, sigma=1.0, n_neighbors=8
    )
    assert embedding.shape == (60, 2)
    assert eigenvalues[0] == pytest.approx(0.0, abs=1e-6)
    # Second embedding coordinate (first nontrivial mode) should separate blobs.
    coord = embedding[:, 1]
    left = coord[:30].mean()
    right = coord[30:].mean()
    assert left * right < 0.0 or abs(left - right) > 0.1
    assert W.shape == (60, 60)
