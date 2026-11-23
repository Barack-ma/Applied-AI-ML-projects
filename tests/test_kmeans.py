"""Tests for Lloyd k-means inertia helpers."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from applied_ml.kmeans import compute_inertia, kmeans_lloyd


def test_inertia_two_point_clusters():
    X = np.array([[0.0, 0.0], [0.0, 0.0], [1.0, 0.0], [1.0, 0.0]])
    labels = np.array([0, 0, 1, 1])
    centroids = np.array([[0.0, 0.0], [1.0, 0.0]])
    assert compute_inertia(X, labels, centroids) == pytest.approx(0.0)


def test_inertia_matches_manual_sum_of_squares():
    X = np.array([[0.0, 0.0], [2.0, 0.0], [0.0, 4.0]])
    labels = np.array([0, 0, 1])
    # Cluster 0 mean is (1, 0): contributions 1 + 1 = 2.
    # Cluster 1 mean is (0, 4): contribution 0.
    assert compute_inertia(X, labels) == pytest.approx(2.0)


def test_inertia_rejects_shape_mismatch():
    X = np.zeros((3, 2))
    with pytest.raises(ValueError):
        compute_inertia(X, np.array([0, 1]))


def test_kmeans_recovers_well_separated_blobs():
    rng = np.random.default_rng(0)
    c0 = rng.normal(loc=[-3.0, 0.0], scale=0.15, size=(40, 2))
    c1 = rng.normal(loc=[3.0, 0.0], scale=0.15, size=(40, 2))
    X = np.vstack([c0, c1])
    true = np.array([0] * 40 + [1] * 40)
    labels, centroids, inertia = kmeans_lloyd(X, n_clusters=2, random_state=0)
    # Labels may be flipped; check partition agreement.
    agree = (labels == true).mean()
    agree_flipped = (labels == 1 - true).mean()
    assert max(agree, agree_flipped) == pytest.approx(1.0)
    assert inertia >= 0.0
    assert centroids.shape == (2, 2)


def test_inertia_nonnegative_after_lloyd():
    rng = np.random.default_rng(7)
    X = rng.normal(size=(50, 3))
    _, _, inertia = kmeans_lloyd(X, n_clusters=3, random_state=7)
    assert inertia >= 0.0
