"""Synthetic datasets for the spectral clustering study."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.datasets import make_blobs, make_circles, make_moons


@dataclass(frozen=True)
class LabeledDataset:
    """A small labeled point cloud with a human-readable name."""

    name: str
    X: np.ndarray
    y: np.ndarray
    n_clusters: int
    description: str


def load_blobs(n_samples: int = 300, random_state: int = 0) -> LabeledDataset:
    """Three well-separated isotropic Gaussian blobs (Euclidean-friendly)."""
    X, y = make_blobs(
        n_samples=n_samples,
        centers=[[-4.5, -2.0], [0.0, 4.0], [4.5, -2.0]],
        cluster_std=0.55,
        random_state=random_state,
    )

    return LabeledDataset(
        name="blobs",
        X=X.astype(float),
        y=y.astype(int),
        n_clusters=3,
        description="Well-separated isotropic Gaussian blobs",
    )


def load_circles(n_samples: int = 400, random_state: int = 1) -> LabeledDataset:
    """Concentric circles: linearly inseparable in ambient Euclidean space."""
    X, y = make_circles(
        n_samples=n_samples,
        factor=0.45,
        noise=0.05,
        random_state=random_state,
    )
    return LabeledDataset(
        name="circles",
        X=X.astype(float),
        y=y.astype(int),
        n_clusters=2,
        description="Concentric circles (nested rings)",
    )


def load_moons(n_samples: int = 400, random_state: int = 2) -> LabeledDataset:
    """Interlocking half-moons: another classic non-convex geometry."""
    X, y = make_moons(n_samples=n_samples, noise=0.07, random_state=random_state)
    return LabeledDataset(
        name="moons",
        X=X.astype(float),
        y=y.astype(int),
        n_clusters=2,
        description="Interlocking crescent moons",
    )


def all_datasets(random_state: int = 0) -> list[LabeledDataset]:
    """Return the three synthetic datasets used throughout the study."""
    return [
        load_blobs(random_state=random_state),
        load_circles(random_state=random_state + 1),
        load_moons(random_state=random_state + 2),
    ]
