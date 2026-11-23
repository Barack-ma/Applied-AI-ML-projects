"""Experiment runners: k-means vs spectral clustering and σ sweeps."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.metrics import adjusted_rand_score

# Allow running as a script without installing the package.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from applied_ml.kmeans import compute_inertia, kmeans_lloyd
from applied_ml.spectral import (
    degree_vector,
    gaussian_similarity,
    knn_sparsify,
    spectral_cluster,
    spectral_embedding,
)

from datasets import LabeledDataset, all_datasets


@dataclass
class ClusteringResult:
    dataset: str
    method: str
    labels: np.ndarray
    ari: float
    inertia_ambient: float
    eigenvalues: np.ndarray | None
    affinity: np.ndarray | None
    params: dict


@dataclass
class SensitivityRow:
    dataset: str
    sigma: float
    n_neighbors: int
    mean_degree: float
    edge_density: float
    spectral_gap: float
    ari_mean: float
    ari_std: float
    ari_values: list[float]


def run_kmeans(
    data: LabeledDataset,
    *,
    random_state: int = 0,
) -> ClusteringResult:
    labels, centroids, _ = kmeans_lloyd(
        data.X, data.n_clusters, random_state=random_state
    )
    return ClusteringResult(
        dataset=data.name,
        method="k-means",
        labels=labels,
        ari=float(adjusted_rand_score(data.y, labels)),
        inertia_ambient=compute_inertia(data.X, labels, centroids),
        eigenvalues=None,
        affinity=None,
        params={"random_state": random_state},
    )


def run_spectral(
    data: LabeledDataset,
    *,
    sigma: float,
    n_neighbors: int,
    random_state: int = 0,
) -> ClusteringResult:
    labels, eigenvalues, affinity, _ = spectral_cluster(
        data.X,
        data.n_clusters,
        sigma=sigma,
        n_neighbors=n_neighbors,
        random_state=random_state,
        n_eigenvalues=12,
    )
    # Ambient-space inertia of the spectral partition (for reporting only).
    inertia = compute_inertia(data.X, labels)
    return ClusteringResult(
        dataset=data.name,
        method="spectral",
        labels=labels,
        ari=float(adjusted_rand_score(data.y, labels)),
        inertia_ambient=inertia,
        eigenvalues=eigenvalues,
        affinity=affinity,
        params={
            "sigma": sigma,
            "n_neighbors": n_neighbors,
            "random_state": random_state,
            "n_clusters": data.n_clusters,
        },
    )



def default_sigma(data: LabeledDataset, n_neighbors: int = 12) -> float:
    """Local bandwidth: median distance to the ``n_neighbors``-th neighbor.

    Using all pairwise distances is dominated by inter-cluster gaps on
    well-separated blobs and oversmooths the Gaussian kernel. A k-NN scale
    tracks local geometry on both convex and non-convex synthetics.
    """
    from applied_ml.spectral import pairwise_squared_distances

    dist2 = pairwise_squared_distances(data.X)
    np.fill_diagonal(dist2, np.inf)
    k = min(n_neighbors, data.X.shape[0] - 1)
    knn = np.partition(dist2, k - 1, axis=1)[:, k - 1]
    median = float(np.median(np.sqrt(knn)))
    return max(median / np.sqrt(2.0), 1e-3)


def compare_methods(
    datasets: list[LabeledDataset] | None = None,
    *,
    n_neighbors: int = 12,
    random_state: int = 0,
) -> list[ClusteringResult]:
    """Run Euclidean k-means and spectral clustering on each dataset."""
    if datasets is None:
        datasets = all_datasets(random_state=random_state)
    results: list[ClusteringResult] = []
    for data in datasets:
        sigma = default_sigma(data, n_neighbors=n_neighbors)
        results.append(run_kmeans(data, random_state=random_state))
        results.append(
            run_spectral(
                data,
                sigma=sigma,
                n_neighbors=n_neighbors,
                random_state=random_state,
            )
        )
    return results


def graph_stats(W: np.ndarray) -> tuple[float, float]:
    """Return mean degree and undirected edge density of an affinity graph."""
    d = degree_vector(W)
    n = W.shape[0]
    # Count edges above a tiny numerical threshold (mutual k-NN still has zeros).
    n_edges = np.count_nonzero(np.triu(W, k=1) > 1e-15)
    density = n_edges / (n * (n - 1) / 2.0)
    return float(d.mean()), float(density)


def sigma_sensitivity(
    data: LabeledDataset,
    sigmas: np.ndarray,
    *,
    n_neighbors: int = 12,
    seeds: list[int] | None = None,
) -> list[SensitivityRow]:
    """Sweep Gaussian bandwidth σ and summarize connectivity + ARI stability."""
    if seeds is None:
        seeds = [0, 1, 2, 3, 4]
    rows: list[SensitivityRow] = []
    for sigma in sigmas:
        aris: list[float] = []
        # Connectivity depends on σ (and k-NN), not on the k-means seed.
        W_dense = gaussian_similarity(data.X, float(sigma))
        W = knn_sparsify(W_dense, n_neighbors)
        mean_degree, density = graph_stats(W)
        _, eigenvalues, _ = spectral_embedding(
            data.X,
            data.n_clusters,
            sigma=float(sigma),
            n_neighbors=n_neighbors,
        )
        # Spectral gap between the last used eigenvector and the next one.
        # Recompute one extra eigenvalue for the gap when possible.
        from applied_ml.spectral import laplacian_eigh, normalized_laplacian

        L = normalized_laplacian(W)
        k_gap = min(data.n_clusters + 1, L.shape[0])
        evals, _ = laplacian_eigh(L, k_gap)
        if len(evals) > data.n_clusters:
            gap = float(evals[data.n_clusters] - evals[data.n_clusters - 1])
        else:
            gap = float(evals[-1] - evals[0])

        for seed in seeds:
            labels, _, _, _ = spectral_cluster(
                data.X,
                data.n_clusters,
                sigma=float(sigma),
                n_neighbors=n_neighbors,
                random_state=seed,
            )
            aris.append(float(adjusted_rand_score(data.y, labels)))

        rows.append(
            SensitivityRow(
                dataset=data.name,
                sigma=float(sigma),
                n_neighbors=n_neighbors,
                mean_degree=mean_degree,
                edge_density=density,
                spectral_gap=gap,
                ari_mean=float(np.mean(aris)),
                ari_std=float(np.std(aris)),
                ari_values=aris,
            )
        )
    return rows


def seed_stability(
    data: LabeledDataset,
    *,
    sigma: float,
    n_neighbors: int = 12,
    seeds: list[int] | None = None,
) -> dict[str, list[float]]:
    """Compare ARI across initialization seeds for both methods."""
    if seeds is None:
        seeds = list(range(10))
    kmeans_aris: list[float] = []
    spectral_aris: list[float] = []
    for seed in seeds:
        km = run_kmeans(data, random_state=seed)
        sp = run_spectral(
            data, sigma=sigma, n_neighbors=n_neighbors, random_state=seed
        )
        kmeans_aris.append(km.ari)
        spectral_aris.append(sp.ari)
    return {"k-means": kmeans_aris, "spectral": spectral_aris, "seeds": seeds}
