"""Figure generation for the spectral clustering study."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from datasets import LabeledDataset
from experiments import ClusteringResult, SensitivityRow


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def plot_side_by_side(
    data: LabeledDataset,
    kmeans_result: ClusteringResult,
    spectral_result: ClusteringResult,
    out_path: Path,
) -> Path:
    """Truth / k-means / spectral labels in one row."""
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), constrained_layout=True)
    panels = [
        ("Ground truth", data.y),
        (f"k-means (ARI={kmeans_result.ari:.2f})", kmeans_result.labels),
        (f"Spectral (ARI={spectral_result.ari:.2f})", spectral_result.labels),
    ]
    for ax, (title, labels) in zip(axes, panels):
        sc = ax.scatter(
            data.X[:, 0],
            data.X[:, 1],
            c=labels,
            cmap="tab10",
            s=14,
            linewidths=0,
        )
        ax.set_title(title)
        ax.set_xlabel("x₁")
        ax.set_ylabel("x₂")
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, alpha=0.25)
    fig.suptitle(f"{data.name}: Euclidean k-means vs spectral clustering", fontsize=12)
    out_path = Path(out_path)
    _ensure_dir(out_path.parent)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return out_path


def plot_eigenvalues(
    results: list[ClusteringResult],
    out_path: Path,
    n_show: int = 12,
) -> Path:
    """Plot the smallest Laplacian eigenvalues for each spectral run."""
    spectral = [r for r in results if r.method == "spectral" and r.eigenvalues is not None]
    fig, axes = plt.subplots(
        1, len(spectral), figsize=(4.0 * max(len(spectral), 1), 3.6), constrained_layout=True
    )
    if len(spectral) == 1:
        axes = [axes]
    for ax, result in zip(axes, spectral):
        ev = result.eigenvalues[:n_show]
        ax.plot(np.arange(1, len(ev) + 1), ev, marker="o", linewidth=1.5)
        ax.axhline(0.0, color="0.5", linewidth=0.8, linestyle="--")
        ax.set_title(f"{result.dataset}")
        ax.set_xlabel("Eigenvalue index")
        ax.set_ylabel("λ (normalized Laplacian)")
        ax.grid(True, alpha=0.3)
        # Mark where the embedding would stop (n_clusters eigenvectors).
        n_clusters = int(result.params.get("n_clusters", 2))
        if 1 <= n_clusters < len(ev):
            ax.axvline(n_clusters + 0.5, color="C3", linestyle=":", linewidth=1.0)

    fig.suptitle("Smallest graph Laplacian eigenvalues", fontsize=12)
    out_path = Path(out_path)
    _ensure_dir(out_path.parent)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return out_path


def plot_ari_bars(
    results: list[ClusteringResult],
    out_path: Path,
) -> Path:
    """Grouped bar chart of adjusted Rand index by dataset and method."""
    datasets = []
    for r in results:
        if r.dataset not in datasets:
            datasets.append(r.dataset)
    methods = ["k-means", "spectral"]
    x = np.arange(len(datasets))
    width = 0.35
    fig, ax = plt.subplots(figsize=(7.5, 4.0), constrained_layout=True)
    for i, method in enumerate(methods):
        vals = []
        for name in datasets:
            match = next(r for r in results if r.dataset == name and r.method == method)
            vals.append(match.ari)
        ax.bar(x + (i - 0.5) * width, vals, width, label=method)
    ax.set_xticks(x)
    ax.set_xticklabels(datasets)
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("Adjusted Rand index")
    ax.set_title("Clustering accuracy vs known synthetic labels")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    out_path = Path(out_path)
    _ensure_dir(out_path.parent)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return out_path


def plot_sigma_sensitivity(
    rows: list[SensitivityRow],
    out_path: Path,
    dataset_name: str,
) -> Path:
    """How σ changes mean degree, spectral gap, and ARI stability."""
    subset = [r for r in rows if r.dataset == dataset_name]
    sigmas = [r.sigma for r in subset]
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.6), constrained_layout=True)

    axes[0].plot(sigmas, [r.mean_degree for r in subset], marker="o")
    axes[0].set_xlabel("σ (Gaussian bandwidth)")
    axes[0].set_ylabel("Mean degree (after k-NN)")
    axes[0].set_title("Connectivity")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(sigmas, [r.spectral_gap for r in subset], marker="o", color="C1")
    axes[1].set_xlabel("σ (Gaussian bandwidth)")
    axes[1].set_ylabel("λ_{k+1} − λ_k")
    axes[1].set_title("Spectral gap")
    axes[1].grid(True, alpha=0.3)

    means = np.array([r.ari_mean for r in subset])
    stds = np.array([r.ari_std for r in subset])
    axes[2].plot(sigmas, means, marker="o", color="C2", label="mean ARI")
    axes[2].fill_between(sigmas, means - stds, means + stds, color="C2", alpha=0.25)
    axes[2].set_xlabel("σ (Gaussian bandwidth)")
    axes[2].set_ylabel("Adjusted Rand index")
    axes[2].set_ylim(-0.05, 1.05)
    axes[2].set_title("Stability across seeds")
    axes[2].grid(True, alpha=0.3)

    fig.suptitle(f"Parameter sensitivity — {dataset_name}", fontsize=12)
    out_path = Path(out_path)
    _ensure_dir(out_path.parent)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return out_path


def plot_seed_stability(
    seed_aris: dict[str, list[float]],
    out_path: Path,
    dataset_name: str,
) -> Path:
    """Box-style comparison of ARI across initialization seeds."""
    fig, ax = plt.subplots(figsize=(5.5, 4.0), constrained_layout=True)
    data = [seed_aris["k-means"], seed_aris["spectral"]]
    ax.boxplot(data, tick_labels=["k-means", "spectral"], showmeans=True)
    ax.set_ylim(-0.05, 1.05)
    ax.set_ylabel("Adjusted Rand index")
    ax.set_title(f"Initialization seed stability — {dataset_name}")
    ax.grid(True, axis="y", alpha=0.3)
    out_path = Path(out_path)
    _ensure_dir(out_path.parent)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return out_path
