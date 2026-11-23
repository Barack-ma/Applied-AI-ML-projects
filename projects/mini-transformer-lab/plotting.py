"""Figure generation for the Mini-Transformer Lab."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from experiments import (
    AttentionCompareResult,
    ProjectionSensitivityResult,
    SparseHeadResult,
)


def _ensure_parent(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def plot_attention_heatmaps(
    result: AttentionCompareResult,
    out_path: Path,
) -> Path:
    """Side-by-side unmasked vs causal attention weight heatmaps."""
    out_path = _ensure_parent(out_path)
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.2), constrained_layout=True)
    panels = [
        ("Unmasked attention weights", result.weights_unmasked),
        ("Causal (lower-triangular) mask", result.weights_causal),
    ]
    vmax = max(float(w.max()) for _, w in panels)
    for ax, (title, W) in zip(axes, panels):
        im = ax.imshow(W, cmap="viridis", vmin=0.0, vmax=vmax, aspect="equal")
        ax.set_title(title)
        ax.set_xlabel("Key position")
        ax.set_ylabel("Query position")
        ax.set_xticks(range(W.shape[1]))
        ax.set_yticks(range(W.shape[0]))
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle(
        f"Attention softmax weights (‖Δ‖_F = {result.frobenius_diff:.3f})",
        fontsize=12,
    )
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_projection_sensitivity(
    result: ProjectionSensitivityResult,
    out_path: Path,
) -> Path:
    """Bar chart of absolute / relative output change under perturbations."""
    out_path = _ensure_parent(out_path)
    labels = ["New orthogonal\nprojections (W)", "Noisy inputs\n(X)"]
    abs_vals = [result.perturb_W_delta, result.perturb_X_delta]
    rel_vals = [result.relative_W, result.relative_X]

    fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.8), constrained_layout=True)
    axes[0].bar(labels, abs_vals, color=["#2a6f97", "#ee6c4d"])
    axes[0].set_ylabel(r"$\|Y' - Y\|_F$")
    axes[0].set_title("Absolute output change")

    axes[1].bar(labels, rel_vals, color=["#2a6f97", "#ee6c4d"])
    axes[1].set_ylabel(r"$\|Y' - Y\|_F / \|Y\|_F$")
    axes[1].set_title("Relative output change")

    fig.suptitle(
        f"Projection / input sensitivity (baseline ‖Y‖_F = {result.baseline_norm:.3f})",
        fontsize=12,
    )
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_coefficient_paths(
    result: SparseHeadResult,
    out_path: Path,
) -> Path:
    """Coefficient-vs-regularization paths for Ridge and LASSO."""
    out_path = _ensure_parent(out_path)
    lambdas = np.asarray(result.lambdas, dtype=float)
    n_features = result.true_beta.shape[0]

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), constrained_layout=True)
    for ax, method, store in (
        (axes[0], "Ridge", result.ridge_coefs),
        (axes[1], "LASSO", result.lasso_coefs),
    ):
        coef_matrix = np.vstack([store[lam] for lam in result.lambdas])
        for j in range(n_features):
            support = abs(result.true_beta[j]) > 1e-12
            ax.plot(
                lambdas,
                coef_matrix[:, j],
                lw=1.6 if support else 0.8,
                alpha=1.0 if support else 0.35,
                label=f"β[{j}]" if support else None,
            )
        ax.axhline(0.0, color="black", lw=0.6)
        ax.set_xscale("log")
        ax.set_xlabel(r"Regularization $\lambda$")
        ax.set_ylabel("Coefficient value")
        ax.set_title(f"{method} coefficient paths")
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(fontsize=8, loc="best")

    fig.suptitle(
        "Sparse linear head: coefficients vs regularization "
        f"(true support size = {int(np.sum(np.abs(result.true_beta) > 0))})",
        fontsize=12,
    )
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_prediction_errors(
    result: SparseHeadResult,
    out_path: Path,
) -> Path:
    """Train/test MSE for OLS and regularized methods across λ."""
    out_path = _ensure_parent(out_path)
    lambdas = np.asarray(result.lambdas, dtype=float)

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0), constrained_layout=True)
    ols_train = result.train_mse["ols"]
    ols_test = result.test_mse["ols"]

    for ax, split, ols_val in (
        (axes[0], "train", ols_train),
        (axes[1], "test", ols_test),
    ):
        ridge_err = [result.train_mse[f"ridge_{lam}"] if split == "train"
                     else result.test_mse[f"ridge_{lam}"] for lam in result.lambdas]
        lasso_err = [result.train_mse[f"lasso_{lam}"] if split == "train"
                     else result.test_mse[f"lasso_{lam}"] for lam in result.lambdas]
        ax.axhline(ols_val, color="#6c757d", ls="--", label=f"OLS ({split} MSE)")
        ax.plot(lambdas, ridge_err, "o-", label="Ridge", color="#2a6f97")
        ax.plot(lambdas, lasso_err, "s-", label="LASSO", color="#ee6c4d")
        ax.set_xscale("log")
        ax.set_xlabel(r"Regularization $\lambda$")
        ax.set_ylabel("Mean squared error")
        ax.set_title(f"{split.capitalize()} prediction error")
        ax.legend(fontsize=8)

    fig.suptitle("OLS vs Ridge vs LASSO prediction errors", fontsize=12)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_sparsity_vs_lambda(
    result: SparseHeadResult,
    out_path: Path,
) -> Path:
    """Number of nonzero coefficients vs λ (highlights LASSO sparsity)."""
    out_path = _ensure_parent(out_path)
    lambdas = np.asarray(result.lambdas, dtype=float)
    ridge_nz = [result.n_nonzero[f"ridge_{lam}"] for lam in result.lambdas]
    lasso_nz = [result.n_nonzero[f"lasso_{lam}"] for lam in result.lambdas]
    true_nz = int(np.sum(np.abs(result.true_beta) > 0))

    fig, ax = plt.subplots(figsize=(7.0, 4.0), constrained_layout=True)
    ax.axhline(true_nz, color="#6c757d", ls="--", label=f"True support ({true_nz})")
    ax.axhline(result.n_nonzero["ols"], color="#adb5bd", ls=":", label="OLS nonzeros")
    ax.plot(lambdas, ridge_nz, "o-", label="Ridge nonzeros", color="#2a6f97")
    ax.plot(lambdas, lasso_nz, "s-", label="LASSO nonzeros", color="#ee6c4d")
    ax.set_xscale("log")
    ax.set_xlabel(r"Regularization $\lambda$")
    ax.set_ylabel("Number of nonzero coefficients")
    ax.set_title("Sparsity vs regularization")
    ax.legend(fontsize=8)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path
