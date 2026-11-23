#!/usr/bin/env python3
"""Reproduce the Mini-Transformer Lab end-to-end.

Usage (from the repository root)::

    PYTHONPATH=src python3 projects/mini-transformer-lab/run_lab.py

Figures are written to ``assets/figures/mini-transformer-lab/``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

_PROJECT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _PROJECT_DIR.parents[1]
sys.path.insert(0, str(_REPO_ROOT / "src"))
sys.path.insert(0, str(_PROJECT_DIR))

from experiments import (  # noqa: E402
    best_lambda_by_test_mse,
    compare_attention_masks,
    projection_sensitivity,
    sparse_linear_head_study,
)
from model import MiniTransformer, MiniTransformerConfig  # noqa: E402
from plotting import (  # noqa: E402
    plot_attention_heatmaps,
    plot_coefficient_paths,
    plot_prediction_errors,
    plot_projection_sensitivity,
    plot_sparsity_vs_lambda,
)

RANDOM_STATE = 412
FIGURE_DIR = _REPO_ROOT / "assets" / "figures" / "mini-transformer-lab"
RESULT_DIR = _PROJECT_DIR / "results"


def main() -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    model = MiniTransformer.create(MiniTransformerConfig(seed=RANDOM_STATE))

    attn = compare_attention_masks(model)
    plot_attention_heatmaps(attn, FIGURE_DIR / "attention_causal_vs_unmasked.png")

    sens = projection_sensitivity(model)
    plot_projection_sensitivity(sens, FIGURE_DIR / "projection_sensitivity.png")

    sparse = sparse_linear_head_study(seed=RANDOM_STATE)
    plot_coefficient_paths(sparse, FIGURE_DIR / "coefficient_paths.png")
    plot_prediction_errors(sparse, FIGURE_DIR / "prediction_errors.png")
    plot_sparsity_vs_lambda(sparse, FIGURE_DIR / "sparsity_vs_lambda.png")

    best_ridge_lam, best_ridge_mse = best_lambda_by_test_mse(sparse, "ridge")
    best_lasso_lam, best_lasso_mse = best_lambda_by_test_mse(sparse, "lasso")

    summary = {
        "disclaimer": (
            "Small NumPy pedagogical transformer with random (not learned) "
            "embeddings; not a pretrained language model."
        ),
        "random_state": RANDOM_STATE,
        "embeddings": "random",
        "attention": {
            "token_ids": attn.token_ids.tolist(),
            "frobenius_diff_unmasked_vs_causal": attn.frobenius_diff,
            "causal_upper_triangle_max_weight": float(
                np.triu(attn.weights_causal, k=1).max()
            ),
        },
        "projection_sensitivity": {
            "baseline_norm": sens.baseline_norm,
            "perturb_W_delta": sens.perturb_W_delta,
            "perturb_X_delta": sens.perturb_X_delta,
            "relative_W": sens.relative_W,
            "relative_X": sens.relative_X,
        },
        "sparse_head": {
            "true_beta": sparse.true_beta.tolist(),
            "true_support": [
                int(i) for i, v in enumerate(sparse.true_beta) if abs(v) > 0
            ],
            "ols_test_mse": sparse.test_mse["ols"],
            "best_ridge": {"lambda": best_ridge_lam, "test_mse": best_ridge_mse},
            "best_lasso": {"lambda": best_lasso_lam, "test_mse": best_lasso_mse},
            "n_nonzero": sparse.n_nonzero,
            "train_mse": sparse.train_mse,
            "test_mse": sparse.test_mse,
            "lambdas": sparse.lambdas,
        },
    }
    summary_path = RESULT_DIR / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))

    print(f"Wrote figures to {FIGURE_DIR}")
    print(f"Wrote metrics to {summary_path}")
    print(
        f"Attention ‖Δ‖_F (unmasked vs causal) = {attn.frobenius_diff:.4f}; "
        f"causal upper-tri max = {summary['attention']['causal_upper_triangle_max_weight']:.2e}"
    )
    print(
        f"Sparse head test MSE — OLS={sparse.test_mse['ols']:.4f}, "
        f"best Ridge(λ={best_ridge_lam})={best_ridge_mse:.4f}, "
        f"best LASSO(λ={best_lasso_lam})={best_lasso_mse:.4f}"
    )


if __name__ == "__main__":
    main()
