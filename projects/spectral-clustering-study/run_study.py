#!/usr/bin/env python3
"""Reproduce the spectral clustering study end-to-end.

Usage (from the repository root)::

    PYTHONPATH=src python projects/spectral-clustering-study/run_study.py

Figures are written to ``assets/figures/spectral-clustering-study/``.
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

from datasets import all_datasets  # noqa: E402
from experiments import (  # noqa: E402
    compare_methods,
    default_sigma,
    seed_stability,
    sigma_sensitivity,
)
from plotting import (  # noqa: E402
    plot_ari_bars,
    plot_eigenvalues,
    plot_seed_stability,
    plot_side_by_side,
    plot_sigma_sensitivity,
)

RANDOM_STATE = 0
N_NEIGHBORS = 12
FIGURE_DIR = _REPO_ROOT / "assets" / "figures" / "spectral-clustering-study"
RESULT_DIR = _PROJECT_DIR / "results"


def main() -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    datasets = all_datasets(random_state=RANDOM_STATE)
    results = compare_methods(
        datasets, n_neighbors=N_NEIGHBORS, random_state=RANDOM_STATE
    )

    # Side-by-side cluster plots per dataset.
    for data in datasets:
        km = next(r for r in results if r.dataset == data.name and r.method == "k-means")
        sp = next(r for r in results if r.dataset == data.name and r.method == "spectral")
        plot_side_by_side(
            data,
            km,
            sp,
            FIGURE_DIR / f"side_by_side_{data.name}.png",
        )

    plot_eigenvalues(results, FIGURE_DIR / "laplacian_eigenvalues.png")
    plot_ari_bars(results, FIGURE_DIR / "ari_comparison.png")

    # σ sensitivity on the two non-convex geometries (and blobs for contrast).
    sensitivity_summary: dict[str, list[dict]] = {}
    for data in datasets:
        sigma0 = default_sigma(data, n_neighbors=N_NEIGHBORS)
        # Log-spaced sweep around the heuristic bandwidth.
        sigmas = np.unique(
            np.clip(
                sigma0 * np.array([0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0]),
                0.05,
                None,
            )
        )
        rows = sigma_sensitivity(
            data,
            sigmas,
            n_neighbors=N_NEIGHBORS,
            seeds=[0, 1, 2, 3, 4],
        )
        plot_sigma_sensitivity(
            rows,
            FIGURE_DIR / f"sigma_sensitivity_{data.name}.png",
            data.name,
        )
        sensitivity_summary[data.name] = [
            {
                "sigma": r.sigma,
                "mean_degree": r.mean_degree,
                "edge_density": r.edge_density,
                "spectral_gap": r.spectral_gap,
                "ari_mean": r.ari_mean,
                "ari_std": r.ari_std,
            }
            for r in rows
        ]

        seed_aris = seed_stability(
            data,
            sigma=sigma0,
            n_neighbors=N_NEIGHBORS,
            seeds=list(range(10)),
        )
        plot_seed_stability(
            seed_aris,
            FIGURE_DIR / f"seed_stability_{data.name}.png",
            data.name,
        )

    summary = {
        "random_state": RANDOM_STATE,
        "n_neighbors": N_NEIGHBORS,
        "comparison": [
            {
                "dataset": r.dataset,
                "method": r.method,
                "ari": r.ari,
                "inertia_ambient": r.inertia_ambient,
                "params": r.params,
                "eigenvalues": None
                if r.eigenvalues is None
                else [float(v) for v in r.eigenvalues],
            }
            for r in results
        ],
        "sigma_sensitivity": sensitivity_summary,
    }
    summary_path = RESULT_DIR / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"Wrote figures to {FIGURE_DIR}")
    print(f"Wrote metrics to {summary_path}")
    print("\nAdjusted Rand index summary:")
    for r in results:
        print(f"  {r.dataset:8s}  {r.method:9s}  ARI={r.ari:.3f}")


if __name__ == "__main__":
    main()
