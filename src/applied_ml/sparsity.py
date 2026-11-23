"""Soft-thresholding and sparse linear solvers (NumPy).

``soft_threshold`` and coordinate-descent LASSO are generalized from MATH 412
mini-transformer coursework scaffolding (Jason Aubrey / University of Arizona).
OLS and Ridge closed forms are portfolio additions for the sparse-head study.
"""

from __future__ import annotations

import numpy as np


def soft_threshold(z: np.ndarray | float, tau: float) -> np.ndarray:
    """Elementwise soft-thresholding: ``sign(z) * max(|z| - tau, 0)``.

    The proximal operator of ``tau * ||·||_1``. Requires ``tau >= 0``.
    """
    if tau < 0:
        raise ValueError("tau must be nonnegative")
    z = np.asarray(z, dtype=float)
    return np.sign(z) * np.maximum(np.abs(z) - tau, 0.0)


def ols(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Ordinary least squares via least-squares solve (minimum-norm if underdetermined)."""
    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float).reshape(-1)
    if A.ndim != 2 or A.shape[0] != b.shape[0]:
        raise ValueError("A must be (m, n) and b must be (m,)")
    x, *_ = np.linalg.lstsq(A, b, rcond=None)
    return x


def ridge(A: np.ndarray, b: np.ndarray, lam: float) -> np.ndarray:
    """Ridge regression: minimize ``||Ax - b||^2 + lam ||x||^2`` (``lam >= 0``)."""
    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float).reshape(-1)
    if A.ndim != 2 or A.shape[0] != b.shape[0]:
        raise ValueError("A must be (m, n) and b must be (m,)")
    if lam < 0:
        raise ValueError("lam must be nonnegative")
    n = A.shape[1]
    # Solve (A^T A + lam I) x = A^T b
    gram = A.T @ A
    gram.flat[:: n + 1] += lam
    return np.linalg.solve(gram, A.T @ b)


def lasso_cd(
    A: np.ndarray,
    b: np.ndarray,
    lam: float,
    max_iter: int = 500,
    tol: float = 1e-6,
) -> np.ndarray:
    """Coordinate-descent LASSO for ``(1/2m)||Ax-b||^2 + lam ||x||_1``.

    Uses the soft-threshold update with column norms. ``lam`` must be
    nonnegative. Columns with zero norm are left at coefficient 0.
    """
    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float).reshape(-1)
    if A.ndim != 2 or A.shape[0] != b.shape[0]:
        raise ValueError("A must be (m, n) and b must be (m,)")
    if lam < 0:
        raise ValueError("lam must be nonnegative")

    m, n = A.shape
    x = np.zeros(n)
    col_norms_sq = (A**2).sum(axis=0)

    for _ in range(max_iter):
        x_old = x.copy()
        for j in range(n):
            if col_norms_sq[j] == 0:
                x[j] = 0.0
                continue
            r_j = b - A @ x + x[j] * A[:, j]
            rho_j = A[:, j] @ r_j
            x[j] = soft_threshold(
                rho_j / col_norms_sq[j],
                lam * m / col_norms_sq[j],
            )
        if np.linalg.norm(x - x_old) < tol:
            break
    return x


def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean squared prediction error."""
    y_true = np.asarray(y_true, dtype=float).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=float).reshape(-1)
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape")
    return float(np.mean((y_true - y_pred) ** 2))
