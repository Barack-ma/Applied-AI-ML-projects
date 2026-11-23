"""Independent numerical tests for soft-thresholding and sparse linear solvers."""

from __future__ import annotations

import numpy as np
import pytest

from applied_ml.sparsity import lasso_cd, mse, ols, ridge, soft_threshold


def test_soft_threshold_shrinks_and_zeros_small_values():
    z = np.array([-2.0, -0.5, 0.0, 0.4, 1.5])
    out = soft_threshold(z, tau=0.5)
    assert np.allclose(out, np.array([-1.5, 0.0, 0.0, 0.0, 1.0]))


def test_soft_threshold_tau_zero_is_identity():
    z = np.array([1.2, -3.4, 0.0])
    assert np.allclose(soft_threshold(z, 0.0), z)


def test_soft_threshold_rejects_negative_tau():
    with pytest.raises(ValueError):
        soft_threshold(np.array([1.0]), tau=-0.1)


def test_soft_threshold_is_prox_of_l1():
    """For scalar z, soft-threshold solves min_x 0.5(x-z)^2 + tau|x|."""
    z = 1.7
    tau = 0.4
    xs = np.linspace(-3, 3, 601)
    objective = 0.5 * (xs - z) ** 2 + tau * np.abs(xs)
    x_star = xs[np.argmin(objective)]
    assert soft_threshold(z, tau) == pytest.approx(x_star, abs=1e-2)


def test_ols_recovers_noiseless_system():
    A = np.array([[1.0, 0.0], [1.0, 1.0], [1.0, 2.0]])
    beta = np.array([2.0, -1.0])
    b = A @ beta
    hat = ols(A, b)
    assert np.allclose(hat, beta, atol=1e-10)


def test_ridge_shrinks_toward_zero_relative_to_ols():
    rng = np.random.default_rng(0)
    A = rng.normal(size=(40, 8))
    b = A @ rng.normal(size=8) + 0.1 * rng.normal(size=40)
    ols_coef = ols(A, b)
    ridge_coef = ridge(A, b, lam=5.0)
    assert np.linalg.norm(ridge_coef) < np.linalg.norm(ols_coef)


def test_lasso_promotes_sparsity_on_sparse_truth():
    rng = np.random.default_rng(1)
    m, n = 100, 20
    A = rng.normal(size=(m, n))
    beta = np.zeros(n)
    beta[[1, 5, 12]] = [2.0, -1.5, 1.0]
    b = A @ beta + 0.05 * rng.normal(size=m)

    ols_coef = ols(A, b)
    lasso_coef = lasso_cd(A, b, lam=0.2)

    ols_nz = int(np.sum(np.abs(ols_coef) > 1e-6))
    lasso_nz = int(np.sum(np.abs(lasso_coef) > 1e-6))
    assert lasso_nz < ols_nz
    # Recovered support should concentrate near the true indices.
    recovered = set(np.where(np.abs(lasso_coef) > 1e-3)[0].tolist())
    assert {1, 5, 12}.issubset(recovered) or len(recovered & {1, 5, 12}) >= 2


def test_lasso_lam_zero_close_to_ols_well_conditioned():
    rng = np.random.default_rng(2)
    A = rng.normal(size=(50, 5))
    b = A @ np.array([1.0, -2.0, 0.5, 0.0, 3.0])
    assert np.allclose(lasso_cd(A, b, lam=0.0, max_iter=2000), ols(A, b), atol=1e-4)


def test_mse_matches_manual_mean():
    y = np.array([1.0, 2.0, 3.0])
    pred = np.array([1.0, 1.0, 5.0])
    assert mse(y, pred) == pytest.approx((0.0 + 1.0 + 4.0) / 3.0)
