"""Experiment runners for the Mini-Transformer Lab."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from applied_ml.projections import orthogonal_projection_matrices
from applied_ml.sparsity import lasso_cd, mse, ols, ridge

from model import MiniTransformer, MiniTransformerConfig


@dataclass
class AttentionCompareResult:
    token_ids: np.ndarray
    weights_unmasked: np.ndarray
    weights_causal: np.ndarray
    frobenius_diff: float


@dataclass
class ProjectionSensitivityResult:
    baseline_norm: float
    perturb_W_delta: float
    perturb_X_delta: float
    relative_W: float
    relative_X: float


@dataclass
class SparseHeadResult:
    true_beta: np.ndarray
    ols_coef: np.ndarray
    ridge_coefs: dict[float, np.ndarray]
    lasso_coefs: dict[float, np.ndarray]
    train_mse: dict[str, float]
    test_mse: dict[str, float]
    n_nonzero: dict[str, int]
    lambdas: list[float]


def default_token_ids(n_tokens: int = 8, vocab_size: int = 32, seed: int = 412) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, vocab_size, size=n_tokens)


def compare_attention_masks(
    model: MiniTransformer | None = None,
    token_ids: np.ndarray | None = None,
) -> AttentionCompareResult:
    """Causal vs fully visible attention weight matrices on the same inputs."""
    model = model or MiniTransformer.create()
    token_ids = (
        np.asarray(token_ids, dtype=int)
        if token_ids is not None
        else default_token_ids(seed=model.config.seed)
    )
    w_full = model.attention_matrix(token_ids, causal=False)
    w_causal = model.attention_matrix(token_ids, causal=True)
    diff = float(np.linalg.norm(w_full - w_causal, ord="fro"))
    return AttentionCompareResult(token_ids, w_full, w_causal, diff)


def projection_sensitivity(
    model: MiniTransformer | None = None,
    token_ids: np.ndarray | None = None,
    noise_scale: float = 0.25,
) -> ProjectionSensitivityResult:
    """Measure how block outputs change under projection / input perturbations."""
    model = model or MiniTransformer.create()
    token_ids = (
        np.asarray(token_ids, dtype=int)
        if token_ids is not None
        else default_token_ids(seed=model.config.seed)
    )
    baseline = model.forward(token_ids, causal=True)
    base_norm = float(np.linalg.norm(baseline))

    # New orthogonal projections from a different seed.
    W_Q, W_K, W_V = orthogonal_projection_matrices(
        model.config.d_model, model.config.d_head, seed=model.config.seed + 99
    )
    out_W = model.forward(token_ids, causal=True, W_Q=W_Q, W_K=W_K, W_V=W_V)
    delta_W = float(np.linalg.norm(out_W - baseline))

    rng = np.random.default_rng(model.config.seed + 7)
    X = model.embed(token_ids)
    X_noisy = X + noise_scale * rng.standard_normal(X.shape)
    out_X = model.forward(token_ids, causal=True, X=X_noisy)
    delta_X = float(np.linalg.norm(out_X - baseline))

    return ProjectionSensitivityResult(
        baseline_norm=base_norm,
        perturb_W_delta=delta_W,
        perturb_X_delta=delta_X,
        relative_W=delta_W / max(base_norm, 1e-12),
        relative_X=delta_X / max(base_norm, 1e-12),
    )


def _synthetic_sparse_design(
    n_train: int,
    n_test: int,
    n_features: int,
    n_nonzero: int,
    noise_std: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Sparse linear target: ``y = H β* + ε`` with ``β*`` having ``n_nonzero`` entries.

    Feature rows are drawn from a mini-transformer block applied to random
    token sequences (random embeddings → fixed weights), then standardized.
    This ties the sparse head to the transformer representation while keeping
    the prediction target mathematically explicit.
    """
    rng = np.random.default_rng(seed)
    cfg = MiniTransformerConfig(
        vocab_size=48, d_model=n_features, d_head=n_features, d_hidden=2 * n_features, seed=seed
    )
    model = MiniTransformer.create(cfg)

    def features(n: int, stream: int) -> np.ndarray:
        local = np.random.default_rng(seed + stream)
        rows = []
        for i in range(n):
            tokens = local.integers(0, cfg.vocab_size, size=6)
            H = model.forward(tokens, causal=True)
            # Mean-pool token representations into one feature row.
            rows.append(H.mean(axis=0))
        return np.asarray(rows, dtype=float)

    H_train = features(n_train, 1)
    H_test = features(n_test, 2)

    # Standardize using training statistics.
    mu = H_train.mean(axis=0)
    sigma = H_train.std(axis=0)
    sigma = np.where(sigma < 1e-8, 1.0, sigma)
    H_train = (H_train - mu) / sigma
    H_test = (H_test - mu) / sigma

    beta = np.zeros(n_features)
    support = rng.choice(n_features, size=n_nonzero, replace=False)
    beta[support] = rng.choice([-2.0, -1.5, 1.5, 2.0], size=n_nonzero)

    y_train = H_train @ beta + noise_std * rng.standard_normal(n_train)
    y_test = H_test @ beta + noise_std * rng.standard_normal(n_test)
    return H_train, y_train, H_test, y_test, beta


def sparse_linear_head_study(
    *,
    n_train: int = 80,
    n_test: int = 40,
    n_features: int = 16,
    n_nonzero: int = 4,
    noise_std: float = 0.5,
    lambdas: list[float] | None = None,
    seed: int = 412,
) -> SparseHeadResult:
    """Compare OLS, Ridge, and LASSO on a known sparse synthetic target."""
    if lambdas is None:
        lambdas = [0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0]

    H_tr, y_tr, H_te, y_te, beta = _synthetic_sparse_design(
        n_train, n_test, n_features, n_nonzero, noise_std, seed
    )

    ols_coef = ols(H_tr, y_tr)
    ridge_coefs: dict[float, np.ndarray] = {}
    lasso_coefs: dict[float, np.ndarray] = {}
    train_mse: dict[str, float] = {"ols": mse(y_tr, H_tr @ ols_coef)}
    test_mse: dict[str, float] = {"ols": mse(y_te, H_te @ ols_coef)}
    n_nz: dict[str, int] = {"ols": int(np.sum(np.abs(ols_coef) > 1e-8))}

    for lam in lambdas:
        r = ridge(H_tr, y_tr, lam)
        l = lasso_cd(H_tr, y_tr, lam)
        ridge_coefs[lam] = r
        lasso_coefs[lam] = l
        train_mse[f"ridge_{lam}"] = mse(y_tr, H_tr @ r)
        test_mse[f"ridge_{lam}"] = mse(y_te, H_te @ r)
        train_mse[f"lasso_{lam}"] = mse(y_tr, H_tr @ l)
        test_mse[f"lasso_{lam}"] = mse(y_te, H_te @ l)
        n_nz[f"ridge_{lam}"] = int(np.sum(np.abs(r) > 1e-8))
        n_nz[f"lasso_{lam}"] = int(np.sum(np.abs(l) > 1e-8))

    return SparseHeadResult(
        true_beta=beta,
        ols_coef=ols_coef,
        ridge_coefs=ridge_coefs,
        lasso_coefs=lasso_coefs,
        train_mse=train_mse,
        test_mse=test_mse,
        n_nonzero=n_nz,
        lambdas=list(lambdas),
    )


def best_lambda_by_test_mse(result: SparseHeadResult, method: str) -> tuple[float, float]:
    """Return ``(lambda, test_mse)`` with lowest test error for ridge or lasso."""
    best_lam = result.lambdas[0]
    best_err = result.test_mse[f"{method}_{best_lam}"]
    for lam in result.lambdas[1:]:
        err = result.test_mse[f"{method}_{lam}"]
        if err < best_err:
            best_err = err
            best_lam = lam
    return best_lam, best_err
