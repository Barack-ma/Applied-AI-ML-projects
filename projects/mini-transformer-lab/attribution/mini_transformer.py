"""Your cumulative mini-transformer: Coding HW11.

START HERE
See coding/Mini_Transformer_Project_Guide.md for the full file workflow.
Project companion: save/reuse the work already assigned in the notebook.
The notebook remains the graded artifact for HW09–HW15; keep this file for
your cumulative project. The HW12 portfolio checkpoint remains optional.

1. Save this assignment's file beside its notebook (keep a backup of older work).
2. Search for THIS WEEK below. Earlier model components are supplied and working.
3. Read the input/output shapes and follow the numbered comments.
4. Save edits. In the notebook run:
       import importlib
       import mini_transformer as mt
       mt = importlib.reload(mt)
       mt.check_milestone()
   Or run `python mini_transformer.py` in a terminal in this folder.
5. A failed check identifies the small task to revisit. Checks are examples,
   not a complete grader. A check is only passed when you see the success message.

SUPPLIED CODE: all sections before THIS WEEK. Do not reimplement them.
No notebook variables or instructor mt_utils imports are needed by this file.
"""

import numpy as np


def causal_mask(length):
    """Return a lower-triangular 0/1 causal mask."""
    return np.tril(np.ones((length, length), dtype=int))


def scaled_dot_product_scores(Q, K):
    Q = np.asarray(Q, dtype=float)
    K = np.asarray(K, dtype=float)
    if Q.ndim != 2 or K.ndim != 2 or Q.shape[1] != K.shape[1]:
        raise ValueError("Q and K must be 2-D with matching feature dimension")
    return (Q @ K.T) / np.sqrt(Q.shape[1])


def softmax(X):
    X = np.asarray(X, dtype=float)
    shifted = X - np.max(X, axis=-1, keepdims=True)
    exponentials = np.exp(shifted)
    return exponentials / np.sum(exponentials, axis=-1, keepdims=True)


def attention(Q, K, V, mask=None):
    scores = scaled_dot_product_scores(Q, K)
    V = np.asarray(V, dtype=float)
    if V.ndim != 2 or V.shape[0] != scores.shape[1]:
        raise ValueError("V must be 2-D with one row per key")
    if mask is not None:
        mask = np.asarray(mask)
        if mask.shape != scores.shape:
            raise ValueError("mask must have the same shape as the score matrix")
        allowed = mask.astype(bool)
        if not np.all(allowed.any(axis=1)):
            raise ValueError("every query must have at least one allowed key")
        scores = np.where(allowed, scores, -np.inf)
    return softmax(scores) @ V


def attention_head(X, W_Q, W_K, W_V, mask=None):
    """Project X to Q, K, V and return one attention-head output."""
    X = np.asarray(X, dtype=float)
    return attention(X @ W_Q, X @ W_K, X @ W_V, mask=mask)


def orthogonal_matrix(rows, cols=None, seed=0):
    """Return a reproducible rows-by-cols matrix with orthonormal short side."""
    if cols is None:
        cols = rows
    if not isinstance(rows, (int, np.integer)) or not isinstance(cols, (int, np.integer)):
        raise TypeError("rows and cols must be integers")
    if rows <= 0 or cols <= 0:
        raise ValueError("rows and cols must be positive")
    rng = np.random.default_rng(seed)
    if rows >= cols:
        Q, _ = np.linalg.qr(rng.standard_normal((rows, cols)), mode="reduced")
        return Q
    Q, _ = np.linalg.qr(rng.standard_normal((cols, rows)), mode="reduced")
    return Q.T


def orthogonal_projection_matrices(d_model, d_head, seed=0):
    """Return distinct orthogonal (W_Q, W_K, W_V) weight matrices."""
    if not isinstance(d_model, (int, np.integer)) or not isinstance(d_head, (int, np.integer)):
        raise TypeError("d_model and d_head must be integers")
    if d_model <= 0 or d_head <= 0:
        raise ValueError("d_model and d_head must be positive")
    return tuple(orthogonal_matrix(d_model, d_head, seed=seed + offset) for offset in range(3))


def relu(Z):
    """Apply max(0, z) entry by entry."""
    return np.maximum(np.asarray(Z, dtype=float), 0.0)


def _validated_feed_forward_inputs(X, W1, b1, W2, b2):
    X = np.asarray(X, dtype=float)
    W1 = np.asarray(W1, dtype=float)
    b1 = np.asarray(b1, dtype=float)
    W2 = np.asarray(W2, dtype=float)
    b2 = np.asarray(b2, dtype=float)
    if X.ndim != 2 or W1.ndim != 2 or W2.ndim != 2 or b1.ndim != 1 or b2.ndim != 1:
        raise ValueError("X, W1, W2 must be matrices and b1, b2 must be vectors")
    _, d_model = X.shape
    if W1.shape[0] != d_model:
        raise ValueError("W1 must have one row per model feature")
    d_hidden = W1.shape[1]
    if b1.shape != (d_hidden,) or W2.shape != (d_hidden, d_model) or b2.shape != (d_model,):
        raise ValueError("incompatible feed-forward shapes")
    return X, W1, b1, W2, b2


def feed_forward(X, W1, b1, W2, b2):
    X, W1, b1, W2, b2 = _validated_feed_forward_inputs(X, W1, b1, W2, b2)
    hidden = relu(X @ W1 + b1)
    return hidden @ W2 + b2


def transformer_block(X, W_Q, W_K, W_V, W1, b1, W2, b2, mask=None):
    X = np.asarray(X, dtype=float)
    attended = attention_head(X, W_Q, W_K, W_V, mask=mask)
    if attended.shape != X.shape:
        raise ValueError("attention output must match X for a residual addition")
    after_attention = X + attended
    return after_attention + feed_forward(after_attention, W1, b1, W2, b2)


def _validated_projection_inputs(X, Q, atol=1e-10):
    """Convert arrays and verify that the columns of Q are orthonormal."""
    X = np.asarray(X, dtype=float)
    Q = np.asarray(Q, dtype=float)
    if X.ndim != 2 or Q.ndim != 2:
        raise ValueError("X and Q must both be matrices")
    if Q.shape[0] != X.shape[1] or Q.shape[1] == 0 or Q.shape[1] > Q.shape[0]:
        raise ValueError("Q must have one row per feature and at least one column")
    if not np.allclose(Q.T @ Q, np.eye(Q.shape[1]), atol=atol, rtol=0):
        raise ValueError("the columns of Q must be orthonormal")
    return X, Q


def project_features(X, Q, atol=1e-10):
    """Project each token row of X onto the feature subspace spanned by Q."""
    X, Q = _validated_projection_inputs(X, Q, atol=atol)
    return X @ Q @ Q.T


def _validated_low_rank_inputs(W, k):
    W = np.asarray(W, dtype=float)
    if W.ndim != 2 or 0 in W.shape:
        raise ValueError("W must be a nonempty matrix")
    if not isinstance(k, (int, np.integer)):
        raise TypeError("k must be an integer")
    if k < 0 or k > min(W.shape):
        raise ValueError("k must satisfy 0 <= k <= min(W.shape)")
    return W, int(k)


def low_rank_weight(W, k):
    """Return the best rank-k approximation of W in Frobenius norm."""
    W, k = _validated_low_rank_inputs(W, k)
    U, singular_values, Vt = np.linalg.svd(W, full_matrices=False)
    return (U[:, :k] * singular_values[:k]) @ Vt[:k, :]


# -------------------------------------------------------------------------
# Coding HW08: control the worst-case amplification of a learned weight
def spectral_normalize(W, target=1.0):
    """Return a copy of W scaled to have spectral norm ``target``."""
    W = np.asarray(W, dtype=float)
    if W.ndim != 2 or 0 in W.shape:
        raise ValueError("W must be a nonempty matrix")
    if not np.isscalar(target) or isinstance(target, (str, bytes)):
        raise TypeError("target must be a positive finite scalar")
    target = float(target)
    if not np.isfinite(target) or target <= 0:
        raise ValueError("target must be a positive finite scalar")
    sigma_max = np.linalg.norm(W, ord=2)
    if sigma_max == 0:
        raise ValueError("the zero matrix cannot be spectrally normalized")
    return (target / sigma_max) * W


# SUPPLIED COMPLETED CHECKPOINT: PCA reconstruction
class PCA:
    def __init__(self, n_components):
        self.k = n_components
        self.components = None   # (k, d) — each row is a principal component
        self.mean = None
        self.explained_variance_ratio = None

    def fit(self, X):
        n, d = X.shape
        self.mean = X.mean(axis=0)
        Xc = X - self.mean                        # center
        Sigma = Xc.T @ Xc / (n - 1)              # covariance (d x d)
        evals, evecs = np.linalg.eigh(Sigma)      # eigenvectors as columns
        # Sort descending
        idx = np.argsort(-evals)
        evals, evecs = evals[idx], evecs[:, idx]
        self.components = evecs[:, :self.k].T     # (k, d)
        self.explained_variance_ratio = evals[:self.k] / evals.sum()
        return self

    def transform(self, X):
        Xc = X - self.mean
        return Xc @ self.components.T             # (n, k)

    def fit_transform(self, X):
        return self.fit(X).transform(X)

class PCAWithReconstruct(PCA):
    def reconstruct(self, Z):
        # SUPPLIED: completed earlier notebook method.
        # 1. Z is (n,k); components is (k,d). Map back with @.
        # 2. Add self.mean (d,) to each reconstructed row.
        # 3. Return the (n,d) result; self means this fitted PCA object.
        return Z @ self.components + self.mean


# SUPPLIED COMPLETED CHECKPOINT: A sparse linear output head
def soft_threshold(z, tau):
    return np.sign(z) * np.maximum(np.abs(z) - tau, 0)

def lasso_cd(A, b, lam, max_iter=500, tol=1e-6):
    # 1. A=(m,n) holds feature rows; b=(m,) holds targets.
    #    Use nonzero feature columns and nonnegative lam.
    # 2. The loop below updates one coefficient at a time.
    # 3. Return x=(n,); predictions for new rows H are H @ x.
    m, n = A.shape
    x = np.zeros(n)
    # Precompute column norms squared
    col_norms_sq = (A**2).sum(axis=0)
    
    for _ in range(max_iter):
        x_old = x.copy()
        for j in range(n):
            # Partial residual: what b looks like if we remove x_j's contribution
            r_j = b - A @ x + x[j] * A[:, j]
            rho_j = A[:, j] @ r_j
            x[j] = soft_threshold(rho_j / col_norms_sq[j],
                                   lam * m / col_norms_sq[j])
        if np.linalg.norm(x - x_old) < tol:
            break
    return x


# THIS WEEK: Clustering token features
# Copy your completed cluster_sizes method from the notebook into the matching
# method below. kmeans and the dataclass structure are supplied.
# Group nearby feature rows and count their members. The random embeddings in this lab have not learned word meanings.

from dataclasses import dataclass

@dataclass
class ClusterResult:
    labels: np.ndarray
    centers: np.ndarray
    inertia: float
    n_iter: int

    def n_clusters(self):
        return len(self.centers)

    def cluster_sizes(self):
        counts = {}
        for cluster_id in range(self.n_clusters()):
            matches = self.labels == cluster_id
            counts[cluster_id] = int(np.sum(matches))
        return counts


def kmeans(X, k, max_iter=100, n_init=5, seed=0):
    rng = np.random.default_rng(seed)
    best = None

    for init in range(n_init):
        # Initialize centroids by picking k random points
        idx = rng.choice(len(X), k, replace=False)
        centers = X[idx].copy()

        for it in range(max_iter):
            # Assignment: each point -> nearest center
            dists = np.linalg.norm(X[:, None, :] - centers[None, :, :], axis=2)
            labels = dists.argmin(axis=1)

            # Update: move centers to cluster means
            new_centers = np.array([
                X[labels == j].mean(axis=0) if (labels == j).any() else centers[j]
                for j in range(k)
            ])

            # Exact equality is sufficient for this small Lloyd implementation:
            # unchanged groups produce the same means on the next update.
            if np.array_equal(centers, new_centers):
                break
            centers = new_centers

        # The last update can change which center is nearest. Refresh labels
        # even when the iteration limit, rather than convergence, stopped us.
        dists = np.linalg.norm(X[:, None, :] - centers[None, :, :], axis=2)
        labels = dists.argmin(axis=1)
        # Inertia uses these returned labels and centers, not old assignments.
        inertia = sum(np.linalg.norm(X[labels==j] - centers[j])**2
                      for j in range(k) if (labels==j).any())
        result = ClusterResult(labels, centers, inertia, it+1)
        if best is None or inertia < best.inertia:
            best = result

    return best

# SUPPLIED SELF-CHECKS: run these; do not edit them to make a test pass.
def check_milestone():
    """Supplied small examples; run these before the notebook experiment."""
    result = ClusterResult(np.array([1, 0, 1]), np.zeros((3, 2)), 0., 1)
    counts = result.cluster_sizes()
    assert counts is not None, "Copy your completed cluster_sizes method into the marked location."
    assert counts == {0: 1, 1: 2, 2: 0}, "Include every center, even one with no assigned rows."
    points = np.array([[0., 0.], [0., 2.]])
    result = kmeans(points, k=1)
    assert np.allclose(result.centers, [[0., 1.]]), "A single center is the mean of all points."
    assert np.isclose(result.inertia, 2.), "Inertia is the sum of squared distances."
    print("HW11 project checks passed. Continue with the notebook experiment.")


if __name__ == "__main__":
    # Running this file checks the current milestone. Importing it runs no checks.
    check_milestone()

# Copyright 2026 Jason Aubrey / University of Arizona. All rights reserved.
