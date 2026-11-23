# `applied_ml` — shared portfolio helpers

NumPy / SciPy utilities reused by the portfolio projects. Import with
`PYTHONPATH=src` (or via `pytest.ini` / `tests/conftest.py`).

## Modules and dependencies

| Module | Role | Depends on |
| --- | --- | --- |
| `embeddings` | Random token embedding matrix + lookup | NumPy only |
| `attention` | Causal masks, scaled scores, softmax, attention | NumPy only |
| `projections` | Orthogonal weights, FFN, one transformer block | `attention` |
| `sparsity` | Soft-threshold, OLS, Ridge, coordinate-descent LASSO | NumPy only |
| `kmeans` | Lloyd k-means + inertia | NumPy only |
| `spectral` | RBF affinities, Laplacians, spectral embedding | `kmeans`, SciPy |

Spectral clustering code is intentionally separate from the transformer stack.

## Provenance

- Attention, soft-thresholding / LASSO CD, orthogonal projections, and the
  residual transformer block are **generalized** from MATH 412 mini-transformer
  coursework scaffolding (© Jason Aubrey / University of Arizona), with
  portfolio adaptations (tests, docs, OLS/Ridge helpers, weight visualization).
- Embeddings helpers follow the course `mt_utils` random-embedding convention:
  **embeddings are random, not learned**.
- K-means and spectral utilities were written for the spectral clustering study.

## Tests

See `tests/test_attention.py`, `tests/test_sparsity.py`, `tests/test_kmeans.py`,
and `tests/test_spectral.py`.
