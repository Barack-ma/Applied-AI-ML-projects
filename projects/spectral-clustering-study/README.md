# Spectral Clustering Study

**Author:** Barack Marwanga  
**Portfolio project:** Applied AI & Machine Learning Projects  
**Question:** Why do graph-based representations recover some cluster structures that ordinary Euclidean k-means struggles to separate?

This study is intentionally **synthetic**: three labeled toy geometries from scikit-learn. The goal is mechanistic clarity (when and why the Laplacian embedding helps), not a claim about real-world datasets.

---

## Research question

Ordinary k-means partitions points by Euclidean proximity to centroids. That objective is natural for compact, roughly spherical groups, but it fails when clusters are **non-convex** or **nested**—for example concentric rings or interlocking moons—because points on opposite sides of a ring can be far apart in Euclidean distance yet belong to the same manifold.

Spectral clustering replaces the ambient metric with a **similarity graph**. Points that are locally similar become neighbors; the graph Laplacian then supplies an embedding in which those manifold-connected groups become more linearly separable, so a final Euclidean k-means step can succeed.

---

## Mathematical basis

### Gaussian similarity graph

Given points \(x_1,\dots,x_n \in \mathbb{R}^d\) and bandwidth \(\sigma > 0\),

\[
W_{ij} = \exp\!\left(-\frac{\|x_i - x_j\|^2}{2\sigma^2}\right)
\quad (i \neq j),\qquad W_{ii} = 0.
\]

Optional **mutual \(k\)-nearest-neighbor** sparsification keeps only strong local edges and symmetrizes the result. Smaller \(\sigma\) emphasizes local geometry; larger \(\sigma\) blends distant points and can wash out structure.

### Graph Laplacians

Let \(D = \mathrm{diag}(d_1,\dots,d_n)\) with \(d_i = \sum_j W_{ij}\). The combinatorial and symmetric normalized Laplacians are

\[
L = D - W,
\qquad
L_{\mathrm{sym}} = I - D^{-1/2} W D^{-1/2}.
\]

Both are symmetric positive semidefinite. For a connected graph, the smallest eigenvalue is \(0\) with a constant (or degree-weighted) eigenvector. A large gap after the first \(k\) eigenvalues is a classical signal that \(k\) well-separated graph communities exist.

### Spectral clustering (Ng–Jordan–Weiss style)

1. Build \(W\) (Gaussian affinities, then \(k\)-NN sparsify).  
2. Compute \(L_{\mathrm{sym}}\) and take the \(k\) eigenvectors for the smallest eigenvalues; stack them as rows of an embedding matrix \(U \in \mathbb{R}^{n \times k}\).  
3. Row-normalize \(U\) and run Lloyd k-means in that embedding.

Intuition: the embedding coordinates vary slowly across densely connected regions and jump across weak cuts, so non-convex ambient clusters can become compact in eigenvector space.

### Evaluation

Because labels are known for the synthetics, we report the **adjusted Rand index (ARI)** against ground truth (chance-corrected agreement). We also record ambient-space k-means **inertia** (within-cluster sum of squares) for the Euclidean baseline and for spectral partitions interpreted back in \(\mathbb{R}^d\).

---

## Implementation

| Piece | Location |
| --- | --- |
| Lloyd k-means + inertia | `src/applied_ml/kmeans.py` |
| Gaussian affinities, \(k\)-NN graph, Laplacians, embedding | `src/applied_ml/spectral.py` |
| Synthetic loaders (blobs / circles / moons) | `projects/spectral-clustering-study/datasets.py` |
| Comparisons, \(\sigma\) sweeps, seed stability | `projects/spectral-clustering-study/experiments.py` |
| Figures | `projects/spectral-clustering-study/plotting.py` |
| End-to-end runner | `projects/spectral-clustering-study/run_study.py` |
| Numerical tests | `tests/test_kmeans.py`, `tests/test_spectral.py` |

**Authorship note.** Graph construction, Laplacians, and Lloyd k-means are implemented in NumPy/SciPy for this portfolio. Dataset generators and ARI come from scikit-learn. No coursework notebooks are required to run the study.

**Library versions** are pinned in the repository root `requirements.txt`.

---

## Experiments

Fixed global seed `random_state=0` for dataset construction and primary runs; sensitivity analyses vary initialization seeds explicitly.

1. **Datasets**  
   - **blobs** — three well-separated isotropic Gaussians (Euclidean-friendly control).  
   - **circles** — concentric rings (nested, non-convex).  
   - **moons** — interlocking crescents (non-convex).

2. **Method comparison** — Euclidean Lloyd k-means (`n_init=10`) vs normalized spectral clustering (heuristic \(\sigma =\) median distance to the \(k\)-th neighbor, scaled by \(1/\sqrt{2}\); mutual \(k=12\) neighbors).

3. **Bandwidth sweep** — for each dataset, vary \(\sigma\) around the heuristic; record mean degree after \(k\)-NN, spectral gap \(\lambda_{k+1}-\lambda_k\), and mean/std ARI over five embedding k-means seeds.

4. **Initialization seeds** — ten Lloyd seeds for both methods at the heuristic \(\sigma\), summarized with box plots.

---

## Results

Figures live under [`assets/figures/spectral-clustering-study/`](../../assets/figures/spectral-clustering-study/).

| Figure | File |
| --- | --- |
| Side-by-side labels (truth / k-means / spectral) | `side_by_side_{blobs,circles,moons}.png` |
| Smallest Laplacian eigenvalues | `laplacian_eigenvalues.png` |
| ARI bar chart | `ari_comparison.png` |
| \(\sigma\) sensitivity (degree, gap, ARI) | `sigma_sensitivity_{blobs,circles,moons}.png` |
| Seed stability | `seed_stability_{blobs,circles,moons}.png` |

Numeric tables are written to `projects/spectral-clustering-study/results/summary.json` when you run the study.

**Primary-run ARI** (`random_state=0`, heuristic \(\sigma\), \(k=12\); regenerate with `run_study.py`):

| Dataset | k-means ARI | Spectral ARI |
| --- | ---: | ---: |
| blobs | 1.000 | 1.000 |
| circles | −0.002 | 1.000 |
| moons | 0.263 | 1.000 |

- On **blobs**, both methods recover the labels: Euclidean centroids already match the geometry.  
- On **circles** and **moons**, Euclidean k-means collapses while spectral clustering recovers the manifold labels when \(\sigma\) and \(k\) keep each ring/crescent internally connected without bridging the two classes.  
- The \(\sigma\)-sweep figures show how bandwidth changes mean degree and the Laplacian gap; too-small or too-large \(\sigma\) can destabilize recovery on some geometries (see `sigma_sensitivity_*.png`).

---

## Limitations

- **Synthetic only.** Real data bring noise, unequal densities, and irrelevant features that a fixed Gaussian kernel may not handle.  
- **Parameter dependence.** Performance hinges on \(\sigma\) and \(k\); the median-distance heuristic is convenient, not universal.  
- **Dense / medium-\(n\) numerics.** The study uses dense `eigh` on a few hundred points; large graphs need sparse eigensolvers.  
- **Final k-means still Euclidean** in the embedding—spectral clustering inherits sensitivity to empty clusters and unlucky seeds, which the seed-stability plots document.  
- **No claim of superiority on every geometry**—blobs remain a case where plain k-means is enough.

---

## Conclusion

Graph-based spectral clustering recovers nested and interlocking structures because the Laplacian embedding encodes **local connectivity**, not raw Euclidean distance to a centroid. When the similarity graph respects the manifold, the leading eigenvectors fold each connected component into a compact region where ordinary k-means finally works. The same experiments show that bandwidth and connectivity are first-class experimental factors: they change both the spectrum and the stability of the recovered partition.

---

## How to run

From the repository root:

```bash
python -m pip install -r requirements.txt
PYTHONPATH=src python projects/spectral-clustering-study/run_study.py
python -m pytest tests/test_kmeans.py tests/test_spectral.py -q
```

- **Reproducibility:** `RANDOM_STATE = 0` and `N_NEIGHBORS = 12` are set in `run_study.py`.  
- **Figures:** overwritten under `assets/figures/spectral-clustering-study/`.  
- **Metrics JSON:** `projects/spectral-clustering-study/results/summary.json`.
