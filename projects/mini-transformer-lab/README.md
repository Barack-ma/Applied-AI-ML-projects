# Mini-Transformer Lab

**Author:** Barack Marwanga  
**Portfolio project:** Applied AI & Machine Learning Projects  
**Question:** What happens inside a small NumPy transformer when we change attention masking, projections, and a sparse linear output head?

> **Disclaimer.** This lab studies a **small NumPy pedagogical model**. It is **not** a pretrained language model. Token **embeddings are random** (fixed Gaussian draws, never trained). Projection weights are fixed orthogonal matrices. No language modeling loss is optimized.

---

## Research question

Modern transformers combine linear projections, masked softmax attention, residual MLPs, and task-specific heads. In a stripped-down NumPy setting we can ask concrete linear-algebra questions:

1. How does a **causal mask** reshape the attention weight matrix relative to fully visible (unmasked) attention?
2. How much do block outputs move when we **replace projection matrices** or **perturb inputs**?
3. When a **sparse linear head** reads pooled transformer features, when do **Ridge** and **LASSO** beat plain **OLS** on a known sparse target?

---

## Mathematical basis

### Scaled dot-product attention

For query/key/value matrices \(Q,K \in \mathbb{R}^{n \times d}\) and \(V \in \mathbb{R}^{n \times d_v}\),

\[
\mathrm{Attention}(Q,K,V)
=
\mathrm{softmax}\!\left(\frac{QK^\top}{\sqrt{d}} + M\right) V,
\]

where a mask \(M\) sets disallowed positions to \(-\infty\) before the softmax. A **causal** (autoregressive) mask is the lower-triangular pattern \(M_{ij} = 0\) if \(i \ge j\) and \(-\infty\) otherwise, so position \(i\) may only attend to keys \(j \le i\).

### Orthogonal projections and a residual block

Given token matrix \(X \in \mathbb{R}^{n \times d_{\mathrm{model}}}\), one head uses

\[
Q = X W_Q,\quad K = X W_K,\quad V = X W_V
\]

with orthonormal-column factors \(W_Q,W_K,W_V\). The lab block is

\[
Y = X + \mathrm{Attn}(X) + \mathrm{FFN}\big(X + \mathrm{Attn}(X)\big),
\]

with a ReLU MLP for \(\mathrm{FFN}\). Because embeddings and weights are frozen random/orthogonal draws, changes in \(Y\) isolate the effect of masks and perturbations—not training dynamics.

### Sparse linear output head

Pooled block features form a design matrix \(H \in \mathbb{R}^{m \times p}\). We generate a **known sparse** coefficient vector \(\beta^\star\) with \(s \ll p\) nonzeros and observe

\[
y = H\beta^\star + \varepsilon.
\]

Estimators compared:

| Method | Objective |
| --- | --- |
| OLS | \(\min_x \|Hx - y\|_2^2\) |
| Ridge | \(\min_x \|Hx - y\|_2^2 + \lambda \|x\|_2^2\) |
| LASSO | \(\min_x \tfrac{1}{2m}\|Hx - y\|_2^2 + \lambda \|x\|_1\) |

LASSO uses coordinate descent with the **soft-threshold** proximal map \(\mathrm{S}_\tau(z) = \mathrm{sign}(z)\max(|z|-\tau,0)\). Ridge shrinks coefficients continuously; LASSO can set them exactly to zero.

**When regularization helps.** With more features than strongly identified directions—or noisy labels—OLS overfits and spreads mass across correlated coordinates. Ridge improves **robustness** by shrinking norms. LASSO improves **sparsity** and support recovery when \(\beta^\star\) is sparse, often lowering test error at moderate \(\lambda\). Too-large \(\lambda\) underfits both.

---

## Implementation

| Piece | Location |
| --- | --- |
| Causal mask, scores, softmax, attention | `src/applied_ml/attention.py` |
| Soft-threshold, OLS / Ridge / LASSO | `src/applied_ml/sparsity.py` |
| Orthogonal weights, FFN, transformer block | `src/applied_ml/projections.py` |
| Random embeddings | `src/applied_ml/embeddings.py` |
| Lab model wrapper | `projects/mini-transformer-lab/model.py` |
| Experiments | `projects/mini-transformer-lab/experiments.py` |
| Figures | `projects/mini-transformer-lab/plotting.py` |
| End-to-end runner | `projects/mini-transformer-lab/run_lab.py` |
| Module dependency notes | `src/applied_ml/README.md` |
| Course source copies (attribution) | `projects/mini-transformer-lab/attribution/` |
| Tests | `tests/test_attention.py`, `tests/test_sparsity.py` |

**Authorship.** Attention, soft-thresholding / LASSO CD, orthogonal projections, and the residual block are generalized from MATH 412 mini-transformer scaffolding (© Jason Aubrey / University of Arizona; see `attribution/`). Random embedding helpers follow the course `mt_utils` convention. Experiment design, OLS/Ridge comparison, figures, tests, and this write-up are portfolio work by Barack Marwanga.

**Library versions** are pinned in the repository root `requirements.txt`.

---

## Experiments

Global seed `RANDOM_STATE = 412` (course default) in `run_lab.py`.

1. **Causal vs unmasked attention** — same random token sequence; plot softmax weight heatmaps; report Frobenius gap and verify causal upper triangle is ~0.
2. **Projection / input sensitivity** — replace \((W_Q,W_K,W_V)\) with a fresh orthogonal triple, and separately add Gaussian noise to embeddings; measure \(\|Y'-Y\|_F\) and relative change.
3. **Sparse linear head** — mean-pool causal block features into \(H\); plant a 4-sparse \(\beta^\star\); fit OLS, Ridge, and LASSO over a \(\lambda\) grid; record train/test MSE and nonzero counts.

---

## Results

Figures live under [`assets/figures/mini-transformer-lab/`](../../assets/figures/mini-transformer-lab/).

| Figure | File |
| --- | --- |
| Unmasked vs causal attention heatmaps | `attention_causal_vs_unmasked.png` |
| Output change under W / X perturbations | `projection_sensitivity.png` |
| Ridge & LASSO coefficient paths | `coefficient_paths.png` |
| Train/test MSE vs \(\lambda\) | `prediction_errors.png` |
| Nonzero counts vs \(\lambda\) | `sparsity_vs_lambda.png` |

Numeric tables are written to `projects/mini-transformer-lab/results/summary.json` when you run the lab.

Primary-run metrics (`RANDOM_STATE = 412`; regenerate with `run_lab.py`):

| Quantity | Value |
| --- | ---: |
| Attention \(\|W_{\mathrm{full}} - W_{\mathrm{causal}}\|_F\) | 1.311 |
| Causal upper-triangle max weight | 0 |
| Relative output change (new projections) | 0.726 |
| Relative output change (noisy inputs) | 11.93 |
| OLS test MSE | 0.390 |
| Best Ridge test MSE (\(\lambda = 0.5\)) | 0.390 |
| Best LASSO test MSE (\(\lambda = 0.05\)) | 0.324 |
| LASSO nonzeros at \(\lambda = 0.1\) (true support = 4) | 4 |

- Causal masking forces a lower-triangular attention pattern; future-key weights are numerically zero.
- Swapping orthogonal projections or perturbing inputs both move the block output; input noise dominates in this configuration.
- LASSO recovers the planted support size and lowers test MSE vs OLS; Ridge shrinks without exact zeros and only slightly improves test error here.

---

## Limitations

- **Toy scale.** Sequence length, width, and depth are tiny; no multi-head stacking, LayerNorm, or positional encodings beyond raw order in the mask.
- **No learning.** Embeddings and weights are not trained; conclusions are about forward-pass linear algebra, not representation learning.
- **Synthetic sparse head.** The prediction target is planted; real language or vision heads have denser, misspecified structure.
- **Single seed primary run.** Sensitivity of LASSO paths to feature correlation and noise level is only partially explored via the \(\lambda\) grid.

---

## Conclusion

Inside this small NumPy transformer, **masking** is a hard linear-algebra constraint on the softmax support; **projection and input changes** propagate through the residual block as measurable output shifts; and a **sparse linear head** makes the usual bias–variance story concrete—Ridge for shrinkage robustness, LASSO for sparsity—when the true map from pooled features to targets is sparse. The lab’s value is mechanistic clarity, not language-model performance.

---

## How to run

From the repository root:

```bash
python3 -m pip install -r requirements.txt
PYTHONPATH=src python3 projects/mini-transformer-lab/run_lab.py
python3 -m pytest tests/test_attention.py tests/test_sparsity.py tests/test_kmeans.py tests/test_spectral.py -q
```

- **Reproducibility:** `RANDOM_STATE = 412` in `run_lab.py`; embeddings flagged as **random / not learned**.
- **Figures:** overwritten under `assets/figures/mini-transformer-lab/`.
- **Metrics JSON:** `projects/mini-transformer-lab/results/summary.json`.
