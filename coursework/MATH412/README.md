# MATH 412 — Linear Algebra for Data Science

Cleaned Jupyter coursework by **Barack Marwanga**, published for portfolio review.

Notebooks live in [`notebooks/`](notebooks/). A raw source archive is kept at [`source/Jupyter_Files.zip`](source/Jupyter_Files.zip).

## How to open and run

From the repository root (Python 3.10+):

```bash
pip install -r requirements.txt
cd coursework/MATH412/notebooks
jupyter lab          # or: jupyter notebook
# Kernel → Restart & Run All in any notebook
```

`mt_utils.py` and `mini_transformer.py` sit beside the notebooks. Several weekly assignments import them; keep that folder as the working directory when executing cells.

## Week index

| Week / file | Topic | Notes |
|-------------|-------|-------|
| [`CodingHW_00_Environment_Check_Local.ipynb`](notebooks/CodingHW_00_Environment_Check_Local.ipynb) | Environment check | Portfolio seed replaces a personal course ID |
| [`CodingHW_01_Matrices_Systems_RowReduction.ipynb`](notebooks/CodingHW_01_Matrices_Systems_RowReduction.ipynb) | Matrices, systems, RREF; first attention scores | |
| [`CodingHW_02_LU_Inverses_Det.ipynb`](notebooks/CodingHW_02_LU_Inverses_Det.ipynb) | LU, inverses, determinants | |
| [`CodingHW_03_VectorSpaces_InnerProducts.ipynb`](notebooks/CodingHW_03_VectorSpaces_InnerProducts.ipynb) | Vector spaces, inner products, attention | **Completed submission** (see HW03 note) |
| [`CodingHW_04_GramSchmidt_Subspaces.ipynb`](notebooks/CodingHW_04_GramSchmidt_Subspaces.ipynb) | QR / Gram–Schmidt, fundamental subspaces | |
| [`CodingHW_05_Eigenvalues.ipynb`](notebooks/CodingHW_05_Eigenvalues.ipynb) | Eigenvalues, spectral theorem | Uses cumulative mini-transformer helpers |
| [`CodingHW_06_SpectralProjection.ipynb`](notebooks/CodingHW_06_SpectralProjection.ipynb) | Projection, positive-definite Gram matrices | |
| [`CodingHW_07_SVD_VectorNorms.ipynb`](notebooks/CodingHW_07_SVD_VectorNorms.ipynb) | SVD, vector norms, low-rank weights | |
| [`CodingHW_08_Matrix_Norms_Conditioning.ipynb`](notebooks/CodingHW_08_Matrix_Norms_Conditioning.ipynb) | Matrix norms and conditioning | |
| [`CodingHW_09_Gradient_Hessian_and_PCA.ipynb`](notebooks/CodingHW_09_Gradient_Hessian_and_PCA.ipynb) | Gradient, Hessian, PCA | |
| [`CodingHW_10_Least_Squares_and_LASSO.ipynb`](notebooks/CodingHW_10_Least_Squares_and_LASSO.ipynb) | Least squares, ridge, LASSO | Optional project-sync cell softens later helper snapshot |
| [`CodingHW_11_Clustering_Token_Representations.ipynb`](notebooks/CodingHW_11_Clustering_Token_Representations.ipynb) | K-means / spectral clustering on tokens | |

## HW03 duplicate resolution

The source zip contained two HW03 notebooks:

- `CodingHW_03_VectorSpaces_InnerProducts.ipynb` — incomplete draft (placeholder returns for independence / projection / cosine similarity; unanswered markdown prompt)
- `CodingHW_03_VectorSpaces_InnerProducts_sub.ipynb` — **completed submission** (implementations filled in, written answer present, prior execution outputs)

**Kept:** the `_sub` notebook, renamed to `CodingHW_03_VectorSpaces_InnerProducts.ipynb` as the single current submission. The incomplete draft is not published under `notebooks/` (it remains only inside `source/Jupyter_Files.zip`).

## Scaffolding vs original student work

| Kind | What it looks like |
|------|--------------------|
| **Assignment scaffolding** | Problem statements, math prompts, check/assert cells, section titles such as “Your task”, and starter signatures |
| **Original student work** | Function bodies, numerical experiments, written short answers, and AI-use notes where present |
| **Portfolio cleanup** | Removed course template branding (institution headers, copyright footers, portal-only submit copy), cleared leftover `# TODO` comments on finished code, and used a fixed portfolio seed in HW00 |

Deeper independent write-ups that grow from this coursework live under [`projects/`](../../projects/) (not rewritten in this PR).

## Cleanup and re-execution notes

- All twelve published notebooks were executed successfully with the pinned root [`requirements.txt`](../../requirements.txt) in this environment.
- HW00 uses portfolio seed `41202026` instead of a personal student ID.
- HW10’s optional `mini_transformer.fit_sparse_head` sync is skipped when the helper file is a later-week cumulative snapshot without that attribute; notebook Tasks 1–4 still run and assert.
- PNG figures written beside the notebooks (`hw8_*.png`, `hw10_*.png`, `hw11_*.png`) come from those successful runs and regenerate on Restart & Run All.
