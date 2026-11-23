# Applied AI & Machine Learning Projects

**Author:** Barack Marwanga  
**Course context:** MATH 412 — Linear Algebra for Data Science (University of Arizona)

A student portfolio that connects linear algebra coursework to applied machine-learning experiments. The repository is organized so cleaned notebooks, two deeper project write-ups, and shared numerical helpers can grow side by side.

## Purpose

- Publish cleaned MATH 412 Jupyter submissions with clear authorship notes (assignment scaffolding vs original work).
- Develop two independent applied-ML studies with a scientific write-up structure: research question → mathematical basis → implementation → experiments → results → limitations → conclusion.
- Keep reusable NumPy / SciPy helpers under `src/applied_ml/` with automated tests for numerical properties.

## Repository structure

```text
.
├── coursework/MATH412/          # Cleaned course notebooks
│   └── notebooks/
├── projects/
│   ├── mini-transformer-lab/    # Project 1 — small NumPy transformer study
│   └── spectral-clustering-study/  # Project 2 — graph clustering study
├── src/applied_ml/              # Shared helpers (attention, sparsity, k-means, spectral)
├── tests/                       # Pytest suite
├── assets/figures/              # Saved experiment figures
├── requirements.txt
└── README.md
```

## Status

| Area | Status |
|------|--------|
| Repo layout, root docs, pinned deps, smoke test | **Ready** |
| [Coursework — MATH 412](coursework/MATH412/) | **Ready** — cleaned weekly notebooks in `notebooks/` |
| [Project 1 — Mini-Transformer Lab](projects/mini-transformer-lab/) | **Ready** — lab code, figures, attention/sparsity tests |
| [Project 2 — Spectral Clustering Study](projects/spectral-clustering-study/) | **Ready** — study code, figures, and tests merged |
| Shared `src/applied_ml` helpers | **Ready** — attention, projections, sparsity, embeddings, k-means, spectral |

MATH 412 notebooks are published under [`coursework/MATH412/`](coursework/MATH412/). Both portfolio projects publish code, scientific READMEs, and figures under [`projects/`](projects/) and [`assets/figures/`](assets/figures/).

## Projects

### [Mini-Transformer Lab](projects/mini-transformer-lab/) — ready

**Research question:** What happens inside a small NumPy transformer when we change attention masking, projections, and the sparse linear output head?

A pedagogical, from-scratch NumPy transformer (**not** a pretrained LM; **random** embeddings). Experiments: causal vs unmasked attention heatmaps, projection/input sensitivity, and OLS / Ridge / LASSO sparse heads on a planted synthetic target.

### [Spectral Clustering Study](projects/spectral-clustering-study/) — ready

**Research question:** Why do graph-based representations recover cluster structures that ordinary Euclidean k-means struggles to separate?

Compares k-means vs spectral clustering on blobs, concentric circles, and interlocking moons, with bandwidth / connectivity sweeps, seed stability checks, and Laplacian eigenvalue diagnostics. See the project README for the full write-up, how to regenerate figures, and limitations.

### [MATH 412 coursework](coursework/MATH412/) — ready

Cleaned weekly notebooks live in [`coursework/MATH412/notebooks/`](coursework/MATH412/notebooks/). See that folder’s README for the week index, HW03 resolution, scaffolding vs original work, and how to open the notebooks.

## Install and run

Requires Python 3.10+.

```bash
# Clone
git clone https://github.com/Barack-ma/Applied-AI-ML-projects.git
cd Applied-AI-ML-projects

# Optional: create a virtual environment
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# Install pinned scientific stack
pip install -r requirements.txt

# Run tests (src/ is on the path via pytest.ini / tests/conftest.py)
pytest
```

Regenerate project figures:

```bash
PYTHONPATH=src python3 projects/mini-transformer-lab/run_lab.py
PYTHONPATH=src python3 projects/spectral-clustering-study/run_study.py
```

Open the MATH 412 notebooks:

```bash
cd coursework/MATH412/notebooks
jupyter lab   # or: jupyter notebook
```

Put `src/` on `PYTHONPATH` (or install an editable package later) when importing `applied_ml` from notebooks or scripts outside pytest:

```bash
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"
```

## Authorship note

Parts of this repository originate from MATH 412 assignment scaffolding (starter notebooks, prompts, and course utilities). Other parts are original student implementations, experiments, and write-ups by Barack Marwanga.

- Course notebooks will state which cells are scaffolding vs original work.
- Project READMEs separate reused course ideas from independent analysis.
- The mini-transformer lab is a **small NumPy model for study**, not a pretrained LM; embeddings are **random / not learned**.
- Transformer helpers in `src/applied_ml/` generalize MATH 412 scaffolding with portfolio tests and experiments; spectral helpers are NumPy/SciPy implementations written for this portfolio; datasets and ARI use scikit-learn.

## Reproducibility

- Dependencies are pinned in [`requirements.txt`](requirements.txt).
- Experiments document random seeds and labeled plots where applicable.
- Generated figures belong under [`assets/figures/`](assets/figures/).

## License / contact

Portfolio work by [Barack Marwanga](https://github.com/Barack-ma). Open an issue on this repository with questions about reproducing a published experiment.
