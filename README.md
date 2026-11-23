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
├── coursework/MATH412/          # Cleaned course notebooks (in progress)
│   └── notebooks/
├── projects/
│   ├── mini-transformer-lab/    # Project 1 — small NumPy transformer study (pending)
│   └── spectral-clustering-study/  # Project 2 — graph clustering study (landed)
├── src/applied_ml/              # Shared helpers (k-means, spectral utils; more coming)
├── tests/                       # Pytest suite
├── assets/figures/              # Saved experiment figures
├── requirements.txt
└── README.md
```

## Status

| Area | Status |
|------|--------|
| Repo layout, root docs, pinned deps, smoke test | **Ready** |
| [Coursework — MATH 412](coursework/MATH412/) | **In progress** — cleaned notebooks will land in `notebooks/` |
| [Project 1 — Mini-Transformer Lab](projects/mini-transformer-lab/) | **In progress** — experiments and write-up not published yet |
| [Project 2 — Spectral Clustering Study](projects/spectral-clustering-study/) | **Ready** — study code, figures, and tests merged |
| Shared `src/applied_ml` helpers | **Partial** — k-means and spectral utilities landed; transformer helpers pending |

Coursework and the mini-transformer lab are still pending. Spectral Clustering Study results and figures are published under [`projects/spectral-clustering-study/`](projects/spectral-clustering-study/) and [`assets/figures/spectral-clustering-study/`](assets/figures/spectral-clustering-study/).

## Projects

### [Mini-Transformer Lab](projects/mini-transformer-lab/) — in progress

**Research question:** What happens inside a small NumPy transformer when we change attention masking, projections, and the sparse linear output head?

A pedagogical, from-scratch transformer (not a pretrained language model). Planned experiments include causal vs unmasked attention visualizations, projection sensitivity, and OLS / Ridge / LASSO sparse heads on synthetic targets.

### [Spectral Clustering Study](projects/spectral-clustering-study/) — ready

**Research question:** Why do graph-based representations recover cluster structures that ordinary Euclidean k-means struggles to separate?

Compares k-means vs spectral clustering on blobs, concentric circles, and interlocking moons, with bandwidth / connectivity sweeps, seed stability checks, and Laplacian eigenvalue diagnostics. See the project README for the full write-up, how to regenerate figures, and limitations.

### [MATH 412 coursework](coursework/MATH412/) — in progress

Cleaned weekly notebooks will live in [`coursework/MATH412/notebooks/`](coursework/MATH412/notebooks/). See that folder’s README for how submissions will be organized.

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

Regenerate the spectral clustering study figures:

```bash
python projects/spectral-clustering-study/run_study.py
```

When coursework notebooks are available:

```bash
jupyter notebook
# or: jupyter lab
```

Put `src/` on `PYTHONPATH` (or install an editable package later) when importing `applied_ml` from notebooks or scripts outside pytest:

```bash
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"
```

## Authorship note

Parts of this repository originate from MATH 412 assignment scaffolding (starter notebooks, prompts, and course utilities). Other parts are original student implementations, experiments, and write-ups by Barack Marwanga.

- Course notebooks will state which cells are scaffolding vs original work.
- Project READMEs separate reused course ideas from independent analysis.
- The mini-transformer work (when published) is a **small NumPy model for study**, not a pretrained LM.
- Spectral clustering helpers in `src/applied_ml/` are NumPy/SciPy implementations written for this portfolio; datasets and ARI use scikit-learn.

## Reproducibility

- Dependencies are pinned in [`requirements.txt`](requirements.txt).
- Experiments document random seeds and labeled plots where applicable.
- Generated figures belong under [`assets/figures/`](assets/figures/).

## License / contact

Portfolio work by [Barack Marwanga](https://github.com/Barack-ma). Open an issue on this repository with questions about reproducing a published experiment.
