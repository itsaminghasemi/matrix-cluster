# 🎯 matrix-cluster

Matrix-based clustering algorithms and linear algebra utilities implemented in NumPy, with SVD dimensionality reduction.

## 📌 Overview

This repository implements core clustering mechanics using vectorized linear algebra operations (`NumPy`) rather than high-level wrappers (`scikit-learn`), making matrix shapes, norms, inner products, and convergence explicit for linear algebra coursework and demonstrations.

## 📁 Project Structure

```text
matrix-cluster/
├── LICENSE
├── README.md
├── pyproject.toml
├── .gitignore
├── src/
│   └── matrix_cluster/
│       ├── __init__.py
│       ├── config.py
│       ├── distance.py
│       ├── kmeans.py
│       ├── reduction.py
│       └── plotting.py
├── scripts/
│   └── run_demo.py
└── tests/
    └── test_kmeans.py
```

## 🚀 Setup & Usage

Create and activate a virtual environment, then install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install numpy matplotlib pytest
```

Run the demo script:

```bash
python scripts/run_demo.py
```

Run unit tests:

```bash
pytest
```

## ⚖️ Assumptions, Trade-offs, and Limitations

* **Squared Euclidean expansion:** Computing pairwise squared distances via expanded norms and matrix multiplication (`X @ C.T`) is fast and vectorized ⚡, but can allocate `(N, K)` memory and suffer minor floating-point cancellation when norms are large and close together.
* **Empty clusters & local minima:** K-Means decreases or maintains inertia monotonically per iteration 📉, but converges to a local minimum and depends on initialization and `n_clusters`.
* **Geometry & dimensionality:** Spherical cluster geometry is implicitly assumed in standard K-Means 🔵; SVD 2D projection helps inspect high-dimensional data linearly, while non-convex shapes or varying densities may require other methods.
