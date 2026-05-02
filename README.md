# Low-Rank Optimal Transport: Factor Relaxation with Latent Coupling (FRLC)

This repository contains a high-performance JAX implementation of the **FRLC** algorithm, as introduced in the NeurIPS 2024 paper: *"Low-Rank Optimal Transport through Factor Relaxation with Latent Coupling"*. 

FRLC improves upon standard Low-Rank Optimal Transport (LOT) by relaxing constraints on inner marginals, allowing for a full latent coupling matrix that provides superior interpretability and handles asymmetric cluster distributions.

---

## ✨ Key Features

*   **Full Latent Coupling**: Unlike standard LOT which is restricted to a diagonal coupling, FRLC utilizes a full $r \times r$ matrix $\mathbf{T}$, enabling complex cluster-to-cluster mass splitting.
*   **JAX Optimized**: Fully compatible with `@jax.jit`, using `jax.lax.scan` for efficient loops and linear memory scaling $O((n+m)r)$.
*   **Numerical Stability**: Implements Sinkhorn projections in the Log-Sum-Exp (LSE) domain to prevent numerical instability at high ranks.
*   **OTT-JAX Integration**: Built to work seamlessly with the Optimal Transport Tools ecosystem.

---

## 📂 Repository Structure

The project is organized into a modular package and interactive tutorials:
```text
OT_PROJECT_LOWRANK/
├── FRLC_solver/              # Core Package
│   ├── __init__.py           # Exposes main functions (Flattened Namespace)
│   ├── solver.py             # FRLC Main loop (Mirror Descent)
│   ├── initialization.py     # LR factors initialization
│   ├── gradients_computation.py 
│   ├── sinkhorn_projection.py # LSE-domain projections
│   └── utils.py              # Projections and Visualization tools
├── FRLC_demo.ipynb           # Interactive Demo & Benchmark comparison
├── FRLC_tutorial.ipynb       # Step-by-step technical guide
├── pyproject.toml            # Project dependencies
└── README.md