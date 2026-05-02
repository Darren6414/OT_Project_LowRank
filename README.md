# Low-Rank Optimal Transport via FRLC (OTT-JAX Tutorial)

This repository contains a high-performance implementation of the **FRLC** (*Factor Relaxation with Latent Coupling*) algorithm, based on the paper by **Halmos et al. (NeurIPS 2024)**. 

The goal of this project is to provide a pedagogical and efficient tutorial integrated into the **OTT-JAX** ecosystem.

## 🚀 Key Implementation Features

Unlike standard low-rank OT (LOT) approaches that enforce identical inner marginals ($g_Q = g_R$), **FRLC** employs a richer latent coupling factorization:
$$P_r = Q \text{diag}(1/g_Q) T \text{diag}(1/g_R) R^\top$$

This implementation leverages **OTT-JAX** low-level primitives to maximize performance:
*   **Log-Sum-Exp (LSE) Stability**: Uses `jax.nn.logsumexp` for perfect numerical stability, even at high ranks.
*   **JAX Acceleration**: The algorithm is fully compiled via `@jax.jit` and optimized using `jax.lax.scan` for fast iterations.
*   **Memory Efficiency**: Utilizes `geom.apply_cost` to avoid the materialization of $N \times M$ cost matrices, ensuring linear scaling.

## 📂 Repository Structure

*   `FRLC_solver/`: Modular algorithmic core containing the solver logic.
*   `FRLC_tutorial.ipynb`: Demonstration notebook including theory, sanity checks, and performance benchmarks.
*   `pyproject.toml`: Dependency management (JAX, OTT-JAX, Matplotlib).
*   `.gitignore`: Prevents unnecessary cache and checkpoint files from polluting the repo.

## 📊 Results Preview

The algorithm demonstrates stable convergence and achieves a lower transport cost than standard `LRSinkhorn` on complex datasets, such as the **Two Moons** to **Eight Gaussians** transfer[cite: 2].

## 🛠 Installation & Usage

To test this tutorial locally, clone the repo and install the dependencies:
```bash
pip install .