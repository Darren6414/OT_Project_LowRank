import jax.numpy as jnp
from .initialization import build_initial_factors
from .gradients_computation import factor_gradients_and_stepsize, latent_gradient_and_stepsize
from .sinkhorn_projection import OTSubproblemSolver
from .utils import iterate_change


def run_frlc(
    C,
    a=None, b=None,
    tau=50.0,
    gamma=70.0,
    rank=10,
    rank2=None,
    max_outer=200,
    min_outer=25,
    tol=1e-5,
    max_inner=500,
    return_costs=True,
):
    """Balanced FRLC via block coordinate mirror descent (Halmos et al., 2024).

    Solves  min_{P in Pi_r(a,b)} <C, P>_F  using the LC factorization
    P = Q diag(1/gQ) T diag(1/gR) R^T.

    Each outer iteration calls three OTT-JAX Sinkhorn sub-problems:
      (1) right semi-relaxed OT for Q  — tau_a=1, tau_b=tau/(tau+eps_k)
      (2) right semi-relaxed OT for R  — same tau_b
      (3) balanced OT for T            — tau_a=tau_b=1

    Parameters
    ----------
    C          : cost matrix (n, m)
    a, b       : marginals — uniform if None
    tau, gamma : relaxation and step-size scale
    rank       : latent rank r
    max_outer  : maximum outer iterations
    min_outer  : minimum outer iterations before convergence check
    tol        : convergence tolerance on Delta_k
    max_inner  : Sinkhorn iterations for each sub-problem
    return_costs : if True, return list of primal costs per iteration

    Returns
    -------
    Q, R, T  : LC factors
    costs    : list of primal transport costs (if return_costs=True)
    """
    N1, N2 = C.shape
    ones_N1 = jnp.ones(N1, dtype=C.dtype)
    ones_N2 = jnp.ones(N2, dtype=C.dtype)

    if a is None:
        a = ones_N1 / N1
    if b is None:
        b = ones_N2 / N2
    if rank2 is None:
        rank2 = rank

    inner_marg_Q = jnp.ones(rank,  dtype=C.dtype) / rank
    inner_marg_R = jnp.ones(rank2, dtype=C.dtype) / rank2

    Q, R, T, X = build_initial_factors(a, b, inner_marg_Q, inner_marg_R, epsilon=gamma)

    sub_Q = OTSubproblemSolver(max_iters=max_inner)
    sub_R = OTSubproblemSolver(max_iters=max_inner)
    sub_T = OTSubproblemSolver(max_iters=max_inner)

    Q_prev = R_prev = T_prev = None
    f1Q = f2Q = f1R = f2R = f1T = f2T = None
    gamma_k = gamma
    costs = []
    k = 0

    while k < max_outer and (
        k < min_outer
        or iterate_change((Q, R, T), (Q_prev, R_prev, T_prev), gamma_k) > tol
    ):
        Q_prev, R_prev, T_prev = Q, R, T

        # Step 1: factor relaxation (Q and R independently)
        grad_Q, grad_R, gamma_k = factor_gradients_and_stepsize(C, Q, R, X, gamma)
        eps = float(gamma_k ** -1)
        tau_b = tau / (tau + eps)

        cost_Q = grad_Q - eps * jnp.log(Q)
        cost_R = grad_R - eps * jnp.log(R)

        Q, f1Q, f2Q = sub_Q.project(cost_Q, a,  Q.T @ ones_N1, (f1Q, f2Q), eps, 1.0, tau_b)
        R, f1R, f2R = sub_R.project(cost_R, b,  R.T @ ones_N2, (f1R, f2R), eps, 1.0, tau_b)

        gQ = Q.T @ ones_N1
        gR = R.T @ ones_N2

        # Step 2: latent coupling update (T)
        grad_T, gamma_T = latent_gradient_and_stepsize(C, Q, R, X, gQ, gR, gamma)
        eps_T = float(gamma_T ** -1)
        cost_T = grad_T - eps_T * jnp.log(T)

        T, f1T, f2T = sub_T.project(cost_T, gQ, gR, (f1T, f2T), eps_T, 1.0, 1.0)

        X = jnp.diag(1.0 / gQ) @ T @ jnp.diag(1.0 / gR)

        # Primal cost via trace identity
        cost = float(jnp.trace(((Q.T @ C) @ R) @ X.T))
        costs.append(cost)
        k += 1

    return (Q, R, T, costs) if return_costs else (Q, R, T)
