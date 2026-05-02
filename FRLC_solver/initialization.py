import jax
import jax.numpy as jnp
from ott.geometry.geometry import Geometry
from ott.problems.linear.linear_problem import LinearProblem
from ott.solvers.linear.sinkhorn import Sinkhorn


def build_initial_factors(marginal_a, marginal_b, inner_a, inner_b,
                          epsilon=1.0, threshold=1e-8, max_iter=500):
    """Initialise (Q, R, T, X) following Algorithm 6 of Halmos et al. 2024.

    Random uniform matrices are used as cost matrices for initial Sinkhorn
    projections, giving a full-rank starting point in Pi_r(a, b).

    Parameters
    ----------
    marginal_a, marginal_b : source and target marginals
    inner_a, inner_b       : initial inner marginals (uniform 1/r)
    epsilon                : Sinkhorn regularisation for initialisation
    """
    dtype = marginal_a.dtype
    n, m  = marginal_a.shape[0], marginal_b.shape[0]
    rQ, rR = inner_a.shape[0], inner_b.shape[0]

    key = jax.random.PRNGKey(0)
    k1, k2, k3 = jax.random.split(key, 3)

    cost_init_Q = jax.random.uniform(k1, (n,  rQ), dtype=dtype)
    cost_init_R = jax.random.uniform(k2, (m,  rR), dtype=dtype)
    cost_init_T = jax.random.uniform(k3, (rQ, rR), dtype=dtype)

    solver = Sinkhorn(threshold=threshold, max_iterations=max_iter)

    def _balanced_proj(cost_rand, marg_left, marg_right):
        geom = Geometry(cost_matrix=-float(epsilon)*cost_rand, epsilon=float(epsilon))
        return solver(LinearProblem(geom, a=marg_left, b=marg_right)).matrix

    Q = _balanced_proj(cost_init_Q, marginal_a, inner_a)
    R = _balanced_proj(cost_init_R, marginal_b, inner_b)

    gQ_actual = Q.T @ jnp.ones(n, dtype=dtype)
    gR_actual = R.T @ jnp.ones(m, dtype=dtype)

    T = _balanced_proj(cost_init_T, gQ_actual, gR_actual)
    X = jnp.diag(1.0 / gQ_actual) @ T @ jnp.diag(1.0 / gR_actual)

    return Q, R, T, X