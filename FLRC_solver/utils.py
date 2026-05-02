import jax
import jax.numpy as jnp


def compute_lc_projections(src_pts, tgt_pts, Q, R):
    """Barycentric cluster representatives from LC sub-couplings.

    Follows Definition 4.1 of Halmos et al. 2024.

    Parameters
    ----------
    src_pts, tgt_pts : point clouds, shape (n, d) and (m, d)
    Q                : left sub-coupling,  shape (n, r)
    R                : right sub-coupling, shape (m, r)

    Returns
    -------
    Y_src : source barycenters, shape (r, d)
    Y_tgt : target barycenters, shape (r, d)
    """
    gQ = jnp.sum(Q, axis=0)
    gR = jnp.sum(R, axis=0)
    return (Q.T @ src_pts) / gQ[:, None], (R.T @ tgt_pts) / gR[:, None]


@jax.jit
def iterate_change(current, previous, gamma_k):
    """Normalised squared iterate change — stopping criterion (Eq. 10)."""
    Q, R, T = current
    Q_prev, R_prev, T_prev = previous
    return (gamma_k ** -2) * (
        jnp.linalg.norm(Q - Q_prev) ** 2
        + jnp.linalg.norm(R - R_prev) ** 2
        + jnp.linalg.norm(T - T_prev) ** 2
    )
