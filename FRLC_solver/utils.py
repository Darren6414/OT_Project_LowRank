import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt


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


def draw_transport_plan(src, tgt, bary_src, bary_tgt, T_mat, title="", linewidth_scale=20):
    """Scatter plot with arrows between LC barycenters, weighted by T (from V2)."""
    plt.scatter(src[:, 0], src[:, 1], color="orange", alpha=0.2, label="Source", s=10)
    plt.scatter(tgt[:, 0], tgt[:, 1], color="green", alpha=0.2, label="Target", s=10)
    plt.scatter(bary_src[:, 0], bary_src[:, 1], color="royalblue", s=90, zorder=5,
                label="Source barycenters $Y^{(1)}$")
    plt.scatter(bary_tgt[:, 0], bary_tgt[:, 1], color="crimson", s=90, zorder=5,
                label="Target barycenters $Y^{(2)}$")
    for i in range(bary_src.shape[0]):
        for j in range(bary_tgt.shape[0]):
            mass = float(T_mat[i, j])
            if mass > 1e-4:
                plt.plot([bary_src[i, 0], bary_tgt[j, 0]], [bary_src[i, 1], bary_tgt[j, 1]],
                         c="steelblue", alpha=0.6, linewidth=mass * linewidth_scale)
    plt.title(title)
    plt.legend(loc="upper right", fontsize=8)
