import jax
import jax.numpy as jnp


@jax.jit
def factor_gradients_and_stepsize(C, Q, R, X, gamma):
    """Rank-1-corrected factor gradients and normalised step size (Algorithm 1)."""
    # grad_Q = C R X^T - 1_n w1^T
    raw_gradQ = (C @ R) @ X.T
    gQ = jnp.sum(Q, axis=0)
    w1 = jnp.sum(raw_gradQ * Q, axis=0) / gQ
    grad_Q = raw_gradQ - w1[None, :]

    # grad_R = C^T Q X - 1_m w2^T
    raw_gradR = (C.T @ Q) @ X
    gR = jnp.sum(R, axis=0)
    w2 = jnp.sum(R * raw_gradR, axis=0) / gR
    grad_R = raw_gradR - w2[None, :]
    gamma_k = gamma / (jnp.maximum(jnp.max(jnp.abs(grad_Q)), jnp.max(jnp.abs(grad_R))))
    return grad_Q, grad_R, gamma_k


@jax.jit
def latent_gradient_and_stepsize(C, Q, R, X, gQ, gR, gamma):
    """Latent coupling gradient and normalised step size for T (Algorithm 1)."""
    gamma_T = gamma / jnp.max(jnp.abs((Q.T @ C @ R) / (gQ[:, None] * gR[None, :])))
    return (Q.T @ C @ R) / (gQ[:, None] * gR[None, :]), gamma_T
