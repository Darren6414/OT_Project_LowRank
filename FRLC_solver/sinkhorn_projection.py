from ott.geometry.geometry import Geometry
from ott.problems.linear.linear_problem import LinearProblem
from ott.solvers.linear.sinkhorn import Sinkhorn


class OTSubproblemSolver:
    """OTT-JAX Sinkhorn wrapper for the three FRLC sub-problems.

    Exposes a unified interface for balanced OT (tau_a = tau_b = 1) and
    right semi-relaxed OT (tau_a = 1, tau_b < 1).
    Dual warm-starting is supported to accelerate the inner Sinkhorn loops.
    """

    def __init__(self, max_iters: int = 500, threshold: float = 1e-8):
        self._solver = Sinkhorn(threshold=threshold, max_iterations=max_iters)

    def project(self, cost_mat, marginal_a, marginal_b,
                warm_start, epsilon, tau_a, tau_b):
        geom = Geometry(cost_matrix=cost_mat, epsilon=epsilon)
        prob = LinearProblem(geom, a=marginal_a, b=marginal_b,
                             tau_a=tau_a, tau_b=tau_b)
        out = (self._solver(prob) if warm_start[0] is None
               else self._solver(prob, init=warm_start))
        return out.matrix, out.f, out.g
