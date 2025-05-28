"""q-space sampling, for finding optimal samplings on the unit sphere.

Code is taken from https://github.com/ecaruyer/qspace/tree/master

Implementation was first described in [1].

References
---
[1] Caruyer, Emmanuel, Christophe Lenglet, Guillermo Sapiro, and Rachid Deriche. "Design of multishell sampling schemes with uniform coverage in diffusion MRI." Magnetic Resonance in Medicine 69, no. 6 (2013): 1534-1540.
"""

from typing import List

import numpy
from scipy.optimize import fmin_slsqp


__all__ = ["optimize_multishell", "optimize_singleshell"]


_epsilon = 1e-9


def _equality_constraints(vects: numpy.ndarray, *args):
    """_equality_constraints computer the spherical equality constraint.

    Returns 0 if vects lies on the unit sphere.

    Args:
        vects (numpy.ndarray) : array of vectors, shape (Nx3,)

    Returns:
        numpy.ndarray: difference between squared norm and 1.
    """
    n = vects.shape[0] // 3
    vects = vects.reshape((n, 3))
    return (vects**2).sum(1) - 1.0


def _electrostatic_repulsion(
    vects: numpy.ndarray,
    weight_matrix: numpy.ndarray,
    antipodal: bool = True,
    alpha: float = 1.0,
) -> float:
    """_electrostatic_repulsion electrostatic-repulsion objective function. The alpha paramter controls the power repulsion (energy varies as $1 / r^alpha$).

    Args:
        vects (numpy.ndarray): array of vectors, (Nx3,)
        weight_matrix (numpy.ndarray): weight of each pair of points, (N,N)
        antipodal (bool, optional):
        alpha (float, optional): controls the power of the repulsion. Defaults to 1.0.

    Returns:
        float: sum of all interactions between any 2 vectors.
    """
    epsilon = 1e-9
    n = vects.shape[0] // 3
    vects = vects.reshape((n, 3))
    energy = 0.0
    for i in range(n):
        indices = numpy.arange(n) > i
        diffs = ((vects[indices] - vects[i]) ** 2).sum(1) ** alpha
        energy += (weight_matrix[i, indices] * (1.0 / (diffs + epsilon))).sum()
        if antipodal:
            sums = ((vects[indices] + vects[i]) ** 2).sum(1) ** alpha
            energy += (
                weight_matrix[i, indices] * (1.0 / (sums + epsilon))
            ).sum()
    return energy


def _grad_electrostatic_repulsion(
    vects, weight_matrix, antipodal=True, alpha=1.0
) -> numpy.ndarray:
    """_grad_electrostatic_repulsion 1st-order derivative of electrostatic-like repulsion energy.

    Args:
        vects (numpy.ndarray): array of input vectors, (Nx3,)
        weight_matrix (numpy.ndarray): contribution of each pair of points, (N,N).
        alpha (float, optional): controls power of the repulsion. Defaults to 1.0

    Returns:
        numpy.ndarray: gradient of electrostatic repulsion objective function.
    """
    n = vects.shape[0] // 3
    vects = vects.reshape((n, 3))
    grad = numpy.zeros((n, 3))
    for i in range(n):
        indices = numpy.arange(n) != i
        diffs = ((vects[indices] - vects[i]) ** 2).sum(1) ** (alpha + 1)
        grad[i] = (
            -2
            * alpha
            * weight_matrix[i, indices]
            * (vects[i] - vects[indices]).T
            / diffs
        ).sum(1)
        if antipodal:
            sums = ((vects[indices] + vects[i]) ** 2).sum(1) ** (alpha + 1)
            grad[i] += (
                -2
                * alpha
                * weight_matrix[i, indices]
                * (vects[i] + vects[indices]).T
                / sums
            ).sum(1)
    grad = grad.reshape(n * 3)
    return grad


def _cost_function(
    vects: numpy.ndarray,
    s: int,
    ks: List[int],
    weights: numpy.ndarray,
    antipodal: bool = True,
) -> float:
    """_cost_function objective function for multiple-shell energy.

    Args:
        vects (numpy.ndarray):
        s (int): # of shells
        ks (List[int]): number of points per shell
        weights (numpy.ndarray): weighting parameter, control coupling between shells and how this balances, shape (s,s).

    Returns:
        float: cost
    """
    k = numpy.sum(ks)
    indices = numpy.cumsum(ks).tolist()
    indices.insert(0, 0)
    weight_matrix = numpy.zeros((k, k))
    for s1 in range(s):
        for s2 in range(s):
            weight_matrix[
                indices[s1] : indices[s1 + 1], indices[s2] : indices[s2 + 1]
            ] = weights[s1, s2]
    return _electrostatic_repulsion(vects, weight_matrix, antipodal)


def _grad_cost(
    vects: numpy.ndarray,
    s: int,
    ks: List[int],
    weights: numpy.ndarray,
    antipodal: bool = True,
):
    """_grad_cost gradient of the objective function for multiple shells sampling.

    Args:
        vects (numpy.ndarray) : array-like shape (N * 3,)
        s (int) : # of shells
        ks (List[int] : # of points/shell.
        weights (numpy.ndarray): weighting parameter, control coupling between shells and how this balances.
        antipodal (bool):

    Returns:
        numpy.ndarray: gradient of cost function
    """
    k = vects.shape[0] // 3
    indices = numpy.cumsum(ks).tolist()
    indices.insert(0, 0)
    weight_matrix = numpy.zeros((k, k))
    for s1 in range(s):
        for s2 in range(s):
            weight_matrix[
                indices[s1] : indices[s1 + 1], indices[s2] : indices[s2 + 1]
            ] = weights[s1, s2]
    return _grad_electrostatic_repulsion(vects, weight_matrix, antipodal)


def optimize_multishell(
    num_shells: int,
    num_pts_per_shell: List[int],
    weights: numpy.ndarray,
    max_iter: int = 100,
    antipodal: bool = True,
    init_points: numpy.ndarray | None = None,
) -> numpy.ndarray:
    """Do the optimization to find optimal sampling for a multishell configuration in three dimensions.

    Args:
        num_shells (int): # of shells
        num_pts_per_shell (List[int]): # of points in each shell
        weights (numpy.ndarray): _description_
        max_iter (int, optional): maximum # of iterations to run for. Defaults to 100.
        antipodal (bool, optional): _description_. Defaults to True.
        init_points (numpy.ndarray | None, optional): initial points for optimization. Defaults to None.

    Returns:
        numpy.ndarray: optimized sampling points (N,3)
    """
    num_shells = len(num_pts_per_shell)
    k = numpy.sum(num_pts_per_shell)  # total # points

    if init_points is None:  # initialize with random directions
        init_points = random_uniform_on_sphere(k)
    vects = init_points.reshape(k * 3)

    vects = fmin_slsqp(
        _cost_function,
        vects.reshape(k * 3),
        f_eqcons=_equality_constraints,
        fprime=_grad_cost,
        iter=max_iter,
        acc=_epsilon,
        args=(num_shells, num_pts_per_shell, weights, antipodal),
        iprint=0,
    )
    vects = vects.reshape((k, 3)) # type: ignore
    vects = (vects.T / numpy.sqrt((vects**2).sum(1))).T
    return vects


def optimize_singleshell(
    num_pts: int,
    max_iter: int = 100,
    antipodal: bool = True,
    init_points: numpy.ndarray | None = None,
    return_angles: bool = False,
) -> numpy.ndarray:
    """Find optimal sampling for a single three-dimensional shell.

    Args:
        num_pts (int): number of points.
        max_iter (int, optional): maximum # of iterations to optimize for. Defaults to 100.
        antipodal (bool, optional):. Defaults to True.
        init_points (numpy.ndarray | None, optional): initial points for optimization. Defaults to None.
        return_angles (bool, optional): return points as pair of angles (theta, phi). Defaults to False, which will return points in xyz.

    Returns:
        numpy.ndarray: points
    """
    shell_groups = [[0], range(0, 1)]
    alphas = numpy.ones(len(shell_groups))
    weights = compute_weights(1, [num_pts], shell_groups, alphas)
    points = optimize_multishell(
        1, [num_pts], weights, max_iter, antipodal, init_points
    )
    if return_angles:
        theta = numpy.arctan2(points[:, 1], points[:, 0])
        phi = numpy.arccos(points[:, 2])
        return numpy.vstack([theta, phi]).T
    else:
        return points


def random_uniform_on_sphere(k: int) -> numpy.ndarray:
    """random_uniform_on_sphere generate a set of k random unit vectors, following a uniform  distribution on the sphere.

    Args:
        k (int): number of vectors to generate
    """
    phi = 2 * numpy.pi * numpy.random.rand(k)
    r = 2 * numpy.sqrt(numpy.random.rand(k))
    theta = 2 * numpy.arcsin(r / 2)
    vects = numpy.zeros((k, 3))
    vects[:, 0] = numpy.sin(theta) * numpy.cos(phi)
    vects[:, 1] = numpy.sin(theta) * numpy.sin(phi)
    vects[:, 2] = numpy.cos(theta)
    return vects


def compute_weights(num_shells, num_points_per_shell, shell_groups, alphas):
    """compute_weights Compute the weights array from a set of shell groups to couple, and coupling weights.

    Args:
        num_shells (int): # of shells
        num_points_per_shell (int): # points/shell
        shell_groups (List[int]):
        alphas (List[float]):
    """
    weights = numpy.zeros((num_shells, num_shells))
    for shell_group, alpha in zip(shell_groups, alphas):
        total_nb_points = 0
        for shell_id in shell_group:
            total_nb_points += num_points_per_shell[shell_id]
        for i in shell_group:
            for j in shell_group:
                weights[i, j] += alpha / total_nb_points**2
    return weights
