"""Left-invariant derivatives."""

from functools import partial

import jax
import jax.numpy as jnp
from jax.tree_util import Partial
from jaxtyping import Array
from jaxtyping import Float

from .._gauss import gaussian_filter


@partial(jax.jit, static_argnums=(2, 3))
def left_invariant_derivative_frame2(
    ori_score: Float[Array, "t y x"],
    thetas: Float[Array, " t"],
    sigma_o: Float,
    sigma_s: Float,
) -> Float[Array, "t y x 3"]:
    """Compute the pixel-wise left-invariant derivative for the input orientation scores. Creates the left-invariant rotating derivative (LID) frame of reference.

    Args:
        ori_score (Float[Array, "t y x"]): array of orientation scores, where the angle is the first dimension and the spatial dimensions are the last two.
        thetas (Float[Array, "a"]): array of angles that was used to compute the orientation scores, corresponds to the first dimension of `ori_score`.
        sigma_o (float): standard deviation of Gaussian used to blur orientation axis.
        sigma_s (float): standard deviation of Gaussian used to blur spatial axes.

    Returns:
        Float[Array, "t y x 3"]: A 4D matrix where the first 3 dimensions match those of the input ori_score, and the last is size-3 and has the (d_zeta, d_eta, dt) components of the derivative.

    Notes:
        See Eqn. 1 of [1] for details and how this is constructed.
        If `sigma_s` and `sigma_o` are zero, `jnp.gradient` will be used to calculate derivatives. Otherwise, uses an order-1 Gaussian filter.

    References:
        [1] Zhang, Jiong, et al. "Robust retinal vessel segmentation via locally adaptive derivative frames in orientation scores." IEEE transactions on medical imaging 35.12 (2016): 2631-2644.
    """
    if sigma_o == 0 and sigma_s == 0:
        dt, dy, dx = jnp.gradient(ori_score)
    else:
        dt = gaussian_filter(ori_score, sigma_o, order=1, axis=0)
        dy = gaussian_filter(ori_score, sigma_s, order=1, axis=1)
        dx = gaussian_filter(ori_score, sigma_s, order=1, axis=2)
    cos = jnp.cos(thetas)[:, jnp.newaxis, jnp.newaxis]
    sin = jnp.sin(thetas)[:, jnp.newaxis, jnp.newaxis]
    zero = jnp.zeros_like(dx)
    # construct reference frame, want them to be row vectors
    d_zeta = jnp.stack([cos * dx, sin * dy, zero], axis=-1)
    d_eta = jnp.stack([cos * dy, -sin * dx, zero], axis=-1)
    d_t = jnp.stack([zero, zero, dt], axis=-1)
    return jnp.stack([d_zeta, d_eta, d_t], axis=-2)


def left_invariant_hessian2(
    ori_score: Float[Array, "t y x"],
    thetas: Float[Array, " t"],
    sigma_o: Float,
    sigma_s: Float,
) -> Float[Array, "t y x 3 3"]:
    """Compute the pixel-wise left-invariant Hessian for the input orientation scores.

    Args:
        ori_score (Float[Array, "t y x"]): array of orientation scores, where the angle is the first dimension and the spatial dimensions are the last two.
        thetas (Float[Array, "a"]): array of angles that was used to compute the orientation scores, corresponds to the first dimension of `ori_score`.
        sigma_o (float): standard deviation of Gaussian used to blur orientation axis.
        sigma_s (float): standard deviation of Gaussian used to blur spatial axes.

    Returns:
        Float[Array, "t y x 3 3"]: A 5D matrix where the first 3 dimensions match those of the input ori_score, and the last two correspond to the components of the Hessian.
    """
    lid_frame = left_invariant_derivative_frame2(
        ori_score, thetas, sigma_o, sigma_s
    )
    return _left_invariant_hessian_from_derivative_frame2(lid_frame, thetas)


def _left_invariant_hessian_from_derivative_frame2(
    lid_frame: Float[Array, "t y x 3 3"],
    thetas: Float[Array, " t"],
) -> Float[Array, "t y x 3 3"]:
    grad = jnp.sum(lid_frame, axis=-1)  # shape: (a b b 3)
    dfun = Partial(
        left_invariant_derivative_frame2, thetas=thetas, sigma_o=0, sigma_s=0
    )
    return jnp.sum(jax.vmap(dfun, -1, -1)(grad), axis=-1)
