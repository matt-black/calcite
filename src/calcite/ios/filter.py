"""Orientation score-based filtering functions."""

from collections.abc import Sequence

import jax
import jax.numpy as jnp
from jaxtyping import Array
from jaxtyping import Bool
from jaxtyping import Float

from .lad import _optimal_tangent_vector2_from_hessian
from .lad import d_h
from .lad import kappa
from .lad import q_kappamu
from .lad import r_dh
from .lid import _left_invariant_hessian_from_derivative_frame2
from .lid import left_invariant_derivative_frame2


def lidos2(
    ori_score: Float[Array, "t y x"],
    thetas: Float[Array, " t"],
    sigma_o: Float,
    sigma_s: Float,
    normalize: Bool = False,
) -> Float[Array, "t y x"]:
    """Compute the 2D left-invariant derivative on orientation scores (LID-OS) filter on the input orientations scores.

    Args:
        ori_score (Float[Array, "t y x"]): orientation score array.
        thetas (Float[Array, "t"]): angles sampled to generate orientation scores.
        sigma_o (float): standard deviation of Gaussian used to blur orientation axis.
        sigma_s (float): standard deviation of Gaussian used to blur spatial axes.
        normalize (bool): normalize the filter response.

    Returns:
        Float[Array, "t y x"]
    """
    lid = left_invariant_derivative_frame2(ori_score, thetas, sigma_o, sigma_s)
    dd_eta = jnp.einsum("...i,...i->...", lid[..., 1, :], lid[..., 1, :])
    # compute second order derivatives
    if normalize:
        return (jnp.square(sigma_s) / jnp.square(sigma_o)) * dd_eta
    else:
        return dd_eta


def multiscale_lidos2(
    ori_score: Float[Array, "t y x"],
    thetas: Float[Array, " t"],
    sigma_o: Float,
    sigma_s: Sequence[float],
    normalize: Bool = False,
) -> Float[Array, "y x"]:
    """Compute the multi-scale LID-OS filter and do image reconstruction on the input orientation scores.

    Args:
        ori_score (Float[Array, "t y x"]): array of orientation scores, where the angle is the first dimension and the spatial dimensions are the last two.
        thetas (Float[Array, "a"]): array of angles that was used to compute the orientation scores, corresponds to the first dimension of `ori_score`.
        sigma_o (float): standard deviation of Gaussian used to blur orientation axis.
        sigma_s (Sequence[float]): per-scale standard deviation of Gaussians used to blur spatial axes.
        normalize (bool): whether to normalize the scores at each scale or not. Defaults to False.

    Returns:
        Float[Array, "y x"]: reconstructed image.
    """
    scales = []
    for scale_sigma in sigma_s:
        scales.append(
            lidos2(ori_score, thetas, sigma_o, scale_sigma, normalize)
        )
    scales = jnp.stack(scales, axis=0)
    return jnp.amax(jnp.sum(scales, axis=0), axis=0)


def lados2(
    ori_score: Float[Array, "t y x"],
    thetas: Float[Array, " t"],
    sigma_o: Float,
    sigma_s: Float,
    mu: Float,
    normalize: Bool = False,
) -> Float[Array, "t y x"]:
    """Compute the 2D LAD-OS filter on the input orientation scores.

    Args:
        ori_score (Float[Array, "t y x"]): array of orientation scores, where the angle is the first dimension and the spatial dimensions are the last two.
        thetas (Float[Array, "a"]): array of angles that was used to compute the orientation scores, corresponds to the first dimension of `ori_score`.
        sigma_o (float): standard deviation of Gaussian used to blur orientation axis.
        sigma_s (float): standard deviation of Gaussian used to blur spatial axes.
        mu (float): intrinsic parameter to balance spatial and orientation distances.
        normalize (bool): whether to normalize the scores at each scale or not. Defaults to False.

    Returns:
        Float[Array, "t y x"]
    """
    lid = left_invariant_derivative_frame2(ori_score, thetas, sigma_o, sigma_s)
    hess_uf = _left_invariant_hessian_from_derivative_frame2(lid, thetas)
    c = _optimal_tangent_vector2_from_hessian(hess_uf, mu)
    qr = jnp.matrix_transpose(q_kappamu(kappa(c), mu)) @ jnp.matrix_transpose(
        r_dh(d_h(c))
    )
    d_b = jnp.matmul(qr, lid)[..., 1, :][..., None, :]
    dd_b = d_b @ hess_uf @ d_b.transpose(0, 1, 2, 4, 3)
    return jnp.nan_to_num(
        jax.lax.select(
            normalize,
            -(jnp.square(sigma_s) / jnp.square(sigma_o)) * dd_b[..., 0, 0],
            -dd_b[..., 0, 0],
        ).real
    )


def multiscale_lados2(
    ori_score: Float[Array, "t y x"],
    thetas: Float[Array, " t"],
    sigma_o: Float,
    sigma_s: Sequence[float],
    mu: Float,
    normalize: Bool = False,
) -> Float[Array, "y x"]:
    """Compute the multi-scale LAD-OS filter and do image reconstruction on the input orientation scores.

    Args:
        ori_score (Float[Array, "t y x"]): array of orientation scores, where the angle is the first dimension and the spatial dimensions are the last two.
        thetas (Float[Array, "a"]): array of angles that was used to compute the orientation scores, corresponds to the first dimension of `ori_score`.
        sigma_o (float): standard deviation of Gaussian used to blur orientation axis.
        sigma_s (Sequence[float]): per-scale standard deviation of Gaussians used to blur spatial axes.
        normalize (bool): whether to normalize the scores at each scale or not. Defaults to False.

    Returns:
        Float[Array, "y x"]
    """
    scales = []
    for scale_sigma in sigma_s:
        scales.append(
            lados2(ori_score, thetas, sigma_o, scale_sigma, mu, normalize)
        )
    scales = jnp.stack(scales, axis=0)
    return jnp.amax(jnp.sum(scales, axis=0), axis=0)
