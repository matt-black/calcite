"""Locally Adaptive Differential (LAD) frames.

References:
    [1] Zhang, Jiong, et al. "Robust retinal vessel segmentation via locally adaptive derivative frames in orientation scores." IEEE transactions on medical imaging 35.12 (2016): 2631-2644.
    [2] Duits, Remco, et al. "Locally adaptive frames in the roto-translation group and their applications in medical imaging." Journal of Mathematical Imaging and Vision 56 (2016): 367-402.
"""

import jax
import jax.numpy as jnp
from jaxtyping import Array
from jaxtyping import Float

from .lid import left_invariant_derivative_frame2
from .lid import left_invariant_hessian2


def locally_adaptive_derivative_frame2(
    ori_score: Float[Array, "t y x"],
    thetas: Float[Array, " t"],
    sigma_o: Float,
    sigma_s: Float,
    mu: Float,
) -> Float[Array, "t y x 3 3"]:
    """Calculate the locally adaptive derivative frame (LAD) for the orientation score input, `ori_score`, of some original 2D image.

    Args:
        ori_score (Float[Array, "t y x"]): array of orientation scores, where the angle is the first dimension and the spatial dimensions are the last two.
        thetas (Float[Array, "a"]): array of angles that was used to compute the orientation scores, corresponds to the first dimension of `ori_score`.
        mu (float): intrinsic parameter to balance spatial and orientation distances (has dimension 1/length, see `mu_matrix`.)
        sigma_o (float): standard deviation of Gaussian used to blur orientation axis.
        sigma_s (float): standard deviation of Gaussian used to blur spatial axes.

    Returns:
        Float[Array, "t y x 3 3"]: LAD frame, (d_a, d_b, d_c) at each point in orientation space.
    """
    c = optimal_tangent_vector2(ori_score, thetas, mu, sigma_o, sigma_s)
    qr = jnp.matrix_transpose(q_kappamu(kappa(c), mu)) @ jnp.matrix_transpose(
        r_dh(d_h(c))
    )
    lid = left_invariant_derivative_frame2(ori_score, thetas, sigma_o, sigma_s)
    return jnp.matmul(qr, lid)


def optimal_tangent_vector2(
    ori_score: Float[Array, "t y x"],
    thetas: Float[Array, " t"],
    sigma_o: Float,
    sigma_s: Float,
    mu: Float,
) -> Float[Array, "t y x 3"]:
    """Calculate optimal tangent vectors, used to define the LAD frame, at each point in orientation space.

    Args:
        ori_score (Float[Array, "t y x"]): orientation scores.
        thetas (Float[Array, "a"]): angles used to compute orientation
        sigma_o (float): standard deviation of Gaussian used to blur orientation axis.
        sigma_s (float): standard deviation of Gaussian used to blur spatial axes.
        mu (Float): intrinsic parameter to balance spatial and orientation distances (has dimension 1/length, see `mu_matrix`.).

    Returns:
        Float[Array, "t y x 3"]: optimal tangent vector at each point in orientation space.
    """
    hess_uf = left_invariant_hessian2(ori_score, thetas, sigma_o, sigma_s)
    return _optimal_tangent_vector2_from_hessian(hess_uf, mu)


def _optimal_tangent_vector2_from_hessian(
    hessian: Float[Array, "t y x 3 3"],
    mu: Float,
) -> Float[Array, "t y x 3"]:
    hess_uf_symm = symmetrized_mu_normalized_left_invariant_hessian2(
        hessian, mu
    )
    eigval, eigvec = jnp.linalg.eig(hess_uf_symm)
    idx = jnp.argsort(eigval, axis=-1, descending=False)
    return jax.vmap(lambda m, i: m[:, i], (0, 0), 0)(
        eigvec.reshape(-1, 3, 3), idx[..., 0].flatten()
    ).reshape(eigval.shape)


def symmetrized_mu_normalized_left_invariant_hessian2(
    hessian: Float[Array, "t y x 3 3"], mu: Float
) -> Float[Array, "t y x 3 3"]:
    """Symmetrize and normalize the hessian matrix, `hessian`.

    Args:
        hessian (Float[Array, "t y x 3 3"]):
        mu (float): intrinsic parameter to balance spatial and orientation distances (has dimension 1/length, see `mu_matrix`.)
    Returns:
        Float[Array, "t y x 3 3"]

    Notes:
        See Eqn. 6 of [1].

    References:
        [1] Zhang, Jiong, et al. "Robust retinal vessel segmentation via locally adaptive derivative frames in orientation scores." IEEE transactions on medical imaging 35.12 (2016): 2631-2644.
    """
    mu_mat1 = mu_matrix(mu)[jnp.newaxis, jnp.newaxis, jnp.newaxis, ...]
    mu_mat2 = mu_matrix(jnp.square(mu))[
        jnp.newaxis, jnp.newaxis, jnp.newaxis, ...
    ]
    return mu_mat1 @ jnp.matrix_transpose(hessian) @ mu_mat2 @ hessian @ mu_mat1


def mu_matrix(mu: Float) -> Float[Array, "3 3"]:
    """Construct diagonal matrix for balancing spatial and orientation distances when computing gradients.

    This matrix is diag([1/mu, 1/mu, 1]), so mu should be set such that it will normalize distances to the same spacing as the angles sampled in orientation space.

    Args:
        mu (float): intrinsic parameter to balance spatial and orientation distances (has dimension 1/length).

    Returns:
        Float[Array, "3 3"]
    """
    return jnp.diag(jnp.asarray([1.0 / mu, 1.0 / mu, 1.0]))


def _local_curvature(c: Float[Array, "t y x 3"]) -> Float[Array, "t y x"]:
    return (c[..., 2] * jnp.sign(c[..., 0])) / jnp.sqrt(
        jnp.square(c[..., 1]) + jnp.square(c[..., 0])
    )


def kappa(c: Float[Array, "t y x 3"]) -> Float[Array, "t y x"]:
    """Compute the local curvature from the tangent vector, `c`.

    Args:
        c (Float[Array, "t y x 3"]): tangent vector array at each point in orientation space.

    Returns:
        Float[Array, "t y x"]: local curvature at each point in orientation space.

    Notes:
        See Eqn. 7 of [1].

    References:
        [1] Zhang, Jiong, et al. "Robust retinal vessel segmentation via locally adaptive derivative frames in orientation scores." IEEE transactions on medical imaging 35.12 (2016): 2631-2644.
    """
    return _local_curvature(c)


def _deviation_from_horizontality(
    c: Float[Array, "t y x 3"],
) -> Float[Array, "t y x"]:
    return jnp.arctan2(c[..., 1], c[..., 0])


def d_h(c: Float[Array, "t y x 3"]) -> Float[Array, "t y x"]:
    """Compute deviation from horizontality for tanget vectors, c.

    Args:
        c (Float[Array, "t y x 3"]): tangent vector array at each point in orientation space.

    Returns:
        Float[Array, "t y x"]: deviation from horizontality at each point in orientation space.

    Notes:
        See Eqn. 7 of [1].

    References:
        [1] Zhang, Jiong, et al. "Robust retinal vessel segmentation via locally adaptive derivative frames in orientation scores." IEEE transactions on medical imaging 35.12 (2016): 2631-2644.
    """
    return _deviation_from_horizontality(c)


def r_dh(d_h: Float[Array, "t y x"]) -> Float[Array, "t y x 3 3"]:
    """Compute R_{d_h} matrix for rotating the LID frame into the LAD frame.

    Args:
        d_h (Float[Array, "t y x"]): local deviatino from horizontality at each pixel.

    Returns:
        Float[Array, "t y x 3 3"]: rotation matrix @ each pixel.
    """
    cos_dh = jnp.cos(d_h)
    sin_dh = jnp.sin(d_h)
    zero = jnp.zeros_like(d_h)
    row1 = jnp.stack([cos_dh, -sin_dh, zero], axis=-1)
    row2 = jnp.stack([sin_dh, cos_dh, zero], axis=-1)
    row3 = jnp.stack([zero, zero, jnp.ones_like(d_h)], axis=-1)
    return jnp.stack([row1, row2, row3], axis=-1)


def q_kappamu(
    kappa: Float[Array, "t y x"], mu: Float
) -> Float[Array, "t y x 3 3"]:
    r"""Compute Q_{\kappa,\mu} matrix for rotating the LID frame into the LAD frame.

    Args:
        kappa (Float[Array, "a b"]): local curvature at each pixel, kappa.
        mu (Float): intrinsic parameter to balance spatial and orientation distances (has dimension 1/length, see `mu_matrix`.)

    Returns:
        Float[Array, "t y x 3 3"]: rotation matrix @ each pixel.
    """
    denom = jnp.sqrt(jnp.square(mu) + jnp.square(kappa))
    zero = jnp.zeros_like(kappa)
    row1 = jnp.stack([mu / denom, zero, kappa / denom], axis=-1)
    row2 = jnp.stack([zero, jnp.ones_like(zero), zero], axis=-1)
    row3 = jnp.stack([-kappa / denom, zero, mu / denom], axis=-1)
    return jnp.stack([row1, row2, row3], axis=-1)
