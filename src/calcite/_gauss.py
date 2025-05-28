from numbers import Number
from typing import List

import jax.numpy as jnp
import jax.scipy as jsp
from jax._src.lax.lax import PrecisionLike
from jaxtyping import Array
from jaxtyping import Float
from jaxtyping import Int
from jaxtyping import Num
from jaxtyping import Real


def gaussian_kernel_1d(
    sigma: float, order: int, radius: int
) -> Float[Array, " 2 * {radius}"]:
    """gaussian_kernel_1d 1-D gaussian kernel.

    Args:
        sigma (float): standard deviation
        order (int): 0 order is a Gaussian, order>0 corresp. to derivatives
        radius (int): radius of the kernel

    Raises:
        ValueError: if order < 0

    Returns:
        Float[Array]
    """
    if order < 0:
        raise ValueError("order must be nonnegative")
    sigma2 = jnp.square(sigma)
    x = jnp.arange(-radius, radius + 1)
    phi_x = jnp.exp(-0.5 / sigma2 * x**2)
    phi_x = phi_x / phi_x.sum()
    if order == 0:
        return phi_x
    else:
        # f(x) = q(x) * phi(x) = q(x) * exp(p(x))
        # f'(x) = (q'(x) + q(x) * p'(x)) * phi(x)
        # p'(x) = -1 / sigma ** 2
        # Implement q'(x) + q(x) * p'(x) as a matrix operator and apply to the
        # coefficients of q(x)
        exponent_range = jnp.arange(order + 1)
        q = jnp.zeros(order + 1).at[0].set(1)
        diag_exp = jnp.diag(exponent_range[1:], 1)  # diag_exp @ q(x) = q'(x)
        diag_p = jnp.diag(
            jnp.ones(order) / -sigma2, -1
        )  # diag_p @ q(x) = q(x) * p'(x)
        qmat_deriv = diag_exp + diag_p
        for _ in range(order):
            q = qmat_deriv.dot(q)
        q = (x[:, None] ** exponent_range).dot(q)
        return q * phi_x


def gaussian_filter_1d(
    input: Real[Array, " a"],
    sigma: float,
    axis: int = -1,
    order: int = 0,
    truncate: float = 4.0,
    radius: int = 0,
    mode: str = "constant",
    cval: float = 0.0,
    precision: PrecisionLike | None = None,
) -> Real[Array, " a"]:
    """gaussian_filter_1d 1-D gaussian filter.

    Args:
        input (Real[Array]): input array
        sigma (float): standard deviation of Gaussian
        axis (int): axis of `input` along which to calculate, optional. Defaults to -1.
        order (int): order of 0 is Gaussian, higher orders are derivatives, optional. Defaults to 0.
        truncate (float): truncate filter at this many std. dev's, optional. Defaults to 4.
        mode (str): how input array is extended beyond boundaries, optional. Defaults to 'constant'.
        cval (float): value to use for `mode='constant'`, optional. Defaults to 0.
        precision (PrecisionLike): precision to use for calculation, optional. Defaults to None.

    Raises:
        NotImplementedError: if mode != 'constant'
        ValueError: if radius < 0

    Returns:
        Real[Array]
    """
    # make the radius of the filter equal to truncate standard deviations
    lw = int(truncate * sigma + 0.5)
    if mode != "constant" or cval != 0.0:
        raise NotImplementedError(
            'Other modes than "constant" with 0. fill value are not'
            "supported yet."
        )
    if radius > 0.0:
        lw = radius
    if lw < 0:
        raise ValueError("Radius must be a nonnegative integer.")
    weights = gaussian_kernel_1d(sigma, order, lw)[::-1]
    # Be careful that modes in signal.convolve refer to the
    # 'same' 'full' 'valid' modes, while in gaussian_filter1d refers to the
    # way the padding is done 'constant' 'reflect' etc.
    return jnp.apply_along_axis(
        jsp.signal.convolve,
        axis,
        input,
        weights,
        mode="same",
        method="fft",
        precision=precision,
    )


def gaussian_filter(
    input: Real[Array, "..."],
    sigma: float | List[float] | Real[Array, " b"],
    order: int = 0,
    truncate: float = 4.0,
    *,
    radius: int = 0,
    mode: str = "constant",
    cval: float = 0.0,
    axis: int | None = None,
    precision: PrecisionLike | None = None,
) -> Real[Array, " a"]:
    """gaussian_filter multi-dimensional Gaussian filter.

    Args:
        input (Real[Array]): input array
        sigma (float): standard deviation of Gaussian
        axis (int): axis of `input` along which to calculate, optional. Defaults to -1.
        order (int): order of 0 is Gaussian, higher orders are derivatives, optional. Defaults to 0.
        truncate (float): truncate filter at this many std. dev's, optional. Defaults to 4.
        mode (str): how input array is extended beyond boundaries, optional. Defaults to 'constant'.
        cval (float): value to use for `mode='constant'`, optional. Defaults to 0.
        precision (PrecisionLike): precision to use for calculation, optional. Defaults to None.

    Returns:
        Real[Array]
    """
    if isinstance(sigma, Number) and axis is None:
        sigma = [sigma,] * input.ndim # type: ignore
    for ax, _sigma in zip(range(input.ndim), sigma): # type: ignore
        input = gaussian_filter_1d(
            input,
            _sigma,
            axis=ax,
            order=order,
            truncate=truncate,
            radius=radius,
            mode=mode,
            precision=precision,
            cval=cval,
        )
    return input


def _gauss_prefactor(
    sigma: float, d: int,
) -> Float[Array, ""]:
    """_gauss_prefactor constant prefactor for gaussians.

    Args:
        sigma (float): standard deviation of gaussian
        d (int): number of dimensions

    Returns:
        float : prefactor value
    """
    return jnp.power(2 * jnp.pi * jnp.square(sigma), -float(d) / 2)


def gaussian_1d(size: int, sigma: float) -> Float[Array, " {size}"]:
    """gaussian_1d create a 1d gaussian vector.

    Args:
        size (int): size of output vector
        sigma (float): standard deviation of gaussian

    Raises:
        ValueError: if size is not odd

    Returns:
        Float[Array]: gaussian vector
    """
    if not size % 2 > 0:
        raise ValueError("size must be odd")

    x = jnp.arange(size)
    cent = size // 2
    return _gauss_prefactor(sigma, 1) * jnp.exp(
        -jnp.square(x - cent) / (2 * jnp.square(sigma))
    )


def gaussian_2d(
    size: int, sigma_x: float, sigma_y: float
) -> Float[Array, " {size size}"]:
    """gaussian_2d create a 2d gaussian on a grid.

    Args:
        size (int): size of output vector
        sigma_x (float): standard deviation of gaussian in x direction
        sigma_y (float): standard deviation of gaussian in y direction

    Raises:
        ValueError: if size is not odd

    Returns:
        Float[Array]: grid of values from 2d gaussian
    """
    cent = size / 2
    x, y = jnp.meshgrid(jnp.arange(size), jnp.arange(size), indexing="xy")
    gx = _gauss_prefactor(sigma_x, 1) * jnp.exp(
        -jnp.square(x - cent) / (2 * jnp.square(sigma_x))
    )
    gy = _gauss_prefactor(sigma_y, 1) * jnp.exp(
        -jnp.square(y - cent) / (2 * jnp.square(sigma_y))
    )
    return (gx + gy) / jnp.sum(gx + gy)


def convolve_2d_gaussian(
    x: Num[Array, "r c"],
    sigma_x: float,
    sigma_y: float,
    n_stds: float = 4,
) -> Num[Array, "r c"]:
    """convolve_2d_gaussian convolve the input array with a gaussian.

    Args:
        x (Num[Array, "r c"]): the input array (2D)
        sigma_x (float): standard deviation in the x-direction
        sigma_y (float): standard deviation in the y-direction
        n_stds (float): number of standard deviations-wide the filter will be.

    Returns:
        Num[Array, "r c"]: blurred input.
    """
    return gaussian_filter(
        gaussian_filter_1d(x, sigma_x, axis=1, order=0, truncate=n_stds),
        sigma_y,
        0,
        truncate=n_stds,
    )


def convolve(x: Array, *sigmas, **kwargs) -> Array:
    """Convolve the input array with an ND Gaussian.

    Args:
        x (Array): array to be convolved
        sigmas (float): per-axis standard deviation of kernel
        kwargs: keyword arguments, passed to ``gaussian_filter_1d``

    Returns:
        Array
    """
    n_dim = len(x.shape)
    if len(sigmas) == 1:
        sigmas = [
            sigmas,
        ] * n_dim
    else:
        if not len(sigmas) == n_dim:
            raise ValueError(
                "invalid # of sigmas, must be 1 or 1 per-axis of input array"
            )
    out = x.copy()
    for axis, sigma in enumerate(sigmas):
        out = gaussian_filter_1d(out, sigma, axis, order=0, **kwargs) # type: ignore
    return out
