""" Cake wavelets.

2D cake wavelets were introduced in [1], while 3D cake wavelets were introduced in [2].

References
---
[1] Bekkers, Erik, et al. "A multi-orientation analysis approach to retinal vessel tracking." Journal of Mathematical Imaging and Vision 49 (2014): 583-610.
[2] Janssen, Michiel HJ, et al. "Design and processing of invertible orientation scores of 3D images." Journal of mathematical imaging and vision 60 (2018): 1427-1458.
"""

import jax.numpy as jnp
from jax.scipy.special import factorial
from jax.tree_util import Partial
from jaxtyping import Array
from jaxtyping import Complex
from jaxtyping import Float
from jaxtyping import Num
from jaxtyping import Real

from .._util import angular_coordinate_grid_2d
from .._util import binomial_coefficient
from .._util import ifft2_centered
from .._util import radial_coordinate_grid_2d
from .._util import shift_remainder


def filter_bank_2d(
    size: int,
    n_scales: int,
    n_orientations: int,
    space: str,
    spline_order: int,
    overlap_factor: int,
    inflection_point_hf: float,
    poly_order_hf: int,
    inflection_point_lf: float | None,
    poly_order_lf: float | None,
    centered: bool = False,
) -> Num[Array, "{n_scales} {n_orientations} {size} {size}"]:
    """filter_bank_2d create a filter bank of 2D cake kernels.

    Args:

        size (int): size of (square) dimension of output
        n_scales (int): number of scales in the filter bank
        n_orientations (int): number of orientations to include in the bank
        space (str): output space of filters, 'real' or 'fourier'
        spline_order (int): order of b-spline
        overlap_factor (int): sets the amount of overlap between adjacent orientations in the bank
        inflection_point_hf (float): sets the high frequency cutoff of the wavelet
        poly_order_hf (int): order of high frequency filter polynomial. higher orders give a sharper rolloff
        inflection_point_lf (float, optional): sets the low frequency cutoff of the wavelet, defaults to None (no low freq. filtering)
        poly_order_lf (int, optional): order of low frequency filter polynomial. higher orders give a sharper rolloff. defaults to None, which means no low freq. filtering.
        centered (bool, optional): return filters centered in the output window (as if they had been `fftshift`'d), defaults to False which will match the API of the other wavelets.

    Raises:
        ValueError: if `space` isn't 'real' or 'fourier'

    Returns:
        Num[Array]: [n_orientations x size x size] array of filters (1 per orientation)
    """
    infl_pts = jnp.linspace(
        inflection_point_lf, inflection_point_hf, n_scales + 1
    )
    fun = Partial(
        orientation_bank_2d,
        size=size,
        n_orientations=n_orientations,
        space=space,
        spline_order=spline_order,
        overlap_factor=overlap_factor,
        poly_order_hf=poly_order_hf,
        poly_order_lf=poly_order_lf,
        centered=centered,
    )
    return jnp.stack(
        [
            fun(inflection_point_hf=iph, inflection_point_lf=ipl)
            for ipl, iph in zip(infl_pts[:-1], infl_pts[1:])
        ],
        axis=0,
    )


def orientation_bank_2d(
    size: int,
    n_orientations: int,
    space: str,
    spline_order: int,
    overlap_factor: int,
    inflection_point_hf: float,
    poly_order_hf: int,
    inflection_point_lf: float | None,
    poly_order_lf: float | None,
    centered: bool = False,
) -> Num[Array, " {n_orientations} {size} {size}"]:
    """orientation_bank_2d create a filter bank of 2D cake kernels.

    Args:
        n_orientations (int): number of orientations to include in the bank
        size (int): size of (square) dimension of output
        space (str): output space of filters, 'real' or 'fourier'
        spline_order (int): order of b-spline
        overlap_factor (int): sets the amount of overlap between adjacent orientations in the bank
        inflection_point_hf (float): sets the high frequency cutoff of the wavelet
        poly_order_hf (int): order of high frequency filter polynomial. higher orders give a sharper rolloff
        inflection_point_lf (float, optional): sets the low frequency cutoff of the wavelet, defaults to None (no low freq. filtering)
        poly_order_lf (int, optional): order of low frequency filter polynomial. higher orders give a sharper rolloff. defaults to None, which means no low freq. filtering.
        centered (bool, optional):

    Raises:
        ValueError: if `space` isn't 'real' or 'fourier'

    Returns:
        Num[Array]: [n_orientations x size x size] array of filters (1 per orientation)
    """
    if space == "real" or space == "r":
        fun = orientation_bank_2d_real
    elif space == "fourier" or space == "f":
        fun = Partial(orientation_bank_2d_fourier, centered=centered)
    else:
        raise ValueError('invalid output space, must be "real" or "fourier"')
    return fun(
        n_orientations,
        size,
        spline_order,
        overlap_factor,
        inflection_point_hf,
        poly_order_hf,
        inflection_point_lf,
        poly_order_lf,
    )


def orientation_bank_2d_real(
    n_orientations: int,
    size: int,
    spline_order: int,
    overlap_factor: int,
    inflection_point_hf: float,
    poly_order_hf: int,
    inflection_point_lf: float | None,
    poly_order_lf: float | None,
) -> Complex[Array, " {n_orientations} {size} {size}"]:
    """orientation_bank_2d_real create a filter bank of real-space 2D cake kernels.

    Args:
        n_orientations (int): number of orientations to include in the bank
        size (int): size of (square) dimension of output
        spline_order (int): order of b-spline
        overlap_factor (int): sets the amount of overlap between adjacent orientations in the bank
        inflection_point_hf (float): sets the high frequency cutoff of the wavelet
        poly_order_hf (int): order of high frequency filter polynomial. higher orders give a sharper rolloff
        inflection_point_lf (float): sets the low frequency cutoff of the wavelet
        poly_order_lf (int): order of low frequency filter polynomial. higher orders give a sharper rolloff

    Returns:
        Complex[Array]: [n_orientations x size x size] array of filters (1 per orientation)
    """
    psi_hat = orientation_bank_2d_fourier(
        n_orientations,
        size,
        spline_order,
        overlap_factor,
        inflection_point_hf,
        poly_order_hf,
        inflection_point_lf,
        poly_order_lf,
        centered=True,
    )
    return jnp.stack(
        [
            ifft2_centered(wvlet[0, ...])
            for wvlet in jnp.split(psi_hat, n_orientations, axis=0)
        ],
        axis=0,
    )


def orientation_bank_2d_fourier(
    n_orientations: int,
    size: int,
    spline_order: int,
    overlap_factor: int,
    inflection_point_hf: float,
    poly_order_hf: int,
    inflection_point_lf: float | None,
    poly_order_lf: int | None,
    centered: bool = False,
) -> Real[Array, " {n_orientations} {size} {size}"]:
    """orientation_bank_2d_fourier create a filter bank of fourier-space 2D cake kernels.

    Args:
        n_orientations (int): number of orientations to include in the bank
        size (int): size of (square) dimension of output
        spline_order (int): order of b-spline
        overlap_factor (int): sets the amount of overlap between adjacent orientations in the bank
        inflection_point_hf (float): sets the high frequency cutoff of the wavelet
        poly_order_hf (int): order of high frequency filter polynomial. higher orders give a sharper rolloff
        inflection_point_lf (float): sets the low frequency cutoff of the wavelet
        poly_order_lf (int): order of low frequency filter polynomial. higher orders give a sharper rolloff

    Returns:
        Complex[Array]: [n_orientations x size x size] array of filters (1 per orientation)
    """
    s_phi = (2 * jnp.pi) / n_orientations
    # rad_damping gives the "low-pass" component of the frequency-space filtering
    rad_damping = _radial_window(size, poly_order_hf, inflection_point_hf)
    if (
        inflection_point_lf is not None
    ):  # setup the high-pass component of the filtering
        dc_window = _radial_window(size, poly_order_lf, inflection_point_lf)
    else:
        dc_window = jnp.zeros_like(rad_damping)
    bandpass = rad_damping - dc_window
    ang_grid = angular_coordinate_grid_2d(size)
    thetas = jnp.linspace(0, 2 * jnp.pi, n_orientations, False)
    b_splines = [
        _bspline_profile(
            spline_order, shift_remainder(ang_grid - theta) / s_phi
        )
        / overlap_factor
        for theta in thetas
    ]
    if centered:
        return jnp.stack(
            [bandpass * b_spline for b_spline in b_splines], axis=0
        )
    else:
        return jnp.roll(
            jnp.stack([bandpass * b_spline for b_spline in b_splines], axis=0),
            (-size / 2, -size / 2),
            axis=(-2, -1),
        )


def _bspline_profile(
    n: int, angle_grid: Float[Array, " a a"]
) -> Float[Array, " a a"]:
    """_bspline_profile compute b-spline of order k=n+2.

    Args:
        n (int): spline order
        angle_grid (Float[Array]): 2d grid of angles to compute bspline on

    Returns:
        Float[Array]
    """
    splines, ang_ints = [], []
    orders = jnp.linspace(-n / 2, n / 2, n + 1, True)
    eps = jnp.finfo(float).eps
    for j in range(orders.size):
        ospline = jnp.zeros_like(angle_grid)
        order = orders[j]
        for k in range(n + 2):
            ospline += (
                binomial_coefficient(n + 1, k)
                * jnp.power((angle_grid + (n + 1) / 2 - k), n)
                * jnp.power(-1, k)
                * jnp.sign(order + (n + 1) / 2 - k)
            )
        splines.append(ospline / (2 * factorial(n)))
    ang_interval = jnp.heaviside(
        (angle_grid - (order - 0.5 + eps)), 1
    ) * jnp.heaviside((-(angle_grid - (order + 0.5))), 1)
    ang_ints.append(ang_interval)
    return jnp.sum(
        jnp.stack(ang_ints, axis=0) * jnp.stack(splines, axis=0), axis=0
    )


def _radial_window(
    size: int, n: int, inflection_point: float
) -> Float[Array, " {size size}"]:
    """_radial_window windowing function for radial dimension (frequency).

    Windowing function, M_N, is essentially a Gaussian multiplied with the
    Taylor series of its inverse up to a finite order 2N.

    Args:
        size (int): size of output window (output will be 2d matrix size x size)
        n (int): 1/2 order of Taylor series of inverse
        inflection_point (float): gamma, 0 << 1, ~sets frequency cutoff
    Returns:
        Float[Array]
    """
    grid = radial_coordinate_grid_2d(size)
    rho_coeff = 1 / jnp.sqrt(
        2 * jnp.square(inflection_point * jnp.floor(size / 2)) / (1 + 2 * n)
    )
    rho = grid * rho_coeff
    window = jnp.zeros([size, size])
    for k in range(n + 1):
        window += (
            jnp.power(rho, 2 * k) / factorial(k) * jnp.exp(-jnp.square(rho))
        )
    return window
