"""Cake wavelets.

2D cake wavelets were introduced in [1], while 3D cake wavelets were introduced in [2].

References
---
[1] Bekkers, Erik, et al. "A multi-orientation analysis approach to retinal vessel tracking." Journal of Mathematical Imaging and Vision 49 (2014): 583-610.
[2] Janssen, Michiel HJ, et al. "Design and processing of invertible orientation scores of 3D images." Journal of mathematical imaging and vision 60 (2018): 1427-1458.
"""

from typing import Tuple

import jax
import jax.numpy as jnp
from jax.scipy.special import factorial
from jax.tree_util import Partial
from jaxtyping import Array
from jaxtyping import Complex
from jaxtyping import Float
from jaxtyping import Num
from jaxtyping import Real
from scipy.special import spherical_jn

from .._util import angular_coordinate_grid_2d
from .._util import angular_coordinate_grids_3d
from .._util import binomial_coefficient
from .._util import eval_legendre
from .._util import ifft2_centered
from .._util import radial_coordinate_grid_2d
from .._util import radial_coordinate_grid_3d
from .._util import shift_remainder
from ..qsampling import optimize_singleshell
from ..qsampling import xyz_to_angle


__all__ = [
    "filter_bank_2d",
    "orientation_bank_2d",
    "orientation_bank_2d_real",
    "orientation_bank_2d_fourier",
    "orientation_bank_3d_fourier",
    "cake_wavelet_3d_fourier",
    "cake_wavelet_3d_real",
    "split_cake_wavelet_fourier",
]


def filter_bank_2d(
    size: int,
    n_scales: int,
    n_orientations: int,
    space: str,
    spline_order: int,
    overlap_factor: int,
    inflection_point_hf: float,
    poly_order_hf: int,
    inflection_point_lf: float,
    poly_order_lf: float | None,
    centered: bool = False,
) -> Num[Array, "{n_scales} {n_orientations} {size} {size}"]:
    """Create a filter bank of 2D cake kernels.

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
        return_angles=False,
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
    poly_order_lf: int | None,
    centered: bool = False,
    return_angles: bool = False,
) -> (
    Num[Array, "{n_orientations} {size} {size}"]
    | Tuple[
        Num[Array, "{n_orientations} {size} {size}"],
        Float[Array, " {n_orientations}"],
    ]
):
    """Create a filter bank of 2D cake kernels.

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
        return_angles=return_angles,
    )


def orientation_bank_2d_real(
    n_orientations: int,
    size: int,
    spline_order: int,
    overlap_factor: int,
    inflection_point_hf: float,
    poly_order_hf: int,
    inflection_point_lf: float | None,
    poly_order_lf: int | None,
    return_angles: bool = False,
) -> (
    Complex[Array, " {n_orientations} {size} {size}"]
    | Tuple[
        Complex[Array, " {n_orientations} {size} {size}"],
        Float[Array, " {n_orientations}"],
    ]
):
    """Create a filter bank of real-space 2D cake kernels.

    Args:
        n_orientations (int): number of orientations to include in the bank
        size (int): size of (square) dimension of output
        spline_order (int): order of b-spline
        overlap_factor (int): sets the amount of overlap between adjacent orientations in the bank
        inflection_point_hf (float): sets the high frequency cutoff of the wavelet
        poly_order_hf (int): order of high frequency filter polynomial. higher orders give a sharper rolloff
        inflection_point_lf (float): sets the low frequency cutoff of the wavelet
        poly_order_lf (int): order of low frequency filter polynomial. higher orders give a sharper rolloff
        return_angles (bool, optional): whether to (also) return the angle at which each filter corresponds to. Defaults to False.
    Returns:
        Complex[Array]: [n_orientations x size x size] array of filters (1 per orientation)
    """
    if return_angles:
        psi_hat, thetas = orientation_bank_2d_fourier(
            n_orientations,
            size,
            spline_order,
            overlap_factor,
            inflection_point_hf,
            poly_order_hf,
            inflection_point_lf,
            poly_order_lf,
            centered=True,
            return_angles=True,
        )
        return (
            jnp.stack(
                [
                    ifft2_centered(wvlet[0, ...])
                    for wvlet in jnp.split(psi_hat, n_orientations, axis=0)
                ],
                axis=0,
            ),
            thetas,
        )
    else:
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
            return_angles=False,
        )
        return jnp.stack(
            [
                ifft2_centered(wvlet[0, ...])
                for wvlet in jnp.split(psi_hat, n_orientations, axis=0)  # type: ignore
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
    return_angles: bool = False,
) -> (
    Real[Array, "{n_orientations} {size} {size}"]
    | Tuple[
        Real[Array, "{n_orientations} {size} {size}"],
        Float[Array, " {n_orientations}"],
    ]
):
    """Create a filter bank of fourier-space 2D cake kernels.

    Args:
        n_orientations (int): number of orientations to include in the bank
        size (int): size of (square) dimension of output
        spline_order (int): order of b-spline
        overlap_factor (int): sets the amount of overlap between adjacent orientations in the bank
        inflection_point_hf (float): sets the high frequency cutoff of the wavelet
        poly_order_hf (int): order of high frequency filter polynomial. higher orders give a sharper rolloff
        inflection_point_lf (float): sets the low frequency cutoff of the wavelet
        poly_order_lf (int): order of low frequency filter polynomial. higher orders give a sharper rolloff
        return_angles (bool, optional): whether to (also) return the angle at which each filter corresponds to. Defaults to False.

    Returns:
        Real[Array]: [n_orientations x size x size] array of filters (1 per orientation)
    """
    s_phi = (2 * jnp.pi) / n_orientations
    # rad_damping gives the "low-pass" component of the frequency-space filtering
    rad_damping = _radial_window_2d(size, poly_order_hf, inflection_point_hf)
    if (
        inflection_point_lf is not None
    ):  # setup the high-pass component of the filtering
        dc_window = _radial_window_2d(size, poly_order_lf, inflection_point_lf)  # type: ignore
    else:
        dc_window = jnp.zeros_like(rad_damping)
    bandpass = rad_damping - dc_window
    ang_grid = angular_coordinate_grid_2d(size)
    thetas = jnp.linspace(0, 2 * jnp.pi, n_orientations, False)
    ang_grids = [
        jnp.flipud(
            jnp.fliplr(
                jnp.abs(shift_remainder(ang_grid - theta) - jnp.pi) / s_phi
            )
        )
        for theta in thetas
    ]
    b_splines = [
        _bspline_profile_2d(spline_order, ang_grid) / overlap_factor
        for ang_grid in ang_grids
    ]
    if centered:
        bank = jnp.stack(
            [bandpass * b_spline for b_spline in b_splines], axis=0
        )
    else:
        bank = jnp.roll(
            jnp.stack([bandpass * b_spline for b_spline in b_splines], axis=0),
            (-size // 2, -size // 2),
            axis=(-2, -1),
        )
    if return_angles:
        return bank, thetas
    else:
        return bank


def orientation_bank_3d_fourier(
    size: int,
    num_ori: int,
    gamma_window: float,
    nyquist_freq: float,
    sigma_erf: float,
    s_0: float,
    s_rho: float | None,
    big_ell: int,
    centered: bool = False,
    return_angles: bool = False,
    angle_fmt: str = "spherical",
) -> (
    Complex[Array, "{num_ori} {size} {size} {size}"]
    | Tuple[
        Complex[Array, "{num_ori} {size} {size} {size}"],
        Float[Array, "..."],
    ]
):
    """Generate a filter bank of 3D cake wavelets.

    Args:
        size (int): size of output wavelet (will be size^3)
        num_ori (int): num. of orientations in the filter bank
        gamma_window (float): gamma parameter for the radial window, fixes the inflection point for the rolloff at gamma*nyquist_freq
        nyquist_freq (float): nyquist frequency of the data
        sigma_erf (float): controls steepness of the decay around the Nyquist frequency.
        s_0 (float): controls tradeoff between more uniform reconstruction at the cost of less directionality.
        s_rho (float|None):
        big_ell (int): order to compute the spherical harmonics up to.
        centered (bool, optional): whether the wavelets are centered in Fourier space (as if fftshift'd). Defaults to False.
        return_angles (bool, optional): return angles corresponding to each wavelet in the filter bank. Defaults to False.
        angle_fmt (str, optional): the format to return the angles in, if `return_angles` is set to `True`. Defaults to "spherical".

    Returns:
        Complex[Array, {num_ori} {size} {size} {size}] | Tuple[Complex[Array, {num_ori} {size} {size} {size}], Float[Array, ...]]
    """
    xyz = optimize_singleshell(num_ori, max_iter=100, antipodal=False)
    abg = xyz_to_angle(xyz, "euler_xyz")
    cake_fun = Partial(
        cake_wavelet_3d_fourier,
        size,
        gamma_window,
        nyquist_freq,
        sigma_erf,
        s_0,
        s_rho,
        big_ell=big_ell,
        centered=centered,
    )
    bank = jnp.stack(
        [
            cake_fun(abg[i, 0], abg[i, 1], abg[i, 2])
            for i in range(abg.shape[0])
        ],
        axis=0,
    )
    if return_angles:
        angles = jnp.asarray(xyz_to_angle(xyz, angle_fmt))
        return bank, angles
    else:
        return bank


def _bspline_profile_2d(
    n: int, angle_grid: Float[Array, "a a"]
) -> Float[Array, " a a"]:
    """Compute b-spline of order k=n+2.

    Args:
        n (int): spline order
        angle_grid (Float[Array, "a a"]): 2d grid of angles to compute bspline on

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


def _radial_window_2d(
    size: int, n: int, inflection_point: float
) -> Float[Array, " {size size}"]:
    """Windowing function for radial dimension (frequency).

    Windowing function, M_N, is essentially a Gaussian multiplied with the
    Taylor series of its inverse up to a finite order 2N.

    Args:
        size (int): size of output window (output will be 2d matrix size x size)
        n (int): 1/2 order of Taylor series of inverse
        inflection_point (float): gamma, 0 >> 1, ~sets frequency cutoff
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


def _radial_window_fft_3d(
    size: int,
    gamma: float,
    nyquist_freq: float,
    sigma_erf: float | None = None,
):  # -> Float[Array, " {size} {size} {size}"]:
    """Windowing function for radial dimension (frequency) in three dimensions.

    See Eqn. 50, Fig. 5 of [2].

    Args:
        size (int): size of output window (output will be 3d matrix (size x size x size))
        gamma (float): controls inflection point of the error function
        nyquist_freq (float): nyquist frequency of the data
        sigma_erf (float | None, optional): controls steepness of the decay when approaching the nyquist frequency. Defaults to None, which will use (nyquist-rho2)/3.

    Returns:
        Float[Array]
    """
    rho = radial_coordinate_grid_3d(size)
    infl_pt = gamma * nyquist_freq  # Eqn. 51
    if sigma_erf is None:
        sigma_erf = (nyquist_freq - infl_pt) / 3.0
    window = 0.5 * (1 - jax.scipy.special.erf((rho - infl_pt) / sigma_erf))
    return window


def _low_frequency_gaussian_window(
    size: int,
    s_rho: float,
    num_spatial_dim: int,
) -> Float[Array, "{size} {size} {size}"]:
    """Filter to pick out low frequencies for wavelet splitting.

    Args:
        size (int): size of output window
        s_rho (float): variance of distribution

    Returns:
        Float[Array]
    """
    if num_spatial_dim == 3:
        rho = radial_coordinate_grid_3d(size)
    elif num_spatial_dim == 2:
        rho = radial_coordinate_grid_2d(size)
    else:
        raise ValueError("invalid # of spatial dimensions, must be 2 or 3")
    vals = (1 / jnp.power(4 * jnp.pi * s_rho, 1.5)) * jnp.exp(
        -rho / (4 * s_rho)
    )
    return vals / jnp.amax(vals)


def _coeff_a_0l(ell: int, s0: float) -> Float[Array, ""]:
    """Coefficient for spherical harmonics in wavelet.

    See Eqn. 58 of [2].

    Args:
        ell (int): order of accompanying spherical harmonic
        s0 (float): controls tradeoff between more uniform reconstruction at the cost of less directionality.

    Returns:
        float: the coefficient value
    """
    return jnp.sqrt((2 * ell + 1) / (4 * jnp.pi)) * jnp.exp(
        -ell * (ell + 1) * s0
    )


def _coeff_c_0l(ell: int, s0: float) -> Float[Array, ""]:
    """Coefficient for 3D cake wavelet.

    See Eqn. 63 of [2].

    Args:
        ell (int): order of accompanying spherical harmonic
        s0 (float): controls tradeoff between more uniform reconstruction at the cost of less directionality.

    Returns:
        float: coefficient value
    """
    # legendre_l0 = legendre_scipy(ell, 0.)
    legendre_l0 = eval_legendre(ell, 0)  # type: ignore
    a_l0 = _coeff_a_0l(ell, s0)
    # return (legendre_l0 + (1 - jnp.pow(-1, ell)) / 2.0) * a_l0
    return legendre_l0 * a_l0 + (1 - (jnp.pow(-1, ell)) / 2.0) * a_l0


def cake_wavelet_3d_fourier(
    size: int,
    gamma_window: float,
    nyquist_freq: float,
    sigma_erf: float,
    s_0: float,
    s_rho: float | None,
    alpha: float,
    beta: float,
    gamma: float,
    big_ell: int,
    centered: bool = False,
) -> Complex[Array, "{size} {size} {size}"]:
    """3D cake wavelet, in the Fourier domain.

    Args:
        size (int): size of output wavelet (will be size^3)
        gamma_window (float): gamma parameter for the radial window, fixes the inflection point for the rolloff at `gamma*nyquist_freq`.
        nyquist_freq (float): nyquist frequency of the data (same units as size)
        sigma_erf (float): controls the steepness of the decay around the Nyquist frequency.
        s_0 (float): controls tradeoff between more uniform reconstruction at the cost of less directionality.
        s_rho (float): low frequency window variance.
        alpha (float): rotation around x-axis
        beta (float): rotation around y-axis
        gamma (float): rotation around z-axis
        big_ell (int): order to compute the spherical harmonics up to. In [2], this is "L".
        centered (bool, optional): whether the wavelet is centered in Fourier space (as if it were `fftshift`'d). Defaults to False.

    Returns:
        Complex[Array]: a single cake wavelet, in the Fourier domain.
    """
    g_rho = _radial_window_fft_3d(size, gamma_window, nyquist_freq, sigma_erf)
    theta, phi = angular_coordinate_grids_3d(size, alpha, beta, gamma)
    c_0l = jnp.expand_dims(
        jnp.asarray([_coeff_c_0l(ell, s_0) for ell in range(0, big_ell + 1)]),
        axis=0,
    )
    sph_harm = Partial(
        jax.scipy.special.sph_harm_y,
        jnp.array([0]),
        jnp.arange(0, big_ell + 1, 1),
    )
    sigma_c0l_times_y0l = jnp.sum(
        c_0l * jax.vmap(sph_harm, (0, 0))(theta.flatten(), phi.flatten()),
        axis=-1,
    ).reshape(phi.shape)
    wavelet = g_rho * sigma_c0l_times_y0l
    if s_rho is not None:
        _, wavelet = split_cake_wavelet_fourier(wavelet, s_rho)
    if centered:
        return wavelet
    else:
        return jnp.roll(
            wavelet,
            (-size // 2, -size // 2, -size // 2),
            axis=(-3, -2, -1),
        )


def _coeff_s_alpha_nl(
    ell: int,
    p: int,
    alpha: int,
    q: float,
) -> Float:
    n = ell + 2 * p
    return jax.lax.select(
        q == 0,
        (
            jnp.sqrt(jnp.pi)
            * jax.scipy.special.gamma(1 + alpha)
            / (4 * jax.scipy.special.gamma(2.5 + alpha))
        )
        * jnp.ones_like(q),
        (jnp.pow(2, alpha) * jnp.pow(-1, p) * (p + 1))
        * (spherical_jn(n + alpha + 1, q) / jnp.pow(q, alpha + 1)),
    )


def _coeff_c0_nl(ell: int, p: int, alpha: int, s0: float) -> Float:
    legendre_l0 = eval_legendre(ell, 0)  # type: ignore
    paren_term = legendre_l0 + (1 - (-1) ** ell) / 2
    return paren_term * _coeff_a_0l(ell, s0) * _coeff_b_eqn101(ell, p, alpha)


def _coeff_b_eqn101(
    ell: int,
    p: int,
    alpha: int,
) -> Float:
    return jnp.sum(
        jnp.asarray(
            [_coeff_b_eqn101_singleterm(ell, p, alpha, i) for i in range(3)]
        )
    )


def _coeff_b_eqn101_singleterm(
    ell: int,
    p: int,
    alpha: int,
    i: int,
    beta: int = 2,
) -> Float:
    rho_max = jnp.sqrt((0.5 * beta) / (alpha + 0.5 * beta))
    if i == 0:
        c = 1 + jnp.pow(alpha + 1, 3) / (2 * alpha) * jnp.pow(rho_max, 4)
    elif i == 1:
        c = -2 * jnp.pow(alpha + 1, 3) / (2 * alpha) * jnp.square(rho_max)
    elif i == 2:
        c = jnp.pow(alpha + 1, 3) / (2 * alpha)
    else:
        raise ValueError("only calculated up for values i = 0,1,2")
    b_numer = (beta - ell) / binomial_coefficient(2, p)
    b_denom = (2 * alpha + beta + ell + 2 * p + 3) * binomial_coefficient(
        0.5 * (beta + ell + 1) + alpha + p, alpha + p
    )
    b = b_numer / b_denom
    return c * b


def cake_wavelet_3d_real(
    size: int,
    alpha_super: int,
    nyquist_freq: float,
    s_0: float,
    alpha: float,
    beta: float,
    gamma: float,
    big_ell: int,
    big_p: int = 21,
) -> Complex[Array, "{size} {size} {size}"]:
    """Create a 3D cake wavelet kernel.

    Args:
        size (int): size of output wavelet (will be size^3)
        alpha_super (int):
        nyquist_freq (float): nyquist frequency of the data
        s_0 (float): controls tradeoff between more uniform reconstruction at the cost of less directionality.
        alpha (float): rotation around x-axis
        beta (float): rotation around y-axis
        gamma (float): rotation around z-axis
        big_ell (int): order to compute the spherical harmonics up to.
        big_p (int): order to compute sum up to (l - n = 2p).

    Returns:
        Complex[Array, "{size} {size} {size}"]
    """
    # formulate grid to compute on in spherical coordinates (r, theta, phi)
    x, y, z = jnp.meshgrid(*([jnp.linspace(-size / 2.0, size / 2.0, size)] * 3))
    r = jnp.sqrt(jnp.square(x) + jnp.square(y) + jnp.square(z))
    theta, phi = angular_coordinate_grids_3d(size, alpha, beta, gamma)

    # function to generate a single term
    # see Eqn. 105 of [1]
    def _single_term(r: Float, theta: Float, phi: Float, _ell: int, _p: int):
        c0_nl = _coeff_c0_nl(_ell, _p, alpha_super, s_0)
        sa_nl = _coeff_s_alpha_nl(
            _ell, _p, alpha_super, 2 * jnp.pi * r * nyquist_freq
        )
        sph_harm = jax.vmap(
            Partial(
                jax.scipy.special.sph_harm_y, jnp.array([0]), jnp.array([_ell])
            ),
            (0, 0),
            0,
        )(theta.flatten(), phi.flatten()).reshape(phi.shape)
        oth_term = 4 * jnp.pi * jnp.pow(jax.lax.complex(0.0, 1.0), _ell)
        out = c0_nl * oth_term * sa_nl * sph_harm
        return out

    map_fun = Partial(_single_term, r, theta, phi)
    components = []
    for p in range(big_p):
        for ell in range(big_ell):
            components.append(map_fun(ell, p))
    components = jnp.stack(components, axis=0)
    return jnp.nansum(components, axis=0)


def split_cake_wavelet_fourier(
    wavelet: Complex[Array, "..."], s_rho: float
) -> Tuple[Complex[Array, "..."], Complex[Array, "..."]]:
    """Split the wavelet into high/low frequency components.

    See Sect. 2.1.1 of [2], this implements Eqns 18 with Gaussian window specified by Eqn. 19.

    Returns:
        Tuple[Complex[Array],Complex[Array]]: (low, high) frequency parts of input wavelet.
    """
    size = wavelet.shape[0]
    num_spatial_dim = len(wavelet.shape)
    lf_win = _low_frequency_gaussian_window(size, s_rho, num_spatial_dim)
    return lf_win * wavelet, (1 - lf_win) * wavelet
