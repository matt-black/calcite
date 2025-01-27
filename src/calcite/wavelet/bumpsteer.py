"""Bump steerable wavelets.

Commonly used in phase harmonic scattering networks. Introduced in [1].
Implementation used here is based on the pyWPH implementation, see [2].

References
---
[1] Mallat, Stéphane, Sixin Zhang, and Gaspar Rochette. "Phase harmonic correlations and convolutional neural networks." Information and Inference: A Journal of the IMA 9.3 (2020): 721-747.
[2] Regaldo-Saint Blancard, B., Allys, E., Boulanger, F., Levrier, F., & Jeffrey, N. (2021). A new approach for the statistical denoising of Planck interstellar dust polarization data. arXiv:2102.03160
"""

import math

import jax
import jax.numpy as jnp
from jaxtyping import Array
from jaxtyping import Num

from ..periodize import periodize_filter


def filter_bank_2d(
    size_h: int,
    size_w: int,
    n_scales: int,
    n_orientations: int,
    n_alphas: int,
    delta_n: int,
    space: str,
    adicity: int = 2,
    sigma_prefactor: float = 1.0,
    freq_prefactor: float = 0.85 * math.pi,
) -> Num[
    Array, "{n_scales} {n_orientations} {n_alphas} {delta_n} {size_h} {size_w}"
]:
    """filter_bank_2d create a filter bank of 2D bump-steerable kernels.

    Args:
        size_h (int): size of output filters, in rows
        size_w (int): size of output filters, in cols
        n_scales (int): number of scales
        n_orientations (int): number of orientations (must be even)
        n_alphas (int): number of anglular shifts in phase
        delta_n (int): number of spatial shifts in phase
        space (str): output space of filter bank (one of 'real' or 'fourier')
        adicity (int, optional): adicity of scale separation. Defaults to 2.
        sigma_prefactor (float, optional): prefactor for scale-specific sigma parameter. Defaults to 1.0.
        freq_prefactor (float, optional): prefactor for scale-specific central frequency parameter. Defaults to 0.85*math.pi.

    Raises:
        ValueError: if number of orientations is not even
        ValueError: if output space isn't one of ('real', 'fourier')

    Returns:
        Num[Array]: [n_scales x n_orientations x n_alphas x delta_n x size_h x size_w] array of filters
    """
    if n_orientations % 2 == 0:
        raise ValueError("# of orientations must be divisible by 2")
    ell = n_orientations // 2  # upper-case L in Ref. [2]
    sigmas = [sigma_prefactor * adicity**j for j in range(n_scales)]
    freqs = [freq_prefactor / adicity**j for j in range(n_scales)]
    thetas = [jnp.pi * ell / ell for ell in range(n_orientations)]
    alphas = [jnp.pi * ell / ell for ell in range(n_alphas)]
    ns = list(range(delta_n))
    if space == "fourier":
        kernel_fun = bump_steerable_kernel_2d_fourier
    elif space == "real":
        kernel_fun = bump_steerable_kernel_2d_real
    else:
        raise ValueError("invalid output space, must be 'real' or 'fourier'")
    return jnp.expand_dims(
        jnp.stack(
            [
                jnp.stack(
                    [
                        jnp.stack(
                            [
                                jnp.stack(
                                    [
                                        kernel_fun(
                                            size_h,
                                            size_w,
                                            freq,
                                            theta,
                                            sigma,
                                            n,
                                            alpha,
                                            ell,
                                        )
                                        for n in ns
                                    ],
                                    axis=0,
                                )
                                for alpha in alphas
                            ],
                            axis=0,
                        )
                        for theta in thetas
                    ],
                    axis=0,
                )
                for sigma, freq in zip(sigmas, freqs)
            ],
            axis=0,
        ),
        2,
    )


def bump_steerable_kernel_2d_real(
    size_h: int,
    size_w: int,
    xi: float,
    theta: float,
    sigma: float,
    n: float,
    alpha: float,
    ell: int,
) -> Num[Array, "{size_h} {size_w}"]:
    """bump_steerable_kernel_2d_real Real-space kernel for 2D bump steerable wavelet.

    Args:
        size_h (int): spatial size of filter, height, in pixels
        size_w (int): spatial size of filter, width, in pixels
        xi (float): central frequency of filter
        theta (float): angle of filter, in [0, 2*pi]
        sigma (float): bandwidth of filter
        n (float): amount of translation of filter in k-space
        alpha (float): angle of translation of filter, relative to theta
        ell (int): number of filters in bank, in half-plane of angles [0, pi]

    Returns:
        Num[Array, {size_h} {size_w}]
    """
    return jnp.fft.ifft2(
        bump_steerable_kernel_2d_fourier(
            size_h, size_w, xi, theta, sigma, n, alpha, ell
        )
    )


def bump_steerable_kernel_2d_fourier(
    size_h: int,
    size_w: int,
    xi: float,
    theta: float,
    sigma: float,
    n: float,
    alpha: float,
    ell: int,
) -> Num[Array, "{size_h} {size_w}"]:
    """bump_steerable_kernel_2d_fourier Fourier-space kernel for 2D bump steerable wavelet.

    Args:
        size_h (int): spatial size of filter, height, in pixels
        size_w (int): spatial size of filter, width, in pixels
        xi (float): central frequency of filter
        theta (float): angle of filter, in [0, 2*pi]
        sigma (float): bandwidth of filter
        n (float): amount of translation of filter in k-space
        alpha (float): angle of translation of filter, relative to theta
        ell (int): number of filters in bank, in half-plane of angles [0, pi]

    Returns:
        Num[Array, {size_h} {size_w}]
    """
    # generate coordinate grid to compute filter values over
    # NOTE: initial grid is 2x larger than output size, because we'll
    # do the periodization of these wavelets by folding
    k_x, k_y = jnp.meshgrid(
        2 * (2 * jnp.pi) * jnp.fft.fftfreq(2 * size_w),
        2 * (2 * jnp.pi) * jnp.fft.fftfreq(2 * size_h),
        indexing="xy",
    )
    # compute angle and modulus of wavevector at each grid location
    k = jax.lax.complex(k_x, k_y)
    a = jnp.angle(k)
    mod = jnp.abs(k)
    # compute translation amounts
    n_x = n * jnp.cos(theta - alpha)
    n_y = n * jnp.sin(theta - alpha)
    # do the computation... see eqn. (A.1) of [2]
    psi = (
        jnp.exp(jax.lax.complex(0.0, sigma * (k_x * n_x + k_y * n_y)))
        * _modulus_window_mask(mod, xi, ell)
        * _angular_window_mask(a, theta, ell)
    )
    # return the periodized filter
    # if no translation, fourier representation is entirely real
    return periodize_filter(
        2, (psi.real if n_x == 0 and n_y == 0 else psi), 2, 1
    )


def _c(ell: int):
    return (
        1
        / 1.29
        * 2 ** (ell - 1)
        * (
            math.factorial(ell - 1)
            / math.sqrt(ell * math.factorial(2 * (ell - 1)))
        )
    )


def _modulus_window_mask(modulus: Array, xi: float, ell: int):
    c = _c(ell)
    frac = (-jnp.square(modulus - xi)) / (xi**2 - jnp.square(modulus - xi))
    return jnp.where(
        jnp.logical_and(modulus > 0, modulus < 2 * xi), c * jnp.exp(frac), 0
    )


def _angular_window_mask(angles: Array, theta: float, ell: int) -> Array:
    return jnp.where(
        jnp.logical_or(
            (angles - theta) % (2 * jnp.pi) <= jnp.pi / 2,
            (angles - theta) % (2 * jnp.pi) >= 3 * jnp.pi / 2,
        ),
        jnp.power(jnp.cos(angles - theta), ell - 1),
        0,
    )
