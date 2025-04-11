"""Morlet wavelets.

Morlet filters are Gabor filters with a pre-modulation offset. This offset
subtracts a gaussian profile centered at the original -- cancelling out the
zero-frequency component and making the Gabor a bandpass filter.
"""

import math
from typing import Tuple

import jax.numpy as jnp
from jaxtyping import Array
from jaxtyping import Complex
from jaxtyping import Num
from jaxtyping import Real

from .gabor import gabor_kernel_2d_real
from .gabor import gabor_kernel_2d_real_scikit


def filter_bank_2d(
    size_h: int,
    size_w: int,
    n_scales: int,
    n_orientations: int,
    n_phases: int,
    space: str,
    adicity: int = 2,
    sigma_prefactor: float = 0.8,
    freq_prefactor: float = 3 * math.pi / 4,
    gamma_prefactor: float = 4.0,
) -> Num[Array, "{n_scales} {n_orientations} {n_phases} {size_h} {size_w}"]:
    """filter_bank_2d Filter bank of 2D Morlet filters.

    The resulting bank is a 5D tensor (n_scales x n_orientations x n_phases x size_h x size_w).

    Args:
        size_h (int): spatial size of filter, height. Specified in pixels.
        size_w (int): Spatial size of filter, width. Specified in pixels.
        n_scales (int): Number of spatial scales in the filter bank.
        n_orientations (int): Number of orientations in the filter bank.
        n_phases (int): Number of phases in the filter bank.
        space (str): Output space of filters, one of 'real' or 'fourier'.
        adicity (int, optional): Adicity of the spatial scale separation. Defaults to 2.
        sigma_prefactor (float, optional): sigma = (sigma_prefactor) * adicity**scale. Defaults to 0.8.
        freq_prefactor (float, optional): freq = (freq_prefactor) / adicity**scale. Defaults to 3*math.pi/4.
        gamma_prefactor (float, optional): gamma = gamma_prefactor / n_orientations. Defaults to 4.0.

    Raises:
        ValueError: if output space isn't one of 'real' or 'fourier'

    Returns:
        ndarray: real if output space is 'fourier', complex if output space is 'real'.
    """
    sigmas = [sigma_prefactor * adicity**j for j in range(n_scales)]
    freqs = [freq_prefactor / adicity**j for j in range(n_scales)]
    thetas = [
        math.floor(n_orientations - n_orientations / 2 - ell)
        * math.pi
        / n_orientations
        for ell in range(n_orientations)
    ]
    phases = [
        math.floor(n_phases - n_phases / 2 - ell) * math.pi / n_phases
        for ell in range(n_phases)
    ]
    if space == "fourier":
        kernel_fun = morlet_kernel_2d_fourier
    elif space == "real":
        kernel_fun = morlet_kernel_2d_real
    else:
        raise ValueError('invalid output space, must be "real" or "fourier"')
    return jnp.stack(
        [
            jnp.stack(
                [
                    jnp.stack(
                        [
                            kernel_fun(
                                size_h,
                                size_w,
                                sigma,
                                freq,
                                theta,
                                gamma_prefactor / n_orientations,
                                phase,
                            )
                            for phase in phases
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
    )


def morlet_kernel_2d_real(
    size_h: int,
    size_w: int,
    sigma: float,
    frequency: float,
    theta: float,
    gamma: float = 1.0,
    offset: float = 0.0,
    dtype=jnp.complex64,
) -> Complex[Array, "{size_h} {size_w}"]:
    """morlet_kernel_2d Real-space kernel for 2D Morlet filter.

    Args:
        size_h (int): spatial size of filter, height, in pixels.
        size_w (int): spatial size of filter, width, in pixels.
        sigma (float) : bandwidth of filter (linear size of receptive field).
        frequency (float): spatial frequency of the cosine factor
        theta (float): angle of filter, in [0, pi]
        gamma (float, optional): ellipticity of the filter. Defaults to 1.0.
        offset (float, optional): phase offset of the cosine factor. Defaults to 0.0.
        dtype : datatype of output. One of `jnp.complex64` or `jnp.complex128`. Defaults to jnp.complex64.

    Returns:
        Complex[Array]
    """
    gab = gabor_kernel_2d_real(
        size_h, size_w, sigma, frequency, theta, gamma, offset, dtype
    )
    mod = gabor_kernel_2d_real(
        size_h, size_w, sigma, 0.0, theta, gamma, offset, dtype
    )
    ratio = jnp.sum(gab) / jnp.sum(mod)
    return gab - ratio * mod


def morlet_kernel_2d_fourier(
    size_h: int,
    size_w: int,
    sigma: float,
    frequency: float,
    theta: float,
    gamma: float = 1.0,
    offset: float = 0.0,
    dtype=jnp.float32,
) -> Real[Array, "{size_h} {size_w}"]:
    """morlet_kernel_2d Real-space kernel for 2D Morlet filter.

    Args:
        size_h (int): spatial size of filter, height, in pixels.
        size_w (int): spatial size of filter, width, in pixels.
        sigma (float) : bandwidth of filter (linear size of receptive field).
        frequency (float): spatial frequency of the cosine factor
        theta (float): angle of filter, in [0, pi]
        gamma (float, optional): ellipticity of the filter. Defaults to 1.0.
        offset (float, optional): phase offset of the cosine factor. Defaults to 0.0.
        dtype : datatype of output. One of `jnp.complex64` or `jnp.complex128`. Defaults to jnp.complex64.

    Returns:
        Real[Array]
    """
    cdtype = jnp.complex64 if dtype == jnp.float32 else jnp.complex128
    return jnp.real(
        jnp.fft.fft2(
            morlet_kernel_2d_real(
                size_h, size_w, sigma, frequency, theta, gamma, offset, cdtype
            )
        )
    )


def filter_bank_2d_scikit(
    n_stds: float,
    n_orientations: int,
    scale: int = 0,
    space: str = "real",
    adicity: int = 2,
    sigma_prefactor: float = 0.8,
    freq_prefactor: float = 3 * math.pi / 4,
    gamma_prefactor: float = 4.0,
):
    """filter_bank_2d_scikit Filter bank of 2D Morlet kernels, generated via the scikit API.

    Args:
        n_stds (float): size of output kernel, in standard deviations
        n_orientations (int): # of orientations in filter bank
        scale (int, optional): scale at which the filter bank will be generated. Defaults to 0.
        space (str, optional): space of kernels ('real' or 'fourier'). Defaults to 'real'.
        adicity (int, optional): adicity of scale separation being used. Defaults to 2.
        sigma_prefactor (float, optional): sigma = (sigma_prefactor) * adicity**scale. Defaults to 0.8.
        freq_prefactor (float, optional): freq = (freq_prefactor) / adicity**scale. Defaults to 3*math.pi/4.
        gamma_prefactor (float, optional): gamma = gamma_prefactor / n_orientations. Defaults to 4.0.

    Raises:
        NotImplementedError: if `space=='fourier'`. only real-space kernels implemented.

    Returns:
        Complex[Array]: (# orientations) x (kernel_height) x (kernel_width)
    """
    if space == "fourier":
        raise NotImplementedError(
            "scikit-style filters only supported for real space"
        )
    freq = freq_prefactor / adicity**scale
    thetas = [
        math.floor(n_orientations - n_orientations / 2 - ell)
        * math.pi
        / n_orientations
        for ell in range(n_orientations)
    ]
    sigma_x = sigma_prefactor * adicity**scale
    sigma_y = sigma_x * gamma_prefactor / n_orientations
    kshapes = [
        _kernel_shape_for_sigmas(n_stds, sigma_x, sigma_y, theta)
        for theta in thetas
    ]
    ksize_h = max([r for r, _ in kshapes])
    ksize_w = max([c for _, c in kshapes])
    return jnp.stack(
        [
            morlet_kernel_2d_real_scikit(
                freq,
                theta,
                1.0,
                sigma_x,
                sigma_y,
                n_stds,
                0,
                (ksize_h, ksize_w),
            )
            for theta in thetas
        ],
        axis=0,
    )


def _kernel_shape_for_sigmas(
    n_stds: float, sigma_x: float, sigma_y: float, theta: float
) -> Tuple[int, int]:
    ct = math.cos(theta)
    st = math.sin(theta)
    x0 = math.ceil(
        max(abs(n_stds * sigma_x * ct), abs(n_stds * sigma_y * st), 1)
    )
    size_x = len(list(range(-x0, x0 + 1)))
    y0 = math.ceil(
        max(abs(n_stds * sigma_y * ct), abs(n_stds * sigma_x * st), 1)
    )
    size_y = len(list(range(-y0, y0 + 1)))
    return size_y, size_x


def morlet_kernel_2d_real_scikit(
    frequency: float,
    theta: float = 0,
    bandwidth: float = 1.0,
    sigma_x: float | None = None,
    sigma_y: float | None = None,
    n_stds: float = 3,
    offset: float = 0,
    shape: Tuple[int, int] | None = None,
    dtype=jnp.complex64,
) -> Complex[Array, "h w"]:
    """morlet_kernel_2d_real_scikit Real-space 2D Morlet kernel.

    Args:
        frequency (float): Spatial frequency of the harmonic function, in pixels.
        theta (float, optional): Orientation in radians. Defaults to 0..
        bandwidth (float, optional): Filter bandwidth, sets sigma_x,y. Defaults to 1..
        sigma_x (float | None, optional): Standard deviation in x-direction (pre-rotation). Defaults to None.
        sigma_y (float | None, optional): Standard deviation in y-direction (pre-rotation). Defaults to None.
        n_stds (float, optional): Size of the output kernel, in standard deviations. Defaults to 3.
        offset (float, optional): Phase offset of the harmonic function. Defaults to 0.
        shape (Tuple[int,int], optional): Directly specify size of output kernel. Defaults to None.
        dtype (optional): Datatype of output kernel (single or double precision). Defaults to jnp.complex128.

    Raises:
        ValueError: If output dtype is not complex

    Returns:
        Complex[Array]
    """
    gab = gabor_kernel_2d_real_scikit(
        frequency,
        theta,
        bandwidth,
        sigma_x,
        sigma_y,
        n_stds,
        offset,
        shape,
        dtype,
    )
    mod = gabor_kernel_2d_real_scikit(
        0, theta, bandwidth, sigma_x, sigma_y, n_stds, offset, shape, dtype
    )
    ratio = jnp.sum(gab) / jnp.sum(mod)
    return gab - ratio * mod
