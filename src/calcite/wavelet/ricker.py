"""Ricker Wavelets.

Also known as Laplacian of Gaussian (LoG), "Mexican hat", or Marr wavelets. The ricker wavelet is the negative normalized 2nd derivative of a Gaussian function.

References
---
[1] Sage, et al. "Automatic Tracking of Individual Fluorescence Particles: Application to the Study of Chromosome Dynamics" IEEE Transactions on Image Processing, vol. 14, no. 9, pp. 1372–1383, September 2005.
"""

import jax.numpy as jnp
from jaxtyping import Array
from jaxtyping import Complex
from jaxtyping import Float


__all__ = [
    "ricker_kernel_1d_real",
    "ricker_kernel_2d_real",
    "ricker_kernel_2d_fourier",
    "ricker_kernel_3d_real",
]


def ricker_kernel_1d_real(
    length: int,
    sigma: float,
) -> Float[Array, " {length}"]:
    """One dimensional Ricker kernel.

    Args:
        length: length of the filter.
        sigma: standard deviation.

    Returns:
        Float[Array, "{length}"]: the 1D kernel
    """
    t = jnp.linspace(-length / 2, length / 2, length)
    return (
        (2 / (jnp.sqrt(3 * sigma) * jnp.pow(jnp.pi, 0.25)))
        * (1 - jnp.square(t / sigma))
        * jnp.exp(-(jnp.square(t) / (2 * jnp.square(sigma))))
    )


def ricker_kernel_2d_real(
    size_h: int,
    size_w: int,
    sigma_x: float,
    sigma_y: float,
) -> Float[Array, "{size_h} {size_w}"]:
    """Generate a real-space 2D Ricker kernel.

    Args:
        size_h (int): size of the filter, # rows
        size_w (int): size of the filter, # cols
        sigma_x (float): standard deviation in the x-direction (cols)
        sigma_y (float): standard deviation in the y-direction (rows)

    Returns:
        Float[Array, "{size_h} {size_w}"]
    """
    y, x = jnp.meshgrid(
        jnp.linspace(-size_h / 2, size_h / 2, size_h),
        jnp.linspace(-size_w / 2, size_w / 2, size_w),
        indexing="ij",
    )
    return (
        (1 - 0.5 * (jnp.square(x / sigma_x) + jnp.square(y / sigma_y)))
        * jnp.exp(-((jnp.square(x) + jnp.square(y)) / (2 * sigma_x * sigma_y)))
    ) / (jnp.pi * jnp.square(sigma_x) * jnp.square(sigma_y))


def ricker_kernel_2d_fourier(
    size_h: int,
    size_w: int,
    sigma_x: float,
    sigma_y: float,
) -> Complex[Array, "{size_h} {size_w}"]:
    """Generate a fourier-space 2D Ricker kernel.

    Args:
        size_h (int): size of the filter, # rows
        size_w (int): size of the filter, # cols
        sigma_x (float): standard deviation in the x-direction (cols)
        sigma_y (float): standard deviation in the y-direction (rows)

    Returns:
        Complex[Array, "{size_h} {size_w}"]
    """
    return jnp.fft.fft2(ricker_kernel_2d_real(size_h, size_w, sigma_x, sigma_y))


def ricker_kernel_3d_real(
    size_z: int,
    size_h: int,
    size_w: int,
    sigma_x: float,
    sigma_y: float,
    sigma_z: float,
) -> Float[Array, "{size_z} {size_h} {size_w}"]:
    """Generate a real-space 3D Ricker kernel.

    Formula is taken from [1].

    Args:
        size_z (int): size of the filter in depth
        size_h (int): size of the filter, # rows
        size_w (int): size of the filter, # cols
        sigma_x (float): standard deviation in the x-direction (cols)
        sigma_y (float): standard deviation in the y-direction (rows)
        sigma_z (float): standard deviation in the z-direction (depth)

    Returns:
        Float[Array, "{size_z} {size_h} {size_w}"]
    """
    z, y, x = jnp.meshgrid(
        jnp.linspace(-size_z / 2, size_z / 2, size_z),
        jnp.linspace(-size_h / 2, size_h / 2, size_h),
        jnp.linspace(-size_w / 2, size_w / 2, size_w),
        indexing="ij",
    )
    prefactor = 1 / (
        jnp.float_power(2 * jnp.pi, 3.0 / 2.0) * sigma_x * sigma_y * sigma_z
    )
    exp_term = jnp.exp(
        -jnp.square(x) / (2 * jnp.square(sigma_x))
        - jnp.square(y) / (2 * jnp.square(sigma_y))
        - jnp.square(z) / (2 * jnp.square(sigma_z))
    )
    return (
        prefactor
        * exp_term
        * (
            jnp.square(x) / jnp.power(sigma_x, 4)
            - 1 / jnp.square(sigma_x)
            + jnp.square(y) / jnp.power(sigma_y, 4)
            - 1 / jnp.square(sigma_y)
            + jnp.square(z) / jnp.power(sigma_z, 4)
            - 1 / jnp.square(sigma_z)
        )
    )
