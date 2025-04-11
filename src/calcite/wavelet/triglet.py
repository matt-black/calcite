"""Triglet wavelets.

Introduced in [1].

References
---
[1] Saydjari, Andrew K., and Douglas P. Finkbeiner. "Equivariant wavelets: Fast rotation and translation invariant wavelet scattering transforms." IEEE Transactions on Pattern Analysis and Machine Intelligence 45.2 (2022): 1716-1731.
"""

from functools import partial

import jax.numpy as jnp
from jaxtyping import Array
from jaxtyping import Float

from .._util import angular_coordinate_grid_2d
from .._util import radial_coordinate_grid_2d


def filter_bank_2d(
    size: int, j_im: int, ell: int, w: int = 2, t: int = 1
) -> Float[Array, " {j_im-2} {ell} {size} {size}"]:
    """filter_bank triglet filter bank.

    Args:
        size (int): size of filter in single dimension (output will be size x size)
        j_im (int): log_2(size of one dimension of image)
        ell (int): number of angular bins
        w (int): angular width in multiples of pi/ell.
        t (int): whether to subdivide Fourier half (t=1) or full (t=2) plane.

    Returns:
        Float[Array]
    """
    return jnp.stack(
        [
            filter_bank_2d_fixed_scale(size, j_im, ell, j, w, t)
            for j in range(1, j_im - 1)
        ],
        axis=0,
    )


def filter_bank_2d_fixed_scale(
    size: int, j_im: int, ell: int, j: int, w: int = 2, t: int = 1
) -> Float[Array, " {ell} {size} {size}"]:
    """filter_bank_2d_fixed_scale triglet filter bank at fixed scale, `j`.

    Args:
        size (int): size of filter in single dimension (output will be size x size)
        j_im (int): log_2(size of one dimension of image)
        ell (int): number of angular bins
        j (int): radial bin index
        w (int): angular width in multiples of pi/ell.
        t (int): whether to subdivide Fourier half (t=1) or full (t=2) plane.

    Returns:
        Float[Array]
    """
    return jnp.stack(
        [triglet_2d(size, j, a_bin, j_im, ell, w, t) for a_bin in range(ell)],
        axis=0,
    )


def triglet_2d(
    size: int, j: int, a_bin: int, j_im: int, ell: int, w: int = 2, t: int = 1
) -> Float[Array, " {size} {size}"]:
    """triglet_2d 2D triglet filter.

    Args:
        size (int): size of filter in single dimension (output will be size x size)
        j (int): radial bin index
        a_bin (int): angular bin index
        j_im (int): log_2(size of one dimension of image)
        ell (int): number of angular bins
        w (int): angular width in multiples of pi/ell.
        t (int): whether to subdivide Fourier half (t=1) or full (t=2) plane.

    Returns:
        Float[Array]
    """
    r = radial_coordinate_grid_2d(size)
    theta = angular_coordinate_grid_2d(size)
    psihat = partial(psihat_jl, j=j, l=a_bin, j_im=j_im, ell=ell, w=w, t=t)
    return psihat(r, theta)


def psihat_jl(
    r: float,
    theta: float,
    j: int,
    l: int,
    j_im: int,
    ell: int,
    w: int = 2,
    t: int = 1,
) -> float:
    r"""psihat_jl value of triglet at (r, \theta).

    Args:
        r (float): Fourier-space radial coordinate
        theta (float): Fourier-space angular coordinate
        j (int): radial bin index
        l (int): angular bin index
        j_im (int): log_2(size of one dimension of image)
        ell (int): number of angular bins
        w (int): angular width in multiples of pi/ell.
        t (int): whether to subdivide Fourier half (t=1) or full (t=2) plane.

    Raises:
        ValueError: if t doesn't equal 1 or 2.

    Returns:
        float
    """
    if not (t == 1 or t == 2):
        raise ValueError("t must be 1 or 2")
    arg1 = 0.5 * jnp.pi * (jnp.log2(r) - (j_im - j - 1))
    ind1 = jnp.logical_and(arg1 >= -jnp.pi / 2, arg1 <= jnp.pi / 2)
    arg2 = (ell / (2 * w * t)) * (theta - l * t * jnp.pi / ell)
    ind2 = jnp.logical_and(arg2 >= -jnp.pi / 2, arg2 <= jnp.pi / 2)
    ind = jnp.logical_and(ind1, ind2).astype(arg1.dtype)
    return 1 / jnp.sqrt(w) * jnp.cos(arg1) * jnp.cos(arg2) * ind
