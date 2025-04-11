r"""Solid Harmonic Wavelets.

First introduced in [1]:
"Solid harmonics are solutions of the Laplace equation, $\Delta{}f=0$."
Wavelets are obtained by localizing the support of a solid harmonic in space
by multiplication with a Gaussian.

In 2D: $\psi_\ell(r,\phi) =
In 3D: TODO

References
---
[1] Eickenberg, Michael, et al. "Solid harmonic wavelet scattering: Predicting quantum molecular energy from invariant descriptors of 3D electronic densities." Advances in Neural Information Processing Systems 30 (2017).
[2] Eickenberg, Michael, et al. "Solid harmonic wavelet scattering for predictions of molecule properties." The Journal of chemical physics 148.24 (2018).
"""

import jax.lax
import jax.numpy as jnp
from jax.scipy.special import sph_harm
from jaxtyping import Array
from jaxtyping import Float

from .._util import angular_coordinate_grid_2d
from .._util import radial_coordinate_grid_2d


def solid_harmonic_2d(
    size: int, space: str, ell: int
) -> Float[Array, " {size} {size}"]:
    """solid_harmonic_2d 2d solid harmonic wavelet filter.

    Args:
        size (int): size of one dimension of (square) filter
        space (str): 'real' or 'fourier', space of filter
        ell (int): polynomial order

    Raises:
        ValueError: if space != 'real' or 'fourier'

    Returns:
        Float[Array]
    """
    r = radial_coordinate_grid_2d(size)
    psi = angular_coordinate_grid_2d(size)
    if space == "real":
        return phi_ell(ell, r, psi)
    elif space == "fourier":
        return phi_hat_ell(ell, r, psi)
    else:
        raise ValueError("invalid space")


def phi_ell(ell: int, r: float, psi: float) -> float:
    """phi_ell 2D solid harmonic wavelet value.

    Args:
        ell (float): polynomial order
        r (float): radial coordinate of polar coordinates
        psi (float): angular coordinate of polar coordinates

    Returns:
        float
    """
    return (
        (1 / jnp.sqrt(4 * jnp.pi**2))
        * jnp.exp(-0.5 * r**2)
        * jnp.power(r, ell)
        * jnp.exp(jax.lax.complex(0, ell * psi))
    )


def phi_hat_ell(ell: int, _lambda: float, alpha: float) -> float:
    """phi_hat_ell fourier space 2D solid harmonic wavelet value.

    Args:
        ell (float): polynomial order
        _lambda (float): frequency
        alpha (float): angle

    Returns:
        float
    """
    return (
        jnp.power(-1j, ell)
        * jnp.exp(-0.5 * _lambda**2)
        * jnp.power(_lambda, ell)
        * jnp.exp(jax.lax.complex(0, ell * alpha))
    )


def phi_ell_m(ell: int, m: int, r: float, theta: float, psi: float) -> float:
    """phi_ell_m real space 3D solid harmonic wavelet value.

    Args:
        ell (int): degree of spherical harmonic
        m (int): order of spherical harmonic
        r (float): radius
        theta (float): azimuthal angle
        psi (float): elevation angle

    Returns:
        float
    """
    return (
        jnp.exp(-0.5 * jnp.square(r))
        / jnp.sqrt(8 * jnp.pi**3)
        * jnp.power(r, ell)
        * sph_harm(ell, m, theta, psi)
    )


def phi_hat_ell_m(
    ell: int, m: int, _lambda: float, alpha: float, beta: float
) -> float:
    """phi_hat_ell_m fourier space 3D solid harmonic wavelet value.

    Args:
        ell (int): degree of spherical harmonic
        m (int): order of spherical harmonic
        _lambda (float): wavelength
        alpha (float): azimuthal angle
        beta (float): elevation angle

    Returns:
        float
    """
    return (
        jnp.power(-1j, ell)
        * jnp.exp(-0.5 * jnp.square(_lambda))
        * jnp.power(_lambda, ell)
        * sph_harm(ell, m, alpha, beta)
    )
