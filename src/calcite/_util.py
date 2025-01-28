""" Private utilities. For internal library use only.

To expose a utility function via the public API, import it in `util.py`
"""

import jax
import jax.numpy as jnp
import jax.scipy as jsp
from jax.tree_util import Partial
from jaxtyping import Array
from jaxtyping import Complex
from jaxtyping import Float
from jaxtyping import Real


def binomial_coefficient(
    x: Real[Array, "..."] | Real, y: Real[Array, "..."] | Real
) -> Float[Array, "..."]:
    """binomial_coefficient binomial coefficient as a function of 2 variables.

    Returns:
        Real[Array,"..."]|Real: value of binomial coefficient
    """
    return jsp.special.gamma(x + 1) / (
        jsp.special.gamma(y + 1) * jsp.special.gamma(x - y + 1)
    )


def ifft2_centered(
    centered_fft: Complex[Array, "a a"]
) -> Complex[Array, "a a"]:
    """ifft2_centered inverse fft of a centered fft.

    Args:
        centered_fft (Complex[Array]): fourier-domain image which has been shifted so that 0 frequency is in the middle.

    Returns:
        Complex[Array]
    """
    cent = centered_fft.shape[0] // 2
    # move 0 freq to top left, then take the ifft and retranslate to middle
    fft = jnp.roll(centered_fft, (-cent, -cent), (0, 1))
    im = jnp.fft.ifft2(fft)
    return jnp.roll(im, (cent, cent), (0, 1))


# polar coordinate grid generation
def radial_coordinate_grid_2d(size: int) -> Float[Array, "{size} {size}"]:
    """radial_coordinate_grid_2d 2d grid of radial coordinates for x,y.

    r = sqrt(x^2 + y^2)

    Args:
        size (int): size of single dimension of grid

    Returns:
        Float[Array]: 2d grid of radial coordinate values
    """
    cent = size / 2
    x, y = jnp.meshgrid(jnp.arange(size), jnp.arange(size))
    return jnp.sqrt(jnp.power(x - cent, 2) + jnp.power(y - cent, 2))


def angular_coordinate_grid_2d(size: int) -> Float[Array, "{size} {size}"]:
    """angular_coordinate_grid_2d 2d grid of angles for each x,y in grid.

    a = atan2(y/x) where y and x are based on the grid center being (0,0)

    Args:
        size (int): size of single dimension of grid

    Returns:
        Float[Array]: 2d square grid of angular values at each coordinate
    """
    cent = size / 2
    x, y = jnp.meshgrid(jnp.arange(size), jnp.arange(size))
    return jnp.arctan2(y - cent, x - cent)


def shift_remainder(v: Array) -> Array:
    """shift_remainder shift negative angles to equiv. positive values in interval [0, 2*pi].

    Args:
        v (Float[Array]): input values (angles)

    Returns:
        Float[Array]
    """
    return jnp.remainder(v + jnp.pi, 2 * jnp.pi) - jnp.pi


def polarize2d(
    wvlet: Array, positive: bool, y_axis: bool = True, centered: bool = False
) -> Array:
    """polarize2d make input, double-sided fourier domain wavelet into single-sided version.

    Args:
        x (Array): input wavelet, should be fourier domain
        positive (bool): whether to take the positive or negative side (positive if True, negative if False)
        y_axis (bool, optional): split along y-axis. Defaults to True.
        centered (bool, optional): whether or not input wavelet is centered in fourier domain. Defaults to False.

    Returns:
        Array: single-sided version of wavelet
    """
    size_h, size_w = wvlet.shape[-2], wvlet.shape[-1]
    shift = (-size_h / 2, -size_w / 2)
    if y_axis:
        x, _ = jnp.meshgrid(
            jnp.arange(0, size_w) - size_w / 2,
            jnp.arange(0, size_h) - size_h / 2,
            indexing="xy",
        )
    else:
        _, x = jnp.meshgrid(
            jnp.arange(0, size_w) - size_w / 2,
            jnp.arange(0, size_h) - size_h / 2,
            indexing="xy",
        )
    if centered:
        w_x = 0.5 * jax.lax.erf(x) + 0.5
    else:
        w_x = jnp.roll((0.5 * jax.lax.erf(x) + 0.5), shift, axis=(-2, -1))
    if positive:
        return w_x * wvlet
    else:
        return (1 - w_x) * wvlet


def polarize_filter_bank_2d(
    bank: Array, y_axis: bool = True, centered: bool = False
) -> Array:
    """polarize_filter_bank_2d polarize all of the filters in the filter bank into positive and negative versions.

    Polarization as is done here is described in [1].

    [1]: Bekkers, Erik, et al. "A multi-orientation analysis approach to retinal vessel tracking." Journal of Mathematical Imaging and Vision 49 (2014): 583-610.

    Args:
        bank (Array): input filter bank of double-sided filters
        y_axis (bool, optional): polarize along y-axis in fourier domain (breaks pi, 2pi symmetry). Defaults to True.
        centered (bool, optional): filters in input bank are centered in fourier space. Defaults to False.

    Returns:
        Array: input bank, but with single-sided filters. output will have extra dimension of size 2 in last "filter dimension" corresponding to positive & negative versions of filter. for example, if input was [N x M x size_h x size_w], output will be [N x M x 2 x size_h x size_w]
    """
    inpt_shape = bank.shape
    size_h, size_w = inpt_shape[-2], inpt_shape[-1]
    polarize = Partial(polarize2d, y_axis=y_axis, centered=centered)
    pos_fun = jax.vmap(Partial(polarize, positive=True), 0, 0)
    neg_fun = jax.vmap(Partial(polarize, positive=False), 0, 0)
    return jnp.stack(
        [
            pos_fun(bank.reshape(-1, size_h, size_w)).reshape(inpt_shape),
            neg_fun(bank.reshape(-1, size_h, size_w)).reshape(inpt_shape),
        ],
        axis=-3,
    )
