"""Private utilities. For internal library use only.

To expose a utility function via the public API, import it in `util.py`
"""

from numbers import Number
from typing import Tuple

import jax
import jax.numpy as jnp
import jax.scipy as jsp
from jax.scipy.spatial.transform import Rotation
from jaxtyping import Array
from jaxtyping import Complex
from jaxtyping import Float
from jaxtyping import Int
from jaxtyping import Real


def binomial_coefficient(x: Array | Real, y: Array | Real) -> Array:
    """Binomial coefficient as a function of 2 variables.

    Returns:
        Real[Array,"..."]|Real: value of binomial coefficient
    """
    return jsp.special.gamma(x + 1) / (
        jsp.special.gamma(y + 1) * jsp.special.gamma(x - y + 1)
    )


def ifft_centered(centered_fft: Complex[Array, "..."]) -> Complex[Array, "..."]:
    """Inverse fft of a centered fft, using jnp.roll.

    Raises:
        ValueError: if input is not a 2D or 3D array

    Returns:
        Complex[Array]
    """
    n_dims = len(centered_fft.shape)
    if n_dims == 2:
        return ifft2_centered(centered_fft)
    elif n_dims == 3:
        return ifft3_centered(centered_fft)
    else:
        raise ValueError("invalid # of dimensions for input array")


def ifft2_centered(
    centered_fft: Complex[Array, "a a"],
) -> Complex[Array, "a a"]:
    """Inverse fft of a centered fft.

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


def ifft3_centered(
    centered_fft: Complex[Array, "a a a"],
) -> Complex[Array, "a a a"]:
    """Take the 3D inverse FFT of a centered FFT by rolling.

    Args:
        centered_fft (Complex[Array]): a 3D FFT'd signal, that has been centered.

    Returns:
        Complex[Array]
    """
    cent = centered_fft.shape[0] // 2
    fft = jnp.roll(centered_fft, (-cent, -cent, -cent), (0, 1, 2))
    im = jnp.fft.ifftn(fft)
    return jnp.roll(im, (cent, cent, cent), (0, 1, 2))


def uncenter_fft(
    centered_fft: Complex[Array, "a a a"] | Complex[Array, "a a"],
) -> Complex[Array, "a a a"] | Complex[Array, "a a"]:
    """Take a center-shifted FFT and unshift it.

    Args:
        centered_fft (Complex[Array]): an FFT'd signal that has been centered

    Returns:
        Complex[Array]
    """
    cent = centered_fft.shape[0] // 2
    n_dim = len(centered_fft.shape)
    if n_dim == 3:
        return jnp.roll(centered_fft, (-cent, -cent, -cent), (0, 1, 2))
    elif n_dim == 2:
        return jnp.roll(centered_fft, (-cent, -cent), (0, 1))
    else:
        raise ValueError("invalid # of dimensions, input array must be 2 or 3d")


# polar coordinate grid generation
def radial_coordinate_grid_2d(size: int) -> Float[Array, "{size} {size}"]:
    """2d grid of radial coordinates for x,y.

    r = sqrt(x^2 + y^2)

    Args:
        size (int): size of single dimension of grid

    Returns:
        Float[Array]: 2d grid of radial coordinate values
    """
    cent = size / 2.0
    x, y = jnp.meshgrid(jnp.arange(size), jnp.arange(size))
    return jnp.sqrt(jnp.power(x - cent, 2) + jnp.power(y - cent, 2))


def radial_coordinate_grid_3d(size: int) -> Float[Array, "{size} {size}"]:
    """3d grid of radial coordinates for x, y, z.

    Returns:
        Float[Array]: 3d grid of radial coordinate values
    """
    cent = size / 2.0
    x, y, z = jnp.meshgrid(
        jnp.arange(size), jnp.arange(size), jnp.arange(size), indexing="xy"
    )
    return jnp.sqrt(
        jnp.power(x - cent, 2) + jnp.power(y - cent, 2) + jnp.power(z - cent, 2)
    )


def angular_coordinate_grid_2d(size: int) -> Float[Array, "{size} {size}"]:
    """2d grid of angles for each x,y in grid.

    a = atan2(y/x) where y and x are based on the grid center being (0,0)

    Args:
        size (int): size of single dimension of grid

    Returns:
        Float[Array]: 2d square grid of angular values at each coordinate
    """
    cent = size / 2.0
    x, y = jnp.meshgrid(jnp.arange(size), jnp.arange(size), indexing="xy")
    return jnp.fliplr(jnp.arctan2(y - cent, x - cent)) + jnp.pi


def angular_coordinate_grids_3d(
    size: int, alpha: float = 0, beta: float = 0.0, gamma: float = 0.0
) -> Tuple[Float[Array, "{size} {size}"], Float[Array, "{size} {size}"]]:
    """Grids for (theta, phi) angle at each point in grid.

    theta is the angle between the x and y axes (`tan^(-1)(y/x)`)
    phi is the angle relative to the z-axis

    Returns:
        (Float[Array],Float[Array]): (theta, phi) arrays
    """
    x, y, z = jnp.meshgrid(*([jnp.linspace(-size / 2.0, size / 2.0, size)] * 3))
    c = jnp.stack([x, y, z], axis=-1)
    # do the rotation
    if alpha != 0 or beta != 0 or gamma != 0:
        rot = Rotation.from_euler(
            "zyx", jnp.asarray([gamma, beta, alpha]), degrees=False
        )
        c = jnp.einsum("ij,...j->...i", rot.as_matrix(), c)
    rho = jnp.sqrt(jnp.sum(jnp.square(c), axis=-1))
    theta = jnp.arctan2(c[..., 1], c[..., 0])
    phi = jnp.where(rho > 0, jnp.arccos(c[..., 2] / rho), 0)
    return theta, phi


def shift_remainder(v: Array) -> Array:
    """Shift negative angles to equiv. positive values in interval [0, 2*pi].

    Args:
        v (Float[Array]): input values (angles)

    Returns:
        Float[Array]
    """
    return (v + 2 * jnp.pi) % (2 * jnp.pi)


def polarize2d(
    wvlet: Array, positive: bool, y_axis: bool = True, centered: bool = False
) -> Array:
    """Make input, double-sided fourier domain wavelet into single-sided version.

    Args:
        wvlet (Array): input wavelet, should be fourier domain
        positive (bool): whether to take the positive or negative side (positive if True, negative if False)
        y_axis (bool, optional): split along y-axis. Defaults to True.
        centered (bool, optional): whether or not input wavelet is centered in fourier domain. Defaults to False.

    Returns:
        Array: single-sided version of wavelet
    """
    size_h, size_w = wvlet.shape[-2], wvlet.shape[-1]
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
    # if centered:
    w_x = 0.5 * jax.lax.erf(x) + 0.5
    # else:
    #    w_x = jnp.roll((0.5 * jax.lax.erf(x) + 0.5), shift, axis=(-2, -1))
    if positive:
        return w_x * wvlet
    else:
        return (1 - w_x) * wvlet


def polarize_filter_bank_2d(
    bank: Array,
    positive: bool,
    y_axis: bool = True,
    centered: bool = False,
    filter: bool = False,
) -> Array:
    """Polarize all of the filters in the filter bank into positive and negative versions.

    Polarization as is done here is described in [1].

    [1]: Bekkers, Erik, et al. "A multi-orientation analysis approach to retinal vessel tracking." Journal of Mathematical Imaging and Vision 49 (2014): 583-610.

    Args:
        bank (Array): input filter bank of double-sided filters
        y_axis (bool, optional): polarize along y-axis in fourier domain (breaks pi, 2pi symmetry). Defaults to True.
        centered (bool, optional): filters in input bank are centered in fourier space. Defaults to False.

    Returns:
        Array: input bank, but with single-sided filters. output will have extra dimension of size 2 in last "filter dimension" corresponding to positive & negative versions of filter. for example, if input was [N x M x size_h x size_w], output will be [N x M x 2 x size_h x size_w]
    """
    raise NotImplementedError("todo")


def legendre_recurrence(
    n: Int[Array, " n"], x: Float[Array, " m"], n_max: Int[Array, ""]
) -> Float[Array, "n m"]:
    """Compute the Legendre polynomials up to degree n_max at a given point or array of points x.

    The first two Legendre polynomials are initialized as P_0(x) = 1 and P_1(x) = x. The subsequent polynomials are computed using the recurrence relation: P_{n+1}(x) = ((2n + 1) * x * P_n(x) - n * P_{n-1}(x)) / (n + 1).

    Args:
        n_max (int): The highest degree of Legendre polynomial to compute. Must be a non-negative integer.
        x (Array): The point(s) at which the Legendre polynomials are to be evaluated. Can be a single point (float) or an array of points.

    Returns:
        Array: A sequence of Legendre polynomial values of shape (n_max+1,) + x.shape, evaluated at point(s) x. The i-th entry of the output array corresponds to the Legendre polynomial of degree i.

    Notes:
        Implementation taken from: https://github.com/jax-ml/jax/issues/14101
    """
    p_init = jnp.zeros((2,) + x.shape)
    p_init = p_init.at[0].set(1.0)  # Set the 0th degree Legendre polynomial
    p_init = p_init.at[1].set(x)  # Set the 1st degree Legendre polynomial

    def body_fun(carry, _):
        i, (p_im1, p_i) = carry
        p_ip1 = ((2 * i + 1) * x * p_i - i * p_im1) / (i + 1)

        return ((i + 1).astype(int), (p_i, p_ip1)), p_ip1

    (_, (_, _)), p_n = jax.lax.scan(
        f=body_fun,
        init=(1, (p_init[0], p_init[1])),
        xs=(None),
        length=(n_max - 1),  # type: ignore
    )
    p_n = jnp.concatenate((p_init, p_n), axis=0)
    return p_n[n]


def eval_legendre(
    n: Int[Array, " n"], x: Float[Array, " m"]
) -> Float[Array, "n m"]:
    """Evaluate Legendre polynomials of specified degrees at provided point(s).

    Parameters:
        n (Array): An array of integer degrees for which the Legendre polynomials are to be evaluated. Each element must be a non-negative integer and the array can be of any shape.
        x (Array): The point(s) at which the Legendre polynomials are to be evaluated. Can be a single point (float) or an array of points. The shape must be broadcastable to the shape of 'n'.

    Returns:
        Array: An array of Legendre polynomial values. The output has the same shape as 'n' and 'x' after broadcasting. The i-th entry corresponds to the Legendre polynomial of degree 'n[i]' evaluated at point 'x[i]'.

    Notes:
        Implementation taken from: https://github.com/jax-ml/jax/issues/14101
    """
    if n == 0:
        return jnp.array(1)
    n = jnp.asarray([n]) if isinstance(n, int) else jnp.asarray(n)
    x = jnp.asarray([x]) if isinstance(x, Number) else jnp.asarray(x)
    n_max = n.max()

    if n.ndim == 1 and x.ndim == 1:
        p = jax.vmap(
            lambda ni: jax.vmap(lambda xi: legendre_recurrence(ni, xi, n_max))(
                x
            )
        )(n)
        p = jnp.diagonal(
            p
        )  # get diagonal elements to match the scipy.special.eval_legendre output
    else:
        p = jax.vmap(
            lambda ni: jax.vmap(lambda xi: legendre_recurrence(ni, xi, n_max))(
                x
            )
        )(n)
    return jnp.squeeze(p)
