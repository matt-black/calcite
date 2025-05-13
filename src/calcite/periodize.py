"""Wavelet (FFT-domain) periodization.

Periodization is accomplished by atrous convolution with a kernel of ones. This makes the implementation GPU-friendly and fast.
"""

import math

import jax
import jax.numpy as jnp
from jax.tree_util import Partial
from jaxtyping import Array
from jaxtyping import Num


__all__ = ["periodize_filter"]


def periodize_filter(
    num_spatial_dims: int,
    x: Array,
    adicity: int,
    j: int,
) -> Array:
    """Periodize the input fourier-domain input filter.

    Args:
        num_spatial_dims (int): number of spatial dimensions in input.
        x (Array): input array
        adicity (int): adicity of scale separation
        j (int): scale to periodize to (actual will be adicity**j)

    Raises:
        ValueError: if spatial dimensions <1 or >3

    Returns:
        Array: periodized filter
    """
    if num_spatial_dims < 1 or num_spatial_dims > 3:
        raise ValueError("invalid number of spatial dims, must be (1,2,3)")
    if len(x.shape) == num_spatial_dims:
        if num_spatial_dims == 1:
            return _periodize1d(x, adicity, j)
        elif num_spatial_dims == 2:
            return _periodize2d(x, adicity, j)
        else:
            return _periodize3d(x, adicity, j)
    else:  # need to map over other dimensions
        k = adicity**j
        inp_shape = x.shape
        rest = x.shape[:-num_spatial_dims]
        out_shape = list(inp_shape)
        for i in range(1, num_spatial_dims + 1):
            out_shape[-i] = out_shape[-i] // k
        reshape_flatchan = [math.prod(rest)] + [
            x.shape[-i] for i in reversed(range(1, num_spatial_dims + 1))
        ]
        if num_spatial_dims == 1:
            fun = Partial(_periodize1d, adicity=adicity, j=j)
        elif num_spatial_dims == 2:
            fun = Partial(_periodize2d, adicity=adicity, j=j)
        else:  # num_spatial_dims == 3
            fun = Partial(_periodize3d, adicity=adicity, j=j)
        out = jnp.reshape(
            jax.vmap(fun, 0)(jnp.reshape(x, reshape_flatchan)), out_shape
        )
        return out


def _periodize1d(
    x: Num[Array, " h"], adicity: int, j: int
) -> Num[Array, " {h//adicity**j}"]:
    k = adicity**j
    sze = x.shape[0]
    mask = jnp.ones(x.shape, jnp.float32)
    len_x = math.floor(sze * (1 - adicity ** (-j)))
    start_x = math.floor(sze * adicity ** (-j - 1))
    mask = mask.at[start_x : start_x + len_x].set(0)
    return jax.lax.conv_general_dilated(
        jnp.expand_dims(jnp.multiply(x, mask), (0, 1)),
        jnp.expand_dims(jnp.ones(k, dtype=x.dtype), (0, 1)),
        window_strides=1,
        padding="valid",
        rhs_dilation=(sze // k),
    )[0, 0, ...]


def _periodize2d(
    x: Num[Array, "h w"], adicity: int, j: int
) -> Num[Array, "{h//adicity**j} {w//adicity**j}"]:
    k = adicity**j
    hgt, wid = x.shape[0], x.shape[1]
    len_x = math.floor(hgt * (1 - adicity ** (-j)))
    start_x = math.floor(hgt * adicity ** (-j - 1))
    len_y = math.floor(wid * (1 - adicity ** (-j)))
    start_y = math.floor(wid * adicity ** (-j - 1))
    # mask out corners, as calculated, above
    mask = jnp.ones(x.shape, jnp.float32)
    mask = mask.at[start_x : start_x + len_x, :].set(0)
    mask = mask.at[:, start_y : start_y + len_y].set(0)
    #
    return jax.lax.conv_general_dilated(
        jnp.expand_dims(jnp.multiply(x, mask), (0, 1)),
        jnp.expand_dims(jnp.ones((k, k), dtype=x.dtype), (0, 1)),
        window_strides=(1, 1),
        padding="valid",
        rhs_dilation=(hgt // k, wid // k),
    )[0, 0, ...]


def _periodize3d(
    x: Num[Array, "d h w"], adicity: int, j: int
) -> Num[Array, "{d//adicity**j} {h//adicity**j} {w//adicity**j}"]:
    k = adicity**j
    dep, hgt, wid = x.shape[0], x.shape[1], x.shape[2]
    len_z = math.floor(dep * (1 - adicity ** (-j)))
    start_z = math.floor(dep * adicity ** (-j - 1))
    len_x = math.floor(hgt * (1 - adicity ** (-j)))
    start_x = math.floor(hgt * adicity ** (-j - 1))
    len_y = math.floor(wid * (1 - adicity ** (-j)))
    start_y = math.floor(wid * adicity ** (-j - 1))
    mask = jnp.ones(x.shape, jnp.float32)
    mask = mask.at[start_z : start_z + len_z, ...].set(0)
    mask = mask.at[:, start_x : start_x + len_x, :].set(0)
    mask = mask.at[..., start_y : start_y + len_y].set(0)

    return jax.lax.conv_general_dilated(
        jnp.expand_dims(jnp.multiply(x, mask), (0, 1)),
        jnp.expand_dims(jnp.ones((k, k, k), dtype=x.dtype), (0, 1)),
        window_strides=(1, 1, 1),
        padding="valid",
        rhs_dilation=(dep // k, hgt // k, wid // k),
    )[0, 0, ...]
