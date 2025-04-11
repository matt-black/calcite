"""Padding utilities.

To avoid artifacts in the convolution, inputs must be appropriately padded.
"""

from typing import Iterable
from typing import List

import jax.numpy as jnp
from jaxtyping import Array
from jaxtyping import Num


def center_pad_2d(
    x: Num[Array, "..."], adicity: int, scale: int
) -> Num[Array, "..."]:
    """center_pad_2d Center-pad the input array to prevent convolution artifacts.

    Args:
        x (Num[Array]): input array, must be at least 2D
        adicity (int): adicity of scale separation
        scale (int): scale to pad for

    Returns:
        Num[Array, "..."]: center-padded image
    """
    h, w = x.shape[-2], x.shape[-1]
    ph = _padded_size(h, adicity, scale)
    pw = _padded_size(w, adicity, scale)
    phl, phr = (ph - h) // 2, (ph - h + 1) // 2
    pad_h = (phl, phr)
    pwl, pwr = (pw - w) // 2, (pw - w + 1) // 2
    pad_w = (pwl, pwr)
    zeros = [(0, 0) for _ in x.shape[:-2]]
    return jnp.pad(x, tuple(zeros + [pad_h, pad_w]), mode="reflect")


def uncenter_pad(
    y: Num[Array, "..."], shape: Iterable[int] | int
) -> Num[Array, "..."]:
    """uncenter_pad Remove center padding of input array.

    Args:
        y (Num[Array]): input array to have padding removed
        shape (Iterable[int]|int): original shape of previously-padded dimensions, before they were padded

    Returns:
        Num[Array]
    """
    pad_shape = reversed([shape[-i] for i in range(len(shape))])
    pad_left = [(ps - os) // 2 for ps, os in zip(pad_shape, shape)]
    pad_right = [(ps - os + 1) // 2 for ps, os in zip(pad_shape, shape)]
    slices = [slice(pl, -pr) for pl, pr in zip(pad_left, pad_right)]
    if len(slices) < len(shape):
        slices = [
            slice(None) for _ in range(len(y.shape) - len(shape))
        ] + slices
    return y[tuple(slices)]


def padded_shape(
    shape: Iterable[int], adicity: int, scale: int
) -> int | List[int]:
    """padded_shape Compute the shape of the padded array.

    Args:
        shape (Iterable[int]): shape of input array in dimensions to pad
        adicity (int): adicity of the scale separation
        scale (int): (max) scale to pad for

    Returns:
        int|List[int]
    """
    if len(shape) == 1:
        return _padded_size(shape[0], adicity, scale)
    else:
        return [_padded_size(s, adicity, scale) for s in shape]


def _padded_size(size: int, adicity: int, scale: int) -> int:
    """_padded_size Compute how much to pad a dimension to avoid border effects/artifacts.

    Args:
        size (int): input size of dimension
        adicity (int): adicity of scale separation (usually 2)
        scale (int): (max) scale to pad for

    Returns:
        int
    """
    return ((size + adicity**scale) // adicity**scale + 1) * adicity**scale
