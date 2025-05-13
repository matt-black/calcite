"""Utility functions for scattering transforms.

These aren't useful for the transform, itself, but for dealing with outputs, misc. data handling, etc.
"""

from typing import List

import jax.numpy as jnp
from jaxtyping import Array

from ..periodize import periodize_filter


__all__ = [
    "apply_filter_bank",
    "subsample_field",
    "flatten_filter_tensor",
    "group_fields_by_size",
]


def apply_filter_bank(
    num_spatial_dims: int,
    x: Array,  # [channels x (spatial dimensions)]
    psi: Array,  # [filter dimensions x (spatial dimensions)]
    adicity: int,
    scale_subsample: int,
    scale_periodize: int | None = None,
) -> Array:
    """Apply a bank of filters to all of the channels in the input.

    Args:
        num_spatial_dims (int): number of spatial dimensions in the input array
        x (Array): input array
        scale_subsample (int): scale to subsample the input by
        scale_periodize (int | None, optional): scale to periodize the filters by. Defaults to None. If None, will be the same as `scale_subsample`.

    Returns:
        Array: output will be shaped [channels x (filter dimensions) x (spatial dimensions)]
    """
    if scale_periodize is None:
        scale_periodize = scale_subsample
    if len(psi.shape) == num_spatial_dims:
        psi = jnp.expand_dims(psi, 0)
    # periodize filters appropriately, then flatten so that output goes from
    # [(filter variant dims) x (spatial dims)] -> [(concat'd filter variants) x (spatial dims)]
    filt_dim_shapes = list(psi.shape[:-num_spatial_dims])
    filt = flatten_filter_tensor(
        num_spatial_dims,
        periodize_filter(num_spatial_dims, psi, adicity, scale_periodize),
    )
    # subsample the field, as appropriate, shape is still [channels x (spatial dims)]
    field = subsample_field(num_spatial_dims, x, adicity, scale_subsample)
    out = jnp.expand_dims(filt, 0) * jnp.expand_dims(field, 1)
    out_space_dims = list(out.shape[-num_spatial_dims:])
    out_shape = (
        list(x.shape[:-num_spatial_dims]) + filt_dim_shapes + out_space_dims
    )
    return jnp.reshape(out, shape=out_shape)


def subsample_field(
    num_spatial_dims: int,
    x: Array,
    adicity: int,
    j: int,
) -> Array:
    """Subsample an input field by local averaging.

    Args:
        num_spatial_dims (int): number of spatial dimensions in input.
        x (Array): input array
        adicity (int): adicity of scale separation
        j (int): scale to subsample at (actual will be adicity**j)

    Returns:
        Array: subsampled input
    """
    if j == 0:
        return x
    k = adicity**j
    if len(x.shape) == num_spatial_dims:
        reshape_dims = list(range(num_spatial_dims))
        leading_shapes = list()
        mean_dims = list(range(0, num_spatial_dims + 1, 2))
    else:
        reshape_dims = [-i for i in reversed(range(1, num_spatial_dims + 1))]
        leading_shapes = list(x.shape[:-num_spatial_dims])
        mean_dims = [r * 2 for r in reshape_dims]
    subsamp_shapes = [
        x for xs in [[k, x.shape[i] // k] for i in reshape_dims] for x in xs
    ]
    reshape = leading_shapes + subsamp_shapes
    return jnp.mean(jnp.reshape(x, reshape), axis=tuple(mean_dims))


def flatten_filter_tensor(
    num_spatial_dims: int,
    psi: Array,
) -> Array:
    """Flatten filter parameter channels (along with possible image channels) into a single array axis.

    Args:
        num_spatial_dims (int): number of spatial dimensions in filter
        psi (Array): filter

    Returns:
        Array
    """
    reshape = [psi.shape[-i] for i in reversed(range(1, num_spatial_dims + 1))]
    reshape.insert(0, -1)
    return jnp.reshape(psi, reshape)


def group_fields_by_size(
    num_spatial_dims: int,
    field1: List[Array],
    field2: List[List[Array]],
) -> List[Array]:
    """Group scattering fields from different layers by size.

    Args:
        field1 (List[Array]): Scattering fields from 1st layer of transform.
        field2 (List[List[Array]]): Scattering fields from 2nd layer of transform.

    Returns:
        List[Array]: list of 6D arrays (CLMHW) where elements correspond to the same spatial size, sorted in descending order.
    """
    num_filt_dims = len(field1.shape[0]) - (num_spatial_dims + 1)
    field1 = [
        jnp.expand_dims(f, axis=range(num_filt_dims + 1, num_filt_dims + 3))
        for f in field1
    ]
    shapes = list(map(lambda f: f.shape[-1], field1))
    idxs = [list() for _ in range(len(field2))]
    for j1 in range(len(field2)):
        for j2 in range(len(field2[j1])):
            shape = field2[j1][j2].shape[-1]
            idx = jnp.argwhere(jnp.asarray(shapes == shape))
            if len(idx):
                idxs[j1].append(int(idx[0][0]))
            else:
                idxs[j1].append(-1)
    for f2i, idxl in enumerate(idxs):
        for f2i2, idx in enumerate(idxl):
            if idx > 0:
                field1[idx] = jnp.concatenate(
                    [field1[idx], field2[f2i][f2i2]], axis=3
                )
    return field1
