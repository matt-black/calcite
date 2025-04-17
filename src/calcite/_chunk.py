"""Utilities for chunk-wise computation.

These utilities are based on the idea that convolution-based computations often contain edge-based artifacts or are subject to edge effects.
To handle these, chunks can be equipped with "overlap" and "padding" that help to deal with these.
"""

from dataclasses import dataclass
from itertools import product
from typing import Optional
from typing import Tuple
from typing import Union

import jax.numpy as jnp
from jaxtyping import Array


# bounding box types
BBox2D = Tuple[Tuple[int, int], Tuple[int, int]]
BBox3D = Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int]]

# type for specifying slices for windows
SliceWindow4D = Tuple[slice, slice, slice, slice]
SliceWindow3D = Tuple[slice, slice, slice]
SliceWindow2D = Tuple[slice, slice]
SliceWindow = SliceWindow4D | SliceWindow3D | SliceWindow2D

# type for specifying chunk paddings
_chunk_pads = Union[
    Tuple[int, int],
    Tuple[Tuple[int, int], Tuple[int, int]],
    Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int]],
    Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int], Tuple[int, int]],
]


@dataclass
class ChunkProperties:
    """ChunkProperties class for keeping track of different aspects/windows of a chunked computation.

    ``data_window`` and ``read_window`` differe because the ``read_window`` may include additional data from an adjacent chunk that is read in so that during computation, the actual edge of the chunk does not have edge artifacts.
    ``out_window`` should only be used *after* the computation on the chunk is done but *before* assigning the computation back into a larger array. The slice coordinates for this correspond to those *internal* to the chunk (whereas ``data`` and ``read`` correspond to the coordinates of the original volume).

    Args:
        data_window (SliceWindow): the slice in the volume that this chunk corresponds to
        read_window (SliceWindow): the slice in the volume to be read for the actual computation
        paddings (Tuple[Tuple[int,int]]): padding for each boundary of the chunk
        out_window (SliceWindow): the slice of the computed chunk to be read out after the computation on the chunk
    """

    data_window: SliceWindow
    read_window: SliceWindow
    paddings: _chunk_pads
    out_window: SliceWindow


def crop_and_pad_for_conv(
    vol: Array, bbox: Optional[BBox3D], pad: int
) -> Array:
    """crop_and_pad_for_conv Crop and pad input volume in preparation for convolution.

    Convolution may produce artifacts at the edges of images (volumes).
    To help remedy this, the input can be padded, deconvolved, and then the padded regions cropped out post-deconvolution.
    This function will take the input volume, crop it, and then pad the edges such that after deconvolution, one can re-crop it to get rid of the aforementioned artifacts. instead of just cropping and padding.
    If the bbox and padding are still contained in the original volume, this function will under-crop the appropriate amount such that original parts of the volume are kept (instead of padding).

    Args:
        vol (Array): input volume to be deconvolved
        bbox (Optional[BBox3D]): bounding box to crop volume down to. If ``None``, cropping is not done.
        pad (int): amount of padding past bbox (same for all axes)

    Returns:
        Array
    """
    if bbox is None:
        bbox = tuple(zip([0, 0, 0], list(vol.shape)))
    (padl, padr), (lbd, ubd) = _crop_bounds_and_padding(vol, bbox, pad)
    return _crop_and_pad(vol, padl, padr, lbd, ubd)


def _crop_bounds_and_padding(vol: Array, bbox: BBox3D, pad: int) -> Tuple[
    Tuple[Tuple[int, int, int], Tuple[int, int, int]],
    Tuple[Tuple[int, int, int], Tuple[int, int, int]],
]:
    """_crop_bounds_and_padding determine amount of padding and upper/lower bounds of crop box to use when cropping/padding the input volume in preparation for convolution.

    Args:
        vol (Array): input volume
        bbox (BBox3D): bounding box to crop volume down to
        pad (int): amount of padding past bbox (same for all axes)
        :param vol: input volume

    Returns:
        paddings and upper/lower bounds
    """
    shp = vol.shape
    # figure out lower bound of (maybe) padded image
    lbd = [x[0] - pad for x, _ in zip(bbox, shp)]
    # if the bound is negative, we have to pad to the left
    # otherwise, no padding
    pdl = tuple([abs(ell) if ell < 0 else 0 for ell in lbd])
    # correct for negative bounds which are now 0 (b/c of padding)
    lbd = tuple([0 if ell < 0 else ell for ell in lbd])
    # now we follow the same logic for the upper bound except
    # minuses are pluses and we have to check >shape instead of <0
    ubd = [x[1] + pad for x, _ in zip(bbox, shp)]
    pdr = tuple([v - s if v > s else 0 for v, s in zip(ubd, shp)])
    ubd = tuple([s if v > s else v for v, s in zip(ubd, shp)])
    return (pdl, pdr), (lbd, ubd)


def _crop_and_pad(
    vol: Array,
    padl: Tuple[int, int, int],
    padr: Tuple[int, int, int],
    lowb: Tuple[int, int, int],
    uppb: Tuple[int, int, int],
):
    return jnp.pad(
        vol[lowb[0] : uppb[0], lowb[1] : uppb[1], lowb[2] : uppb[2]],
        [(lft, rgt) for lft, rgt in zip(padl, padr)],
        "symmetric",
    )


def _pad_amount(dim: int, chunk_dim: int) -> int:
    if dim < chunk_dim:
        raise Exception(f"dim ({dim}) should be >= chunk_dim ({chunk_dim})")
    n = 1
    while chunk_dim * n < dim:
        n += 1
    return chunk_dim * n - dim


def _pad_splits(pad_size: int) -> Tuple[int, int]:
    left = pad_size // 2
    right = pad_size // 2 + pad_size % 2
    return left, right


def calculate_3d_chunks(
    z: int,
    r: int,
    c: int,
    chunk_shape: int | Tuple[int, int, int],
    overlap: int | Tuple[int, int, int],
    channel_slice: slice | None,
) -> dict[int, ChunkProperties]:
    """calculate_3d_chunks Compute how an array to be convolved should be chunked into parts for chunkwise convolution.

    Args:
        z (int): linear shape of volume, in z-direction
        r (int): linear shape of volume, in r-direction (# rows)
        c (int): linear shape of volume, in c-direction (# columns)
        chunk_shape (int | Tuple[int,int,int]): shape of chunk. If ``int``, chunks are assumed cubic, otherwise a tuple with int for each dimension.
        overlap (int | Tuple[int,int,int]): amount of overlap (in pixels) between chunks.
        channel_slice (slice | None): how to slice channels in output. If ``None``, all channels are taken.

    Returns:
        dict[int,ChunkChunkPropertiesProps]
    """
    shape = tuple([z, r, c])
    if isinstance(chunk_shape, int):
        chunk_shape = tuple(
            [
                chunk_shape,
            ]
            * 3
        )
    else:
        chunk_shape = chunk_shape
    if isinstance(overlap, int):
        overlap = tuple(
            [
                overlap,
            ]
            * 3
        )
    else:
        overlap = overlap
    # determine padding & resulting shape
    pad_size = [_pad_amount(d, cd) for d, cd in zip(shape, chunk_shape)]
    pads = [_pad_splits(p) for p in pad_size]
    padded_shape = [s + p for s, p in zip(shape, pad_size)]
    # determine how (padded) array will be chunked
    n_chunk = [s // c for s, c in zip(padded_shape, chunk_shape)]
    # make sure division worked correctly (chunks should divide padded shape evenly)
    if not all(
        [n * c == s for n, c, s in zip(n_chunk, chunk_shape, padded_shape)]
    ):
        raise Exception("padding didnt ensure correct division")
    chunk_mults = product(*[range(n) for n in n_chunk])
    chunk_windows = {}
    for chunk_idx, chunk_mult in enumerate(chunk_mults):
        # these indices represent where in the padded data we are reading from
        idx0 = [m * s for m, s in zip(chunk_mult, chunk_shape)]
        idx1 = [i0 + s for i0, s in zip(idx0, chunk_shape)]
        pad_idxs = list(zip(idx0, idx1))
        # now figure out where in the actual data this corresponds to
        data_idxs, pad_amts, read_idxs, out_idxs = [], [], [], []
        for dim_idx, (i0, i1) in enumerate(pad_idxs):
            i0d, i1d = i0 - pads[dim_idx][0], i1 - pads[dim_idx][0]
            # figure out conditions for "left" index
            if i0d < 0:  # we're not starting at data, in the "pad"
                left_data, left_read = 0, 0
                left_pad = pads[dim_idx][0]
                left_out = pads[dim_idx][0]
            else:  # i0d >= 0 -- we're in the data
                left_data = i0d
                left_read = max(i0d - overlap[dim_idx], 0)
                if left_read == 0:  # cant read full overlap in the data
                    left_pad = max(
                        pads[dim_idx][0] - (overlap[dim_idx] - i0d), 0
                    )
                else:  # overlap region fully contained in data
                    left_pad = 0
                left_out = (left_data - left_read) + left_pad
            # figure out conditions for "right" index
            if i1d > shape[dim_idx]:  # we're outside of the data on the rhs
                right_data, right_read = shape[dim_idx], shape[dim_idx]
                right_pad = i1 - padded_shape[dim_idx]
            else:
                right_data = i1d
                right_read = right_data + overlap[dim_idx]
                if right_read > shape[dim_idx]:
                    over_size = right_read - shape[dim_idx]
                    right_read = shape[dim_idx]
                    right_pad = overlap[dim_idx] - over_size
                else:
                    right_pad = 0
            right_out = left_out + (right_data - left_data)
            data_idxs.append((left_data, right_data))
            pad_amts.append((left_pad, right_pad))
            read_idxs.append((left_read, right_read))
            out_idxs.append((left_out, right_out))
        # now make slicing windows for this chunk
        # data windows corresp. to the actual data that this chunk is
        # responsible for
        # read windows corresp. to what should actually get read in
        # when processing this chunk (includes overlap)
        data_window = [slice(left, right) for left, right in data_idxs]
        read_window = [slice(left, right) for left, right in read_idxs]
        out_window = [slice(left, right) for left, right in out_idxs]
        if channel_slice is not None:
            data_window.insert(0, channel_slice)
            read_window.insert(0, channel_slice)
            out_window.insert(0, channel_slice)
            pad_amts.insert(0, (0, 0))
        chunk_windows[chunk_idx] = ChunkProperties(
            tuple(data_window), tuple(read_window), pad_amts, tuple(out_window)
        )
    return chunk_windows
