"""Discrete Wavelet Transform.

TODO: more text here, explanation
"""

import math
import operator
from collections.abc import Callable
from typing import List
from typing import Optional
from typing import Tuple

import jax
import jax.numpy as jnp
from jaxtyping import Array
from jaxtyping import Float
from jaxtyping import Num

from ._util import pad_for_dwt
from .wavelet._class import DiscreteWavelet
from .wavelet.discrete import build_wavelet


def wavedec(
    x: Num[Array, "..."],
    wavelet: str | DiscreteWavelet,
    pad_mode: str = "symmetric",
    level: Optional[int] = None,
) -> Tuple[Array, List[List[Array]]]:
    """wavedec compute the multilevel discrete wavelet transform (DWT). # noqa: D403 .

    Args:
        x (Array): input array to compute DWT of
        wavelet (str | DiscreteWavelet): wavelet used to compute DWT.
        pad_mode (str, optional): padding mode used during computation. Defaults to "symmetric".
        level (Optional[int], optional): number of decomposition levels to compute. Defaults to None, in which case level will be automatically computed.

    Returns:
        Tuple[Array, List[List[Array]]]: tuple of approximation coefficients, followed by a list of lists of arrays. Each list of arrays in the list corresponds to the detail coefficients at that level.
    """
    if isinstance(wavelet, str):
        wavelet = build_wavelet(wavelet)
    if level is None:
        # compute max. possible level at which we can take a DWT
        min_axis_shape = min(x.shape)
        filt_len = wavelet.dec_lo.shape[0]
        level = math.floor(math.log2(min_axis_shape // (filt_len - 1)))
    approx = x
    coeffs = []
    for _ in range(level):
        approx, *detail = dwt(x, wavelet.dec_lo, wavelet.dec_hi, pad_mode)
        coeffs.append(detail)
    coeffs.append(approx)
    coeffs.reverse()
    return approx, coeffs


def waverec(
    approx: Array,
    coeffs: List[List[Array]],
    wavelet: str | DiscreteWavelet,
    pad_mode: str = "symmetric",
) -> Array:
    """waverec compute the multilevel inverse discrete wavelet transform. # noqa: D403 .

    Args:
        approx (Array): approximation coefficients
        coeffs (List[List[Array]]): list of detail coefficients at each level.
        wavelet (str | DiscreteWavelet): wavelet used to compute transform.
        pad_mode (str, optional): padding mode used during computation. Defaults to "symmetric".

    Returns:
        Array: array reconstructed from input coefficients.
    """
    if isinstance(wavelet, str):
        wavelet = build_wavelet(wavelet)
    for details in coeffs:
        if len(details) == 1:  # 1d, c_d
            c_d = details[0]
            approx = idwt_1d(
                approx, c_d, wavelet.rec_lo, wavelet.rec_hi, pad_mode
            )
        elif len(details) == 3:  # 2d, (c_da, c_ad, c_dd)
            c_da, c_ad, c_dd = details
            approx = idwt_2d(
                approx,
                c_da,
                c_ad,
                c_dd,
                wavelet.rec_lo,
                wavelet.rec_hi,
                pad_mode,
            )
        elif len(details) == 7:  # 3d
            c_aad, c_ada, c_add, c_daa, c_dad, c_dda, c_ddd = details
            approx = idwt_3d(
                approx,
                c_aad,
                c_ada,
                c_add,
                c_daa,
                c_dad,
                c_dda,
                c_ddd,
                wavelet.rec_lo,
                wavelet.rec_hi,
                pad_mode,
            )
        else:
            raise ValueError("invalid detail coefficient list")
    return approx


def dwt(
    x: Num[Array, "..."],
    dec_lo: Float[Array, " t"],
    dec_hi: Float[Array, " t"],
    pad_mode: str,
) -> Tuple[Array, ...]:
    """dwt compute the (single level) discrete wavelet transform for the input array. # noqa: D403 .

    Args:
        x (Array): input array
        dec_lo (Array): low-pass deconstruction filter
        dec_hi (Array): high-pass deconstruction filter
        pad_mode (str): padding mode used during computation

    Raises:
        ValueError: _description_

    Returns:
        tuple of 1 approximation, then N detail coefficient arrays
    """
    num_dims = jnp.ndim(x)
    if num_dims == 1:
        func = dwt_1d
    elif num_dims == 2:
        func = dwt_2d
    elif num_dims == 3:
        func = dwt_3d
    else:
        raise ValueError("invalid # of dimensions, only valid for 1/2d inputs")
    return func(x, dec_lo, dec_hi, pad_mode)


def dwt_1d(
    x: Num[Array, " s"],
    dec_lo: Float[Array, " t"],
    dec_hi: Float[Array, " t"],
    pad_mode: str,
) -> Tuple[Array, Array]:
    """dwt_1d do 1-dimensional discrete wavelet transform on input array, `x`.

    Args:
        x (Array): input array, should be 1d.
        dec_lo (Array): low-pass filter
        dec_hi (Array): high-pass filter
        pad_mode (str): padding mode.

    Returns:
        Tuple[Array, Array]: approximate, detail coefficients
    """
    if jnp.iscomplexobj(x):
        lor, hir = dwt_1d(jnp.real(x), dec_lo, dec_hi, pad_mode)
        loi, hii = dwt_1d(jnp.imag(x), dec_lo, dec_hi, pad_mode)
        return jax.lax.complex(lor, loi), jax.lax.complex(hir, hii)
    # handle edge padding
    pad_size = dec_lo.shape[0]
    x_pad = pad_for_dwt(x, pad_size, pad_mode)
    conv_pad = list((1, 1)) if pad_mode == "periodization" else list((0, 1))
    lo = jax.lax.conv_general_dilated(
        x_pad[None, None, :], dec_lo[None, None, ::-1], (2,), conv_pad
    )[0, 0, 1:-1]
    hi = jax.lax.conv_general_dilated(
        x_pad[None, None, :], dec_hi[None, None, ::-1], (2,), conv_pad
    )[0, 0, 1:-1]
    return lo, hi


def idwt_1d(
    c_a: Num[Array, " s"],
    c_d: Num[Array, " s"],
    rec_lo: Float[Array, " t"],
    rec_hi: Float[Array, " t"],
    pad_mode: str,
):  # -> Num[Array, " s"]:
    """idwt_1d inverse discrete wavelet transform in one-dimension.

    Args:
        c_a (Array): approximation coefficient array
        c_d (Array): detail coefficient array
        rec_lo (Array): low-pass reconstruction filter
        rec_hi (Array): high-pass reconstruction filter
        pad_mode (str): padding mode used during computation.

    Returns:
        Array
    """
    filt_len = rec_lo.shape[0]
    half_fl = filt_len // 2
    c_a = _upsample_interleave_zeros(c_a, 2)
    c_d = _upsample_interleave_zeros(c_d, 2)
    if pad_mode == "periodization":
        c_a = jnp.pad(c_a, rec_lo, half_fl, mode="wrap")
        c_d = jnp.pad(c_d, rec_hi, half_fl, mode="wrap")
    approx = jnp.convolve(c_a, rec_lo, "same")
    detail = jnp.convolve(c_d, rec_hi, "same")
    recon = approx + detail
    if pad_mode == "periodization":
        return recon[half_fl:-half_fl]
    else:
        skip = half_fl - 1
        if skip > 0:
            return recon[skip:-skip]
        else:
            return recon


def dwt_2d(
    x: Num[Array, "r c"],
    dec_lo: Float[Array, " s"],
    dec_hi: Float[Array, " s"],
    pad_mode: str,
) -> Tuple[Array, Array, Array, Array]:
    """dwt_2d do 2-dimensional discrete wavelet transform on input array, `x`.

    Args:
        x (Array): input array, should be 1d.
        dec_lo (Array): low-pass filter
        dec_hi (Array): high-pass filter
        pad_mode (str): padding mode.

    Returns:
        Tuple[Array, Array, Array, Array]: c_aa, c_da, c_ad, c_dd
    """
    if jnp.iscomplexobj(x):
        aa_r, da_r, ad_r, dd_r = dwt_2d(jnp.real(x), dec_lo, dec_hi, pad_mode)
        aa_i, da_i, ad_i, dd_i = dwt_2d(jnp.imag(x), dec_lo, dec_hi, pad_mode)
        c_aa = jax.lax.complex(aa_r, aa_i)
        c_da = jax.lax.complex(da_r, da_i)
        c_ad = jax.lax.complex(ad_r, ad_i)
        c_dd = jax.lax.complex(dd_r, dd_i)
    else:  # inputs are real, do the computation
        c_a, c_d = jnp.apply_along_axis(dwt_1d, 0, x, dec_lo, dec_hi, pad_mode)
        c_aa, c_ad = jnp.apply_along_axis(
            dwt_1d, 1, c_a, dec_lo, dec_hi, pad_mode
        )
        c_da, c_dd = jnp.apply_along_axis(
            dwt_1d, 1, c_d, dec_lo, dec_hi, pad_mode
        )
    return c_aa, c_da, c_ad, c_dd


def idwt_2d(
    c_aa: Num[Array, "r c"],
    c_da: Num[Array, "r c"],
    c_ad: Num[Array, "r c"],
    c_dd: Num[Array, "r c"],
    rec_lo: Float[Array, " t"],
    rec_hi: Float[Array, " t"],
    pad_mode: str,
) -> Num[Array, "y x"]:
    """idwt_2d inverse discrete wavelet transform in 2-dimensions.

    Args:
        c_aa (Array): approximation coefficient
        c_da (Array): detail coefficient c_da
        c_ad (Array): detail coefficient c_ad
        c_dd (Array): detail coefficient c_dd
        rec_lo (Array): low-pass reconstruction filter
        rec_hi (Array): high-pass reconstruction filter
        pad_mode (str): padding mode used during computation

    Returns:
        Array: reconstructed array
    """
    c_a = _apply_along_axis_2(idwt_1d, 1, c_aa, c_ad, rec_lo, rec_hi, pad_mode)
    c_d = _apply_along_axis_2(idwt_1d, 1, c_da, c_dd, rec_lo, rec_hi, pad_mode)
    return _apply_along_axis_2(idwt_1d, 0, c_a, c_d, rec_lo, rec_hi, pad_mode)


def idwt_3d(
    c_aaa: Num[Array, "z r c"],
    c_aad: Num[Array, "z r c"],
    c_ada: Num[Array, "z r c"],
    c_add: Num[Array, "z r c"],
    c_daa: Num[Array, "z r c"],
    c_dad: Num[Array, "z r c"],
    c_dda: Num[Array, "z r c"],
    c_ddd: Num[Array, "z r c"],
    rec_lo: Float[Array, " t"],
    rec_hi: Float[Array, " t"],
    pad_mode: str,
) -> Num[Array, "w y x"]:
    """idwt_3d inverse discrete wavelet transform in 3 dimensions.

    Args:
        c_aaa (Array): approximation coefficient array, c_aaa
        c_aad (Array): detail coefficient array, c_aad
        c_ada (Array): detail coefficient array, c_ada
        c_add (Array): detail coefficient array, c_add
        c_daa (Array): detail coefficient array, c_daa
        c_dad (Array): detail coefficient array, c_dad
        c_dda (Array): detail coefficient array, c_dda
        c_ddd (Array): detail coefficient array, c_ddd
        rec_lo (Array): low-pass reconstruction filter
        rec_hi (Array): high-pass reconstruction filter
        pad_mode (str): padding mode used during computation

    Returns:
        Array: reconstructed array
    """
    c_dd = _apply_along_axis_2(
        idwt_1d, 2, c_dda, c_ddd, rec_lo, rec_hi, pad_mode
    )
    c_da = _apply_along_axis_2(
        idwt_1d, 2, c_daa, c_dad, rec_lo, rec_hi, pad_mode
    )
    c_ad = _apply_along_axis_2(
        idwt_1d, 2, c_ada, c_add, rec_lo, rec_hi, pad_mode
    )
    c_aa = _apply_along_axis_2(
        idwt_1d, 2, c_aaa, c_aad, rec_lo, rec_hi, pad_mode
    )
    c_a = _apply_along_axis_2(idwt_1d, 1, c_aa, c_ad, rec_lo, rec_hi, pad_mode)
    c_d = _apply_along_axis_2(idwt_1d, 1, c_da, c_dd, rec_lo, rec_hi, pad_mode)
    return _apply_along_axis_2(idwt_1d, 0, c_a, c_d, rec_lo, rec_hi, pad_mode)


def dwt_3d(
    x: Num[Array, "z r c"],
    dec_lo: Float[Array, " s"],
    dec_hi: Float[Array, " s"],
    pad_mode: str,
) -> Tuple[Array, Array, Array, Array, Array, Array, Array, Array]:
    """dwt_3d do 3-dimensional discrete wavelet transform on input array, `x`.

    Args:
        x (Array): input array, should be 1d.
        dec_lo (Array): low-pass filter
        dec_hi (Array): high-pass filter
        pad_mode (str): padding mode.

    Returns:
        Tuple[Array, Array, Array, Array, Array, Array, Array, Array]: c_aaa, c_aad, c_ada, c_add, c_daa, c_dad, c_dda, c_ddd
    """
    if jnp.iscomplexobj(x):
        aaa_r, aad_r, ada_r, add_r, daa_r, dad_r, dda_r, ddd_r = dwt_3d(
            jnp.real(x), dec_lo, dec_hi, pad_mode
        )
        aaa_i, aad_i, ada_i, add_i, daa_i, dad_i, dda_i, ddd_i = dwt_3d(
            jnp.imag(x), dec_lo, dec_hi, pad_mode
        )
        c_aaa = jax.lax.complex(aaa_r, aaa_i)
        c_aad = jax.lax.complex(aad_r, aad_i)
        c_ada = jax.lax.complex(ada_r, ada_i)
        c_add = jax.lax.complex(add_r, add_i)
        c_daa = jax.lax.complex(daa_r, daa_i)
        c_dad = jax.lax.complex(dad_r, dad_i)
        c_dda = jax.lax.complex(dda_r, dda_i)
        c_ddd = jax.lax.complex(ddd_r, ddd_i)
    else:

        def func(y: Array, axis: int) -> Array:
            return jnp.apply_along_axis(
                dwt_1d, axis, y, dec_lo, dec_hi, pad_mode
            )

        c_a, c_d = func(x, 0)
        c_aa, c_ad = func(c_a, 1)
        c_da, c_dd = func(c_d, 1)
        c_aaa, c_aad = func(c_aa, 2)
        c_ada, c_add = func(c_ad, 2)
        c_daa, c_dad = func(c_da, 2)
        c_dda, c_ddd = func(c_dd, 2)
    return c_aaa, c_aad, c_ada, c_add, c_daa, c_dad, c_dda, c_ddd


def _upsample_interleave_zeros(
    x: Num[Array, " s"], factor: int
) -> Num[Array, " 2*s"]:
    sz_out = x.shape[0] * factor
    out = jnp.zeros_like(x, shape=(sz_out,))
    return out.at[::factor].set(x)


def _apply_along_axis_2(
    func1d: Callable, axis: int, arr1: Array, arr2: Array, *args, **kwargs
):
    # canonicalize the input axis
    axis = operator.index(axis)
    num_dims = jnp.ndim(arr1)
    if not -num_dims <= axis < num_dims:
        raise ValueError(
            f"axis {axis} is out of bounds for array of dimension {num_dims}"
        )
    if axis < 0:
        axis = axis + num_dims

    def func(arr1: Array, arr2: Array) -> Array:
        return func1d(arr1, arr2, *args, **kwargs)

    for i in range(1, num_dims - axis):
        func = jax.vmap(func, in_axes=i, out_axes=-1)
    for _ in range(axis):
        func = jax.vmap(func, in_axes=0, out_axes=0)
    return func(arr1, arr2)
