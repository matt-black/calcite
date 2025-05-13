"""Discrete Wavelets.

Functions and utilities for handling and computing with discrete wavelets.
"""

from typing import Tuple

from jaxtyping import Array
from jaxtyping import Float

from . import _coeffs
from ._class import DiscreteWavelet
from ._class import WaveletSymmetry


__all__ = ["build_wavelet", "filter_bank", "quadrature_mirror_filter"]


def build_wavelet(name: str) -> DiscreteWavelet:
    """Create a DiscreteWavelet object from the wavelet name.

    Args:
        name (str): name of the wavelet to construct

    Returns:
        DiscreteWavelet
    """
    name = name.lower()
    if name.startswith("haar"):
        dec_lo, dec_hi, rec_lo, rec_hi = filter_bank(_coeffs.haar)
        return DiscreteWavelet(
            1,
            WaveletSymmetry.asymmetric,
            True,
            True,
            True,
            dec_lo,
            dec_hi,
            rec_lo,
            rec_hi,
            1,
            0,
        )
    elif name.startswith("db"):
        order = int(name[2:])
        idx = order - 1  # db1 is smallest Daubechies wavelet
        dec_lo, dec_hi, rec_lo, rec_hi = filter_bank(_coeffs.db[idx])
        return DiscreteWavelet(
            2 * order - 1,
            WaveletSymmetry.asymmetric,
            True,
            True,
            True,
            dec_lo,
            dec_hi,
            rec_lo,
            rec_hi,
            order,
            0,
        )
    elif name.startswith("sym"):
        order = int(name[3:])
        idx = order - 2  # sym2 is smallest symlet
        dec_lo, dec_hi, rec_lo, rec_hi = filter_bank(_coeffs.sym[idx])
        return DiscreteWavelet(
            2 * order - 1,
            WaveletSymmetry.near_symmetric,
            True,
            True,
            True,
            dec_lo,
            dec_hi,
            rec_lo,
            rec_hi,
            order,
            0,
        )
    elif name.startswith("coif"):
        order = int(name[4:])
        idx = order - 1  # coif1 is smallest Coiflet, comes first in list
        dec_lo, dec_hi, rec_lo, rec_hi = filter_bank(_coeffs.coif[idx])
        return DiscreteWavelet(
            6 * order - 1,
            WaveletSymmetry.near_symmetric,
            True,
            True,
            True,
            dec_lo,
            dec_hi,
            rec_lo,
            rec_hi,
            2 * order,
            2 * order - 1,
        )
    elif name.startswith("bior"):
        idx0 = int(name[4]) - 1
        val1 = int(name[-1])
        if idx0 == 0:  # bior1
            if val1 > 1:
                idx1 = 2 if val1 == 3 else 3
            else:
                idx1 = val1
        elif idx0 == 1:  # bior2
            idx1 = val1 // 2
        elif idx0 == 2:  # bior3
            idx1 = {0: 0, 1: 1, 3: 2, 5: 3, 7: 4, 9: 5}[val1]
        elif idx0 == 3:  # bior4
            idx1 = val1 // 4
        elif idx0 == 4:  # bior5
            idx1 = val1 // 5
        elif idx0 == 5:  # bior6
            idx1 = val1 // 8
        dec_lo, dec_hi, rec_lo, rec_hi = filter_bank(_coeffs.bior[idx0][idx1])
        raise NotImplementedError("todo")
    elif name.startswith("dmey"):
        dec_lo, dec_hi, rec_lo, rec_hi = filter_bank(_coeffs.dmey)
        return DiscreteWavelet(
            1,
            WaveletSymmetry.symmetric,
            True,
            True,
            True,
            dec_lo,
            dec_hi,
            rec_lo,
            rec_hi,
            -1,
            -1,
        )
    else:
        raise ValueError("invalid wavelet name")


def filter_bank(
    rec_lo: Float[Array, " s"],
) -> Tuple[Array, Array, Array, Array]:
    """Construct a filter bank from the low-pass reconstruction filter.

    Args:
        rec_lo (Float[Array, 's']): low-pass reconstruction filter, of size "s"
    Returns:
        Tuple[Array,Array,Array,Array]: dec_lo, dec_hi, rec_lo, rec_hi
    """
    dec_lo = rec_lo[::-1]
    rec_hi = quadrature_mirror_filter(rec_lo)
    dec_hi = rec_hi[::-1]
    return (dec_lo, dec_hi, rec_lo, rec_hi)


def quadrature_mirror_filter(x: Float[Array, " s"]) -> Float[Array, " s"]:
    """Create quadrature mirror filter for input filter.

    Args:
        x (Float[Array, 's']): input filter sequence (1d)
    Returns:
        Float[Array, 's']: quadrature mirror filter of input filter
    """
    y = x[::-1]
    return y.at[1::2].set(-y[1::2])
