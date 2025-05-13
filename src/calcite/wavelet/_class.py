from dataclasses import dataclass
from enum import Enum

from jaxtyping import Array
from jaxtyping import Float


class WaveletSymmetry(Enum):
    """Types of symmetry a wavelet can have."""

    unknown = -1
    asymmetric = 0
    near_symmetric = 1
    symmetric = 2
    anti_symmetric = 3


@dataclass
class Wavelet:
    """Base class for wavelets.

    Simple container holding properties that are common to all wavelets, discrete & continuous.
    """

    support_width: int
    symmetry: WaveletSymmetry
    orthogonal: bool
    biorthogonal: bool
    compact_support: bool


@dataclass
class DiscreteWavelet(Wavelet):
    """Discrete wavelet class.

    Contains de- and reconstruction filters for a given wavelet, along with some properties of the wavelet.
    """

    # filters
    dec_lo: Float[Array, " s"]
    dec_hi: Float[Array, " s"]
    rec_lo: Float[Array, " s"]
    rec_hi: Float[Array, " s"]
    # vanishing moments
    vanishing_moments_wavefun: int
    vanishing_moments_scalefun: int
