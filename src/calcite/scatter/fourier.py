"""Fourier-space 2D Scattering Transform.

Scattering transform for 2 dimensions, where all calculations are done in
the Fourier domain.
"""

import math
from collections.abc import Callable
from typing import List
from typing import Tuple

import jax.numpy as jnp
from jax.tree_util import Partial
from jaxtyping import Array
from jaxtyping import Complex
from jaxtyping import Num
from jaxtyping import Real

from ..periodize import periodize_filter
from .util import apply_filter_bank
from .util import subsample_field


def complex_modulus(x: Complex[Array, "..."]) -> Complex[Array, "..."]:
    """complex_modulus Compute modulus of scattering field.

    Args:
        x (Complex[Array]): input field.

    Returns:
        Real[Array]
    """
    return jnp.fft.fft2(jnp.abs(jnp.fft.ifft2(x)))


def scattering_fields(
    x: Array,
    adicity: int,
    num_scales: int,
    num_spatial_dims: int,
    psi1: Array,
    psi2: Array | None = None,
    nonlinearity: Callable[[Array], Array] = complex_modulus,
) -> Tuple[List[Array], List[List[Array]]]:
    """scattering_fields Compute scattering fields.

    Args:
        x (Array): input array
        adicity (int): adicity of scale separation
        num_scales (int): number of scales to compute fields over
        num_spatial_dims (int): number of spatial dimensions in input array (and filters).
        psi1 (Array): wavelet filter bank for first layer
        psi2 (Array, optional): wavelet filter bank for second layer
        nonlinearity (Callable[[Array],Array], optional): nonlinearity to use after each wavelet transform. Defaults to complex modulus.
        pad_input (bool, optional): pad the input to help prevent convolution artifacts. Defaults to True.
    Raises:
        ValueError: if image width or height isn't evenly divisible by adicity^(n_scales)

    Returns:
        Tuple[List[Array],List[List[Array]]]: tuple where first element is list of fields at first layer of transform, second is list of lists of fields at second layer of transform.
    """
    if x.shape[-1] % adicity**num_scales:
        raise ValueError(
            "image width must be evenly divisible by adicity^(# scales)"
        )
    if x.shape[-2] % adicity**num_scales:
        raise ValueError(
            "image height must be evenly divisible by adicity^(# scales)"
        )
    # set 2nd filter bank to first if not specified explicitly
    psi2 = psi1 if psi2 is None else psi2
    field1 = [
        nonlinearity(
            apply_filter_bank(num_spatial_dims, x, psi1[j, ...], adicity, j, j)
        )
        for j in range(num_scales)
    ]
    field2 = []
    for j1, x1 in enumerate(field1[:-1]):
        x2l = [
            nonlinearity(
                apply_filter_bank(
                    num_spatial_dims, x1, psi2[j2, ...], adicity, j2, j1 + j2
                )
            )
            for j2 in range(j1 + 1, num_scales)
        ]
        field2.append(x2l)
    return field1, field2


def scattering_coeffs(
    x: Array,
    adicity: int,
    num_scales: int,
    num_spatial_dims: int,
    phi: Array,
    psi1: Array,
    psi2: Array | None = None,
    strategy: str = "breadth",
    reduction: str = "local",
    nonlinearity: Callable[[Array], Array] = complex_modulus,
) -> Tuple[Real[Array, "..."], Real[Array, "..."], Real[Array, "..."]]:
    """scattering_coeffs Compute scattering coefficients for input field, `x`.

    Args:
        x (Array): input tensor. Coefficients are computed on last 2 dimensions. Should be Fourier domain.
        adicity (int): logarithm-base used for scale separation.
        num_scales (int): number of scales to compute the scattering over
        num_spatial_dims (int): number of spatial dimensions in the input tensor (and filters).
        phi (Array): low pass output filter.
        psi1 (Array): wavelet filters for first scattering field.
        psi2 (Array, optional): wavelet filters for 2nd scattering. Defaults to None (psi1 is re-used).
        strategy (str, optional): whether computation is done breadth- or depth-first down the scattering tree. Defaults to "breadth".
        reduction (str, optional): how to reduce the fields to scattering coefficients. Either "local" or "global". Defaults to "local".
        nonlinearity (Callable[[Array],Array], optional): nonlinearity to use after each wavelet transform. Defaults to complex modulus.

    Raises:
        ValueError: if specified computation strategy isn't "breadth" or "depth"
        ValueError: if image width or height isn't evenly divisible by adicity^(n_scales)

    Returns:
        Tuple[Real[Array, "..."], Real[Array, "..."], Real[Array, "..."]]
    """
    if strategy not in ("depth", "breadth"):
        raise ValueError(
            'invalid computation strategy (`strategy`), must be "breadth" or "depth"'
        )
    if reduction not in ("global", "local"):
        raise ValueError(
            'invalid reduction strategy (`reduction`) must be "global" or "local"'
        )
    if x.shape[-1] % adicity**num_scales:
        raise ValueError(
            "image width must be evenly divisible by adicity^(# scales)"
        )
    if x.shape[-2] % adicity**num_scales:
        raise ValueError(
            "image height must be evenly divisible by adicity^(# scales)"
        )
    psi2 = psi1 if psi2 is None else psi2
    # number of rotations & phases per scale in the filter bank
    # used because the second-layer outputs are stacked
    n_phi1_rotphs = len(psi1.shape) - (num_spatial_dims + 1)

    def scatter_coeff(z: Num[Array, "..."]) -> Real[Array, "..."]:
        """scatter_coeff Compute scattering coefficient.

        Args:
            z (Num[Array]): input field to compute scattering coefficient of.

        Returns:
            Real[Array]
        """
        j = int(math.log(phi.shape[-1] / z.shape[-1], adicity))
        rest = num_scales - j
        if rest > 0:
            out = jnp.fft.ifft2(
                subsample_field(
                    num_spatial_dims,
                    z * periodize_filter(num_spatial_dims, phi, adicity, j),
                    adicity,
                    rest,
                )
            ).real
        else:
            out = jnp.fft.ifft2(
                z * periodize_filter(num_spatial_dims, phi, adicity, j)
            ).real
        # if the inputs were padded, they won't be evenly divisible by (2**n_scales-1)
        # if padded, need to crop out 2 pixels/dim, one on each "side" of dimension
        if out.shape[-2] == x.shape[-2] // adicity ** (num_scales - 1):
            slc_h = slice(None)
        else:
            slc_h = slice(1, -1)
        if out.shape[-1] == x.shape[-1] // adicity ** (num_scales - 1):
            slc_w = slice(None)
        else:
            slc_w = slice(1, -1)
        # local reduction just returns the local scattering coeffs,
        if reduction == "local":
            return out[..., slc_h, slc_w]
        else:  # global reduction takes mean over space dimensions
            spc_dims = reversed([-i for i in range(1, num_spatial_dims + 1)])
            return jnp.mean(out[..., slc_h, slc_w], axis=spc_dims)

    # first (zero-order) scattering coeff. is just low-pass'd input
    s0 = scatter_coeff(x)
    if strategy == "breadth":
        # \rho(x \conv psi_1)
        # compute wavelet convolution by pointwise mult. in fourier domain
        # multiplication should be elementwise (C11HW x 1LPHW => CLPHW)
        # output is sorted by j, each element corresp. with j-scale
        x1l = [
            nonlinearity(
                apply_filter_bank(
                    num_spatial_dims, x, psi1[j, ...], adicity, j, j
                )
            )
            for j in range(num_scales)
        ]
        s1 = jnp.stack(list(map(scatter_coeff, x1l)), axis=1)
        # repeat process for 2nd layer by scattering the first scattering field
        s2 = []
        for j1, x1 in enumerate(x1l):
            # NOTE: we need to periodize the filter to a total scale of j1+j2
            # so that it matches in shape w. the field being (re-)subsampled
            x2l = [
                scatter_coeff(
                    nonlinearity(
                        apply_filter_bank(
                            num_spatial_dims,
                            x1,
                            psi2[j2, ...],
                            adicity,
                            j2,
                            j1 + j2,
                        )
                    )
                )
                for j2 in range(j1 + 1, num_scales)
            ]
            if len(x2l) > 0:
                s2.append(x2l)
        s2 = list(map(Partial(jnp.stack, axis=1 + n_phi1_rotphs), s2))
    else:
        s1, s2 = [], []
        for j1 in range(num_scales):
            field = nonlinearity(
                apply_filter_bank(
                    num_spatial_dims, x, psi1[j1, ...], adicity, j1, j1
                )
            )
            s1.append(scatter_coeff(field))
            s21 = []
            for j2 in range(j1 + 1, num_scales):
                s21.append(
                    scatter_coeff(
                        nonlinearity(
                            apply_filter_bank(
                                num_spatial_dims,
                                field,
                                psi2[j2, ...],
                                adicity,
                                j2,
                                j1 + j2,
                            )
                        )
                    )
                )
            if len(s21):
                s2.append(s21)
        s1 = jnp.stack(s1, axis=1)
        s2 = list(map(Partial(jnp.stack, axis=1 + n_phi1_rotphs), s2))
    return s0, s1, s2
