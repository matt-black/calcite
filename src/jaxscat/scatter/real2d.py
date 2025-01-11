""" Real-space 2D Scattering Transform.

Scattering transform for 2 dimensions, all calculations are done in real domain.
"""

import math
from typing import Callable
from typing import List
from typing import Tuple

import jax
import jax.image as jim
import jax.numpy as jnp
import jax.scipy as jsp
from jax.tree_util import Partial
from jaxtyping import Array
from jaxtyping import Complex
from jaxtyping import Num
from jaxtyping import Real


def apply_filter_bank(
    x: Num[Array, "b c h w"],
    psi: Complex[Array, "l kh kw"],
    mode: str = "same",
    method: str = "fft",
) -> Complex[Array, "b c l h w"]:
    """apply_filter_bank Apply a bank of filters to all channels & batches in input.

    Args:
        x (Num[Array]): input array (4D)
        psi (Complex[Array]): filter bank (NxHxW)
        mode (str, optional): output mode of convolution. Defaults to 'same'.
        method (str, optional): method for computing convolution. Defaults to 'fft'.

    Returns:
        Complex[Array]: 5D tensor of fitler responses, BxCxNxHxW
    """
    return jnp.stack(
        [
            batched_conv2d(x, psi[i, ...], mode, method)
            for i in range(psi.shape[0])
        ],
        axis=-3,
    )


def complex_modulus(x: Complex[Array, "..."]) -> Real[Array, "..."]:
    """complex_modulus Compute complex modulus of input.

    Args:
        x (Complex[Array]): input array

    Returns:
        Real[Array]
    """
    return jnp.abs(x)


def batched_conv2d(
    in1: Num[Array, "..."],
    in2: Num[Array, "kh kw"],
    mode: str = "same",
    method: str = "fft",
) -> Num[Array, "..."]:
    """batched_conv2d 2D convolution of inputs, batched.

    Args:
        in1 (Num[Array]): batched images to be convolved on (4+D)
        in2 (Numy[Array]): second input, kernel, to convolution (2D)
        mode (str, optional): output mode of convolution. Defaults to 'same'.
        method (str, optional): computation method for convolution. Defaults to 'fft'.

    Raises:
        ValueError: if invalid method is supplied. must be 'fft' or 'direct'

    Returns:
        Num[Array]
    """
    in1_shape = in1.shape
    if len(in1.shape) > 4:
        in1 = in1.reshape(in1.shape[0], -1, in1.shape[-2], in1.shape[-1])
    if method == "fft":
        fun = jax.vmap(
            jax.vmap(
                Partial(jsp.signal.fftconvolve, mode=mode, axes=(0, 1)),
                (0, None),
            ),
            (0, None),
        )
    elif method == "direct":
        fun = jax.vmap(
            jax.vmap(Partial(jsp.signal.convolve2d, mode=mode), (0, None)),
            (0, None),
        )
    else:
        raise ValueError("invalid convolution method")
    return fun(in1, in2).reshape(in1_shape)


def scattering_coeffs(
    x: Real[Array, "b c h w"],
    adicity: int,
    n_scales: int,
    phi: Complex[Array, "h w"],
    psi1: Complex[Array, "{n_scales} l kh kw"],
    psi2: Complex[Array, "{n_scales} l kh kw"] | None = None,
    reduction: str = "local",
    conv_method: str = "fft",
    nonlinearity: Callable[[Array], Array] = complex_modulus,
) -> Tuple[Array, Array, List[Array]]:
    """scattering_coeffs Compute scattering coefficients for input.

    Args:
        x (Real[Array]): input image(s) tensor
        adicity (int): logarithm-base used for scale separation.
        n_scales (int): number of scales to compute the scattering over
        phi (Complex[Array, "h w"]): low pass output filter.
        psi1 (Complex[Array, "{n_scales} l h w"]): wavelet filters for first scattering field.
        psi2 (Complex[Array, "{n_scales} l h w"], optional): wavelet filters for 2nd scattering. Defaults to None (psi1 is re-used).
        reduction (str, optional): how to reduce the fields to scattering coefficients. Either "local" or "global". Defaults to "local".
        conv_method (str, optional): method for computing convolutions. Defaults to 'fft'.
        nonlinearity (Callable[[Array],Array], optional): nonlinearity to use after each wavelet transform. Defaults to complex modulus.

    Returns:
        Tuple[Array, Array, List[Array]]: 0th, 1st, 2nd order scattering coefficients.
    """
    psi2 = psi1 if psi2 is None else psi2
    orig_dim = x.shape[-1]

    def scatter_coeff(z: Complex[Array, "..."]) -> Real[Array, "..."]:
        j = int(math.log(orig_dim / z.shape[-1], adicity))
        rest = n_scales - j
        if rest > 0:
            new_shape = list(z.shape[:-2]) + [
                s // (adicity**rest) for s in z.shape[-2:]
            ]
            out = jim.resize(
                batched_conv2d(z, phi, "same", "fft"),
                shape=new_shape,
                method=jim.ResizeMethod.LINEAR,
            )
        else:
            out = batched_conv2d(z, phi, "same", "fft")
        if reduction == "local":
            return out
        else:
            return out.mean(axis=(-2, -1))

    s0 = scatter_coeff(x)
    s1, s2 = [], []
    for j1 in range(n_scales):
        new_shape = list(x.shape[:-2]) + [
            s // (adicity**j1) for s in x.shape[-2:]
        ]
        x_scale = (
            x if j1 == 0 else jim.resize(x, new_shape, jim.ResizeMethod.LINEAR)
        )
        field = nonlinearity(
            apply_filter_bank(x_scale, psi1, mode="same", method=conv_method)
        )
        s1.append(scatter_coeff(field))
        s21 = []
        for j2 in range(j1 + 1, n_scales):
            new_shape = list(field.shape[:-2]) + [
                s // (adicity**j2) for s in field.shape[-2:]
            ]
            f_scale = jim.resize(field, new_shape, jim.ResizeMethod.LINEAR)
            # compute 2nd layer of transform
            f2 = nonlinearity(
                apply_filter_bank(
                    f_scale, psi2, mode="same", method=conv_method
                )
            )
            s21.append(scatter_coeff(f2))
        if len(s21):
            s2.append(s21)
    s1 = jnp.stack(s1, axis=2)
    s2 = list(map(Partial(jnp.stack, axis=2), s2))
    return s0, s1, s2


def scattering_fields(
    x: Real[Array, "b c h w"],
    adicity: int,
    n_scales: int,
    psi1: Complex[Array, "l h w"],
    psi2: Complex[Array, "l h w"] | None = None,
    conv_method: str = "fft",
    nonlinearity: Callable[[Array], Array] = complex_modulus,
) -> Tuple[
    List[Num[Array, "b c l 1 h w"]], List[List[Num[Array, "b c l m h w"]]]
]:
    """scattering_fields Compute scattering fields.

    Args:
        x (Real[Array]): input array, should be 4D
        adicity (int): adicity of scale separation
        n_scales (int): number of scales to compute fields over
        psi1 (Complex[Array]): wavelet filter bank for 1st layer
        psi2 (Complex[Array], optional): wavelet filter bank for second layer. Defaults to None.
        conv_method (str, optional): method for computing convolutions. Defaults to 'fft'.
        nonlinearity (Callable[[Array],Array], optional): nonlinearity to use after each wavelet transform. Defaults to complex modulus.

    Returns:
        Tuple[List[Num[Array]],List[List[Num[Array]]]]: tuple where first element is list of fields at first layer of transform, second is list of lists of fields at second layer of transform.
    """
    psi2 = psi1 if psi2 is None else psi2
    field1, field2 = [], []
    for j1 in range(n_scales):
        # downscale appropriately
        new_shape = [x.shape[-2] // (adicity**j1), x.shape[-1] // (adicity**j1)]
        x_scale = (
            x
            if j1 == 0
            else jim.resize(
                x,
                list(x.shape[:-2]) + new_shape,
                method=jim.ResizeMethod.LINEAR,
            )
        )
        f1 = nonlinearity(
            apply_filter_bank(x_scale, psi1, mode="same", method=conv_method)
        )
        field1.append(f1[:, :, :, None, ...])
        field2s = []
        for j2 in range(j1 + 1, n_scales):
            new_shape = [
                f1.shape[-2] // (adicity**j2),
                f1.shape[-1] // (adicity**j2),
            ]
            f_scale = jim.resize(
                f1,
                list(f1.shape[:-2]) + new_shape,
                method=jim.ResizeMethod.LINEAR,
            )
            # compute 2nd layer of transform
            f2 = nonlinearity(
                apply_filter_bank(
                    f_scale, psi2, mode="same", method=conv_method
                )
            )
            field2s.append(f2)
        if len(field2s):
            field2.append(field2s)
    return field1, field2
