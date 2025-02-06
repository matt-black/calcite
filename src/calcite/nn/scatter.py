""" Layers and utilities for building up wavelet scattering networks.

Currently implemented:
    1) LearnedScatteringLayer: takes in an input, applies a wavelet filter bank to it, then a 1x1 convolution to trim output responses.
"""

import math
from typing import Optional

import equinox as eqx
import jax
from jaxtyping import Array
from jaxtyping import PRNGKeyArray


class LearnedScatteringLayer(eqx.nn.Conv):
    """LearnedScatteringLayer compute a learned scattering field.

    Takes in an N-dimensional, Fourier-space input and computes its channel-wise convolution with the input filter bank whose weights are fixed during training. The output of this is passed to a learnable 1x1 convolution which then produces the output of this layer.
    """

    def __init__(
        self,
        num_spatial_dims: int,
        in_channels: int,
        out_channels: int,
        filter_bank: Array,
    ):
        """__init__ initialize the layer.

        Args:
            num_spatial_dims (int): number of spatial dimensions.
            in_channels (int): number of input channels.
            out_channels (int): number of output channels.
            filter_bank (Array): the fixed wavelet filter bank to use at this layer.
        """
        n_wavelets = math.prod(filter_bank.shape[:-num_spatial_dims])
        super().__init__(
            num_spatial_dims,
            n_wavelets * in_channels,
            out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            dilation=1,
            groups=1,
            use_bias=False,
            padding_mode="ZEROS",
            dtype=jax.numpy.complex64,
        )
        self.filter_bank = jax.stop_gradient(filter_bank)
        self._space_shape = [s for s in filter_bank.shape[-num_spatial_dims:]]

    def __call__(
        self, x: Array, *, key: Optional[PRNGKeyArray] = None
    ) -> Array:
        """__call__ do fixed convolution with the filter bank, then learned 1x1 convolution to trim results of filter bank convolution.

        Args:
            x (Array): the input. should be a JAX array of shape `(in_channels, dim_1, ..., dim_N)` where `N=num_spatial_dims`.
            key (Optional[PRNGKeyArray], optional): ignored, provided for compatability with Equinox API. Defaults to None.

        Returns:
            Array: JAX array of shape `(out_channels, dim_1, ..., dim_N)`
        """
        conv_res = jax.numpy.multiply(
            x[None, ...],
            self.filter_bank.reshape(-1, self._space_shape)[:, None, ...],
        ).reshape(-1, self._space_shape)
        return super().__call__(conv_res, key=key)
