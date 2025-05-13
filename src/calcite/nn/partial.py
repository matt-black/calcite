"""Partial convolutions and related utilities and NN layers.

References
---
[1] Liu, et al. "Image Inpainting for Irregular Holes Using Partial Convolutions" (arXiv:1804.07723)
[2] https://github.com/NVIDIA/partialconv
"""

import math
from collections.abc import Callable
from typing import Optional
from typing import Sequence
from typing import Tuple
from typing import Union

import equinox as eqx
import jax
import jax.numpy as jnp
from jax.tree_util import Partial
from jaxtyping import Array
from jaxtyping import Bool
from jaxtyping import Num
from jaxtyping import PRNGKeyArray


class PartialConv(eqx.nn.Conv):
    """PartialConv JAX implementation of partial convolutions.

    Partial convolutions were first suggested in [1], our implementation is based on the reference PyTorch implementation, [2].

    References
    ---
    [1] Liu, et al. "Image Inpainting for Irregular Holes Using Partial Convolutions" (arXiv:1804.07723)
    [2] https://github.com/NVIDIA/partialconv
    """

    fixed: bool = eqx.field(static=True)
    return_mask: bool = eqx.field(static=True)
    window_size: int = eqx.field(static=True)
    # fft attrs
    is_fft: bool = eqx.field(static=True)
    fft_apply_channelwise: bool = eqx.field(static=True)
    # mask/masking attrs
    mask_update_kernel: Array
    update_mask_fun: Callable[[Array], Array]

    def __init__(
        self,
        num_spatial_dims: int,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Sequence[int]],
        stride: Union[int, Sequence[int]] = 1,
        padding: Union[str, int, Sequence[int], Sequence[Tuple[int, int]]] = 0,
        dilation: Union[int, Sequence[int]] = 1,
        groups: int = 1,
        use_bias: bool = False,
        padding_mode: str = "ZEROS",
        dtype=None,
        return_mask: bool = False,
        fft_conv: bool = False,
        fft_apply_channelwise: bool = False,
        *,
        key: PRNGKeyArray,
    ):
        """__init__ initialize the layer.

        Args:
            num_spatial_dims (int): number of spatial dimensions
            in_channels (int): number of input channels
            out_channels (int): number of output channels
            kernel_size (Union[int, Sequence[int]]): size of the convolution kernel
            key (PRNGKeyArray): a `jax.random.PRNGKey` used to provide randomness for parameter initialization. (Keyword only argument).
            stride (Union[int, Sequence[int]], optional): stride of the convolution. Defaults to 1.
            padding (Union[str, int, Sequence[int], Sequence[Tuple[int, int]]], optional): padding of the convolution. Defaults to 0.
            dilation (Union[int, Sequence[int]], optional): dilation of the convolution. Defaults to 1.
            groups (int, optional): number of input channel groups. Defaults to 1.
            use_bias (bool, optional): whether to add on a bias after the convolution. Defaults to False.
            padding_mode (str, optional): string to specify padding values. See Equinox `nn.Conv` documentation. Defaults to "ZEROS".
            dtype (_type_, optional): dtype to use for the weight and bias in this layer. Defaults to None, which will use either `jnp.float32` or `jnp.float64` depending on whether JAX is in 64-bit mode.
            return_mask (bool, optional): return the current mask. Defaults to False.
            fft_conv (bool, optional): use FFT convolution. Defaults to False.
            fft_apply_channelwise (bool, optional): for FFT convolution, apply filters to all channels independently. Defaults to False.

        Raises:
            NotImplementedError: if `use_bias=True`, which is unimplemented.
        """
        if use_bias:
            raise NotImplementedError("can't use bias with PartialConv, yet")
        super().__init__(
            num_spatial_dims,
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            use_bias,
            padding_mode,
            dtype,
            key=key,
        )
        # internal properties for my use
        # fft specific properties
        self.fixed = False
        self.return_mask = return_mask
        self.is_fft = fft_conv
        self.fft_apply_channelwise = fft_apply_channelwise
        # masking properties for partial convolution
        upd_kernel_size = [out_channels, in_channels]
        if isinstance(self.kernel_size, int):
            upd_kernel_size += [
                self.kernel_size,
            ] * self.num_spatial_dims
        else:
            upd_kernel_size += list(self.kernel_size)
        self.mask_update_kernel = jax.lax.stop_gradient(
            jnp.ones_like(self.weight)
        )
        self.update_mask_fun = Partial(
            jax.lax.conv_general_dilated,
            rhs=self.mask_update_kernel,
            window_strides=self.stride,
            padding=self.padding,
            rhs_dilation=self.dilation,
        )
        self.window_size = math.prod(self.mask_update_kernel[1:].shape)

    def _fft_convolution(self, x: Array, x_fourier: bool) -> Array:
        in_shape = x.shape
        fourier_axes = list(range(1, self.num_spatial_dims + 1))
        if not x_fourier:
            x_fft = jnp.fft.fftn(x, axes=fourier_axes)
        else:
            x_fft = x
        if self.fft_apply_channelwise:
            y_fft = jnp.multiply(
                x_fft[None, ...], self.weight[:, None, ...]
            ).reshape(-1, in_shape[1:])
        else:
            y_fft = jnp.multiply(x_fft, self.weight)
        if x_fourier:
            return y_fft
        else:
            return jnp.fft.ifftn(y_fft, axes=fourier_axes)

    def __call__(
        self,
        x: Num[Array, "..."],
        mask: Bool[Array, "..."] | None = None,
        epsilon: float = 1e-8,
    ) -> Union[Array, Tuple[Array, Array]]:
        """__call__ forward pass of a partial convolution.

        Args:
            x (Num[Array, &quot;...&quot;]): input array
            mask (Bool[Array, &quot;...&quot;] | None, optional): mask array, if None then no masking will be assumed (and its a normal convolution). Defaults to None.
            epsilon (float, optional): small parameter to prevent division by zero. Defaults to 1e-8.

        Raises:
            NotImplementedError: convolution with bias isn't implemented.

        Returns:
            Union[Array, Tuple[Array, Array]]: either the post-convolution array, or the post-convolution array and the updated mask, if `return_mask=True` was set during module initialization.
        """
        # compute updated mask & scaler (ratio)
        if mask is None:
            mask = jnp.ones_like(x)
        # else:
        #    mask = jax.lax.stop_gradient(mask)
        update_mask = self.update_mask_fun(
            mask[None, ...], feature_group_count=1
        ).squeeze(0)
        mask_scaler = self.window_size / (update_mask + epsilon)
        update_mask = jnp.clip(update_mask, 0, 1)
        mask_scaler = jnp.multiply(mask_scaler, update_mask)
        # do the actual convolution, then add bias if applicable
        if self.is_fft:
            out = self._fft_convolution(
                (jnp.multiply(x, mask) if mask is not None else x),
                x_fourier=False,
            )
        else:
            out = super().__call__(
                jnp.multiply(x, mask) if mask is not None else x
            )
        if self.use_bias:
            raise NotImplementedError("todo")
        else:
            out = jnp.multiply(out, mask_scaler)
        if self.return_mask:
            return out, update_mask
        else:
            return out


class PartialPool(eqx.nn.Pool):
    """PartialPool pooling layer to be used with partial convolutions."""

    def __init__(
        self,
        init: Union[int, float, Array],
        operation: Callable[[Array, Array], Tuple[Array, Array]],
        num_spatial_dims: int,
        kernel_size: Union[int, Sequence[int]],
        stride: Union[int, Sequence[int]],
        padding: Union[int, Sequence[int], Sequence[Tuple[int, int]]] = 0,
        use_ceil: bool = False,
    ):
        """__init__ initialize the layer.

        Args:
            init (Union[int, float, Array]): initial value for the reduction
            operation (Callable[[Array, Array], Tuple[Array,Array]]): operation applied to the inputs of each window
            num_spatial_dims (int): # of spatial dimensions
            kernel_size (Union[int, Sequence[int]]): size of convolution kernel
            stride (Union[int, Sequence[int]]): stride of the convolution
            padding (Union[int, Sequence[int], Sequence[Tuple[int, int]]], optional): amount of padding to apply before and after each spatial dimension. Defaults to 0.
            use_ceil (bool, optional): If True, use `ceil` to compute the final output shape instead of `floor`. Defaults to False.
        """
        super().__init__(
            init,
            operation,
            num_spatial_dims,
            kernel_size,
            stride,
            padding,
            use_ceil,
        )

    def __call__(
        self, x: Array, mask: Array, *, key: Optional[PRNGKeyArray] = None
    ) -> Tuple[Array, Array]:
        """__call__ forward pass of pooling operation.

        Args:
            x (Array): input array.
            mask (Array): mask array
            key (Optional[PRNGKeyArray], optional): Ignored; provided for compatibility with equinox API. Defaults to None.

        Returns:
            Tuple[Array, Array]: pooled input and mask.
        """
        x_out = super().__call__(x)
        mask_out = super().__call__(mask)
        return x_out, mask_out


class PartialMaxPool(PartialPool):
    """PartialMaxPool max pooling layer to be used with partial convolutions."""

    def __init__(
        self,
        num_spatial_dims: int,
        kernel_size: Union[int, Sequence[int]],
        stride: Union[int, Sequence[int]] = 1,
        padding: Union[int, Sequence[int], Sequence[Tuple[int, int]]] = 0,
        use_ceil: bool = False,
    ):
        """__init__ initialize a maximum pooling layer for masked inputs.

        Args:
            num_spatial_dims (int): # of spatial dimensions
            kernel_size (Union[int, Sequence[int]]): size of the convolution kernel
            stride (Union[int, Sequence[int]], optional): stride of the convolution. Defaults to 1.
            padding (Union[int, Sequence[int], Sequence[Tuple[int, int]]], optional): amount of padding to add before and after each spatial dimension. Defaults to 0.
            use_ceil (bool, optional): If True, use `ceil` to compute the final output shape instead of `floor`. Defaults to False.
        """
        super().__init__(
            init=-jnp.inf,
            operation=jax.lax.max,
            num_spatial_dims=num_spatial_dims,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            use_ceil=use_ceil,
        )


class PartialConvBlock(eqx.Module):
    """PartialConvBlock A two-layer block of partial convolutions. Often used in UNet-style architectures within the encoder."""

    conv1: PartialConv
    conv2: Optional[PartialConv]
    activation: Callable[[Array], Array]

    def __init__(
        self,
        num_spatial_dims: int,
        single_conv: bool,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: Union[int, Sequence[int]] = 1,
        padding: Union[str, int, Sequence[int], Sequence[Tuple[int, int]]] = 0,
        dilation: Union[int, Sequence[int]] = 1,
        groups: int = 1,
        use_bias: bool = False,
        padding_mode: str = "ZEROS",
        dtype=None,
        fft_conv: bool = False,
        fft_apply_channelwise: bool = False,
        activation: str = "leaky_relu",
        *,
        key: PRNGKeyArray,
    ):
        """__init__ initialize the block of convolutions.

        Args:
            num_spatial_dims (int): # of spatial dimensions
            single_conv (bool): only do a single convolution (instead of default, 2)
            in_channels (int): # of input channels
            out_channels (int): # of output channels
            kernel_size (int): size of convolution kernel
            key (PRNGKeyArray): PRNG key
            stride (Union[int, Sequence[int]], optional): convolution stride, can be specified per-convolution. Defaults to 1.
            padding (Union[str, int, Sequence[int], Sequence[Tuple[int, int]]], optional): padding to apply before/after each spatial dimension. Defaults to 0.
            dilation (Union[int, Sequence[int]], optional): convolution dilation. Defaults to 1.
            groups (int, optional): groups for convolution. Defaults to 1.
            use_bias (bool, optional): whether or not to use a bias term. Defaults to False.
            padding_mode (str, optional): how to do the padding. Defaults to "ZEROS".
            dtype (_type_, optional): datatype for weights in the layer. Defaults to None.
            fft_conv (bool, optional): whether to use FFT convolutions or not. Defaults to False.
            fft_apply_channelwise (bool, optional): whether to apply the FFT convolution channelwise. Defaults to False.
            activation (str, optional): the activation function to use after each convolution. Defaults to "leaky_relu".

        Raises:
            ValueError: if invalid activation function specified.
        """
        key1, key2 = jax.random.split(key, 2)
        self.conv1 = PartialConv(
            num_spatial_dims,
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            use_bias,
            padding_mode,
            dtype,
            True,
            fft_conv,
            fft_apply_channelwise,
            key=key1,
        )
        if not single_conv:
            self.conv2 = PartialConv(
                num_spatial_dims,
                out_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                dilation,
                groups,
                use_bias,
                padding_mode,
                dtype,
                True,
                fft_conv,
                fft_apply_channelwise,
                key=key2,
            )
        else:
            self.conv2 = None
        if activation == "leaky_relu":
            self.activation = Partial(jax.nn.leaky_relu, negative_slope=0.1)
        elif activation == "relu":
            self.activation = jax.nn.relu
        else:
            raise ValueError("only ReLU and Leaky ReLU are valid")

    def __call__(self, x: Array, mask_in: Array) -> Tuple[Array, Array]:
        """__call__ forward pass through the convolution block.

        Args:
            x (Array): input array
            mask_in (Array): mask array

        Returns:
            Tuple[Array,Array]: output array and updated mask
        """
        y, mask = self.conv1(x, mask_in)
        y = self.activation(y)
        if self.conv2 is not None:
            z, mask = self.conv2(x, mask)
            return self.activation(z), mask
        else:
            return y, mask
