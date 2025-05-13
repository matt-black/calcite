"""Self2Self denoising networks.

Self2Self is a denoising architecture that leverages Bernoulli sampling and Dropout to do self-supervised image denoising.

References
---
[1] Quan, et al "Self2Self With Dropout: Learning Self-Supervised Denoising From Single Image", CVPR 2020
[2] https://github.com/scut-mingqinchen/Self2Self
"""

from collections.abc import Callable
from typing import List
from typing import Sequence
from typing import Tuple
from typing import Union

import equinox as eqx
import jax
import numpy
from jax.tree_util import Partial
from jaxtyping import Array
from jaxtyping import PRNGKeyArray
from jaxtyping import PyTree
from tqdm.auto import tqdm

from ..partial import PartialConvBlock
from ..partial import PartialMaxPool


def _select_masked_pixels(
    data: Array, pred: Array, mask: Array
) -> Tuple[Array, Array]:
    """_select_masked_pixels select only the pixels in data and pred inputs that have value 1 at the corresponding coordinate in the input mask.

    Args:
        data (Array): data array
        pred (Array): predictions array
        mask (Array): mask array

    Returns:
        Tuple[Array, Array]: subsets of (data, pred) arrays
    """
    unmask = (1 - mask) > 0.5
    n_pix = jax.numpy.sum(unmask)
    cum_idx = jax.numpy.cumsum(unmask)
    sel_data = (
        jax.numpy.zeros_like(data)
        .at[cum_idx - 1]
        .add(jax.numpy.where(unmask, data, 0))
    )
    sel_pred = (
        jax.numpy.zeros_like(pred)
        .at[cum_idx - 1]
        .add(jax.numpy.where(unmask, pred, 0))
    )
    return sel_data, sel_pred, n_pix


def loss_s2s(data: Array, pred: Array, mask: Array, dist="l2") -> float:
    """loss_s2s loss function used in the Self2Self paper. This is just the L2 loss on the pixels that were masked out during the forward pass.

    Args:
        data (Array): data array
        pred (Array): predictions array
        mask (Array): mask array (should be that used to originally Bernoulli mask data)
        dist (str, optional): distance metric to use, either 'l1' or 'l2'. Defaults to 'l2'.

    Raises:
        ValueError: if invalid distance is specified

    Returns:
        float
    """
    sel_data, sel_pred, num_pix = _select_masked_pixels(data, pred, mask)
    if dist == "l2":
        return (
            jax.numpy.sqrt(jax.numpy.sum(jax.numpy.square(sel_data - sel_pred)))
            / num_pix
        )
    elif dist == "l1":
        return jax.numpy.abs(sel_data - sel_pred) / num_pix
    else:
        raise ValueError("invalid loss function, must be 'l1' or 'l2'")


class BernoulliMaskMaker(eqx.Module):
    """BernoulliMaskMaker module for generating masks for inputs to denoising NNs by Bernoulli sampling pixels/voxel coordinates."""

    dropout: eqx.nn.Dropout
    q_mask: float = eqx.field(static=True)
    indep_channels: bool = eqx.field(static=True)

    def __init__(
        self,
        p_mask: float,
        indep_channels: bool,
    ):
        """__init__ initialize the masking module.

        Args:
            p_mask (float): probability a pixel is masked out.
            indep_channels (bool, optional): whether the channels should be independently masked. if False, a single-channel mask will be generated and applied to all channels, if True then all pixels/voxels in the input are treated independently when masking. Defaults to False.
        """
        self.q_mask = 1.0 - p_mask
        self.indep_channels = indep_channels
        self.dropout = eqx.nn.Dropout(p_mask)

    def __call__(self, x: Array, key: PRNGKeyArray) -> Array:
        """__call__ generate a Bernoulli-sampled mask for the input.

        Args:
            x (Array): input array, to be masked
            key (PRNGKeyArray): PRNG key

        Returns:
            Array: a mask for the input array
        """
        if self.indep_channels:
            return self.dropout(jax.numpy.ones_like(x), key=key) * self.q_mask
        else:
            # generate a mask of a single channel, then repeat it so that all channels are masked in the same way
            mask = self.dropout(
                jax.numpy.ones_like(x, shape=[1] + list(x.shape[1:])), key=key
            )
            return mask.repeat(x.shape[0], axis=0)


class ConvBlock(eqx.Module):
    """ConvBlock a block of convolutions with dropout.

    Used in the upsampling path for Self2Self-style UNets.
    """

    conv1: eqx.nn.Conv
    dropout: eqx.nn.Dropout
    conv2: eqx.nn.Conv
    activation: Callable[[Array], Array]

    def __init__(
        self,
        num_spatial_dims: int,
        in_channels: int,
        out_channels: Union[int, Tuple[int, int]],
        kernel_size: int,
        stride: Union[int, Sequence[int]] = 1,
        padding: Union[str, int, Sequence[int], Sequence[Tuple[int, int]]] = 0,
        dilation: Union[int, Sequence[int]] = 1,
        groups: int = 1,
        use_bias: bool = False,
        padding_mode: str = "ZEROS",
        dtype=None,
        activation: str = "leaky_relu",
        dropout_prob: float = 0.0,
        *,
        key: PRNGKeyArray,
    ):
        """__init__ initialize the block of convolutions.

        Args:
            num_spatial_dims (int): number of spatial dimensions in inputs
            in_channels (int): # of input channels
            out_channels (Union[int, Tuple[int, int]]): # of output channels, if a Tuple will specify the # of intermediate channels in the block.
            kernel_size (int): size of convolution kernel
            key (PRNGKeyArray): PRNG key, keyword-only
            stride (Union[int, Sequence[int]], optional): convolution stride. Defaults to 1.
            padding (Union[str, int, Sequence[int], Sequence[Tuple[int, int]]], optional): amount of padding. Defaults to 0.
            dilation (Union[int, Sequence[int]], optional): convolution dilation. Defaults to 1.
            groups (int, optional): groups used in convolution. Defaults to 1.
            use_bias (bool, optional): whether or not to include a bias term in convolutions. Defaults to False.
            padding_mode (str, optional): values used for padding. Defaults to "ZEROS".
            dtype (_type_, optional): datatype used for weights. Defaults to None.
            activation (str, optional): the activation function to use after convolutions. Defaults to "leaky_relu".
            dropout_prob (float, optional): probability of dropout between the convolutions. Defaults to 0.0.

        Raises:
            ValueError: if invalid activation function specified
        """
        key1, key2 = jax.random.split(key, 2)
        if isinstance(out_channels, int):
            out_channels = (out_channels, out_channels)
        self.conv1 = eqx.nn.Conv(
            num_spatial_dims,
            in_channels,
            out_channels[0],
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            use_bias,
            padding_mode,
            dtype,
            key=key1,
        )
        if dropout_prob == 0:
            self.dropout = eqx.nn.Dropout(inference=True)
        else:
            self.dropout = eqx.nn.Dropout(p=dropout_prob)
        self.conv2 = eqx.nn.Conv(
            num_spatial_dims,
            out_channels[0],
            out_channels[1],
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            use_bias,
            padding_mode,
            dtype,
            key=key2,
        )
        if activation == "leaky_relu":
            self.activation = Partial(jax.nn.leaky_relu, negative_slope=0.1)
        elif activation == "relu":
            self.activation = jax.nn.relu
        else:
            raise ValueError("only ReLU and Leaky ReLU are valid")

    def __call__(self, x: Array, key: PRNGKeyArray) -> Array:
        """__call__ forward pass through the block.

        Args:
            x (Array): input array
            key (PRNGKeyArray): PRNG key

        Returns:
            Array
        """
        y = self.activation(self.conv1(x))
        z = self.dropout(y, key=key)
        return z if self.conv2 is None else self.activation(self.conv2(z))


class UpBlock(eqx.Module):
    """UpBlock Decoder block for use in a UNet.

    Takes in the current array being passed through the network as well as an "encoding" that comes from a skip connection from the encoder.
    """

    block: ConvBlock
    upsample: Union[eqx.nn.ConvTranspose, Callable[[Array], Array]]

    def __init__(
        self,
        num_spatial_dims: int,
        in_channels: int,
        out_channels: Union[int, Tuple[int, int]],
        upsampling_mode: str,
        kernel_size: Union[int, Sequence[int]] = 3,
        stride: Union[int, Sequence[int]] = 1,
        padding: Union[str, int, Sequence[int], Sequence[Tuple[int, int]]] = 0,
        dilation: Union[int, Sequence[int]] = 1,
        groups: int = 1,
        use_bias: bool = False,
        padding_mode: str = "ZEROS",
        dtype=None,
        activation: str = "leaky_relu",
        dropout_prob: float = 0.3,
        *,
        key: PRNGKeyArray,
    ):
        """__init__ initialize the decoder block.

        Args:
            num_spatial_dims (int): # of spatial dims in input arrays (assumed trailing)
            in_channels (int): # of input channels
            out_channels (Union[int, Tuple[int,int]]): # of output channels, if Tuple, will also specify intermediate # of channels.
            upsampling_mode (str): 'conv' or 'interp', how to do the upsampling
            key (PRNGKeyArray): PRNG key, keyword-only
            kernel_size (Union[int, Sequence[int]], optional): size of convolution kernel. Defaults to 3.
            stride (Union[int, Sequence[int]], optional): convolutional stride. Defaults to 1.
            padding (Union[str, int, Sequence[int], Sequence[Tuple[int, int]]], optional): amount of padding to add. Defaults to 0.
            dilation (Union[int, Sequence[int]], optional): convolutional dilation. Defaults to 1.
            groups (int, optional): # of groups for convolution. Defaults to 1.
            use_bias (bool, optional): whether to use a bias term or not. Defaults to False.
            padding_mode (str, optional): values used for padding. Defaults to "ZEROS".
            dtype (_type_, optional): datatype of weights for this block. Defaults to None.
            activation (str, optional): activation function to use after convolutions. Defaults to "leaky_relu".
            dropout_prob (float, optional): probability of dropout in the convolution block. Defaults to 0.3.
        """
        key1, key2 = jax.random.split(key, 2)
        self.block = ConvBlock(
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
            activation,
            dropout_prob,
            key=key1,
        )
        if upsampling_mode == "conv":
            self.upsample = eqx.nn.ConvTranspose(
                num_spatial_dims,
                in_channels,
                in_channels,
                kernel_size=2,
                stride=2,
                key=key2,
            )
        else:
            self.upsample = Partial(
                _upsample_2x,
                num_spatial_dims=num_spatial_dims,
                method=upsampling_mode,
            )

    def __call__(self, x: Array, enc: Array, key: PRNGKeyArray) -> Array:
        """__call__ forward pass of the decoder block.

        Takes in the current array being fed through the network and the "encoding" that comes from a skip connection made from the encoder at the equivalent resolution, combines them (channel-wise), and then does convolutions.

        Args:
            x (Array): input array
            enc (Array): encoded array from "across" the UNet
            key (PRNGKeyArray): PRNG key

        Returns:
            Array
        """
        y = self.upsample(x)
        return self.block(jax.numpy.concatenate([y, enc], axis=0), key)


def _upsample_2x(x: Array, num_spatial_dims: int, method: str) -> Array:
    """_upsample_2x upsample the input array to twice its original size.

    Args:
        x (Array): input array, to be upsampled
        num_spatial_dims (int): # of spatial dimensions in the input (assumed trailing)
        method (str): interpolation method for upsampling.

    Returns:
        Array
    """
    output_shape = [s for s in x.shape[:-num_spatial_dims]] + [
        2 * s for s in x.shape[-num_spatial_dims:]
    ]
    return jax.image.resize(x, shape=output_shape, method=method)


class PartialUNet(eqx.Module):
    """PartialUNet UNet architecture that uses partial convolutions in the encoder.

    Default parameters are taken from the Self2Self paper (see [1]).
    """

    encoder_layers: List[PartialConvBlock]
    decoder_layers: List[UpBlock]
    maxpool_layer: PartialMaxPool
    output_dropout: eqx.nn.Dropout
    output_conv: eqx.nn.Conv
    output_activation: Callable[[Array], Array]

    def __init__(
        self,
        num_spatial_dims: int,
        in_channels: int,
        out_channels: int,
        enc_channels: Sequence[int] = tuple(
            [
                48,
            ]
            * 6
        ),
        dec_channels: Sequence[Union[int, Tuple[int, int]]] = tuple(
            [
                96,
                96,
                96,
                96,
                (64, 32),
            ]
        ),
        kernel_size: Union[int, Sequence[int]] = 3,
        stride: Union[int, Sequence[int]] = 1,
        padding: Union[str, int, Sequence[int], Sequence[Tuple[int, int]]] = 1,
        dilation: Union[int, Sequence[int]] = 1,
        groups: int = 1,
        use_bias: bool = False,
        padding_mode: str = "ZEROS",
        dtype=None,
        upsampling_mode: str = "linear",
        activation: str = "leaky_relu",
        output_activation: str = "sigmoid",
        dropout_prob: float = 0.0,
        *,
        key: PRNGKeyArray,
    ):
        """__init__ initialize the network.

        Args:
            num_spatial_dims (int): number of spatial dimensions
            in_channels (int): number of input channels
            out_channels (int): number of output channels
            key (PRNGKeyArray): a `jax.random.PRNGKey` used to provide randomness for parameter initialization. (Keyword only argument)
            enc_channels (Sequence[int], optional): list of number of channels output at each layer of the encoder. Defaults to [48,]*6.
            dec_channels (Sequence[Union[int,Tuple[int,int]]], optional): list of number of channels output at each layer of the decoder. Tuples can be specified so that the intermediate channel at the block can be specified. Defaults to [96, 96, 96, 96, (64, 32)].
            kernel_size (Union[int, Sequence[int]], optional): size of convolutional kernels. Defaults to 3.
            stride (Union[int, Sequence[int]], optional): convolution stride. Defaults to 1.
            padding (Union[str, int, Sequence[int], Sequence[Tuple[int, int]]], optional): padding of the convolution. Defaults to 0.
            dilation (Union[int, Sequence[int]], optional): dilation of the convolution. Defaults to 1.
            groups (int, optional): number of input channel groups. Defaults to 1.
            use_bias (bool, optional): whether to add a bis on after each convolution. Defaults to False.
            padding_mode (str, optional): string to specify padding values. Defaults to "ZEROS".
            dtype (_type_, optional): dtype to use for the weight and bias in this layer. Defaults to None.
            upsampling_mode (str, optional): how upsampling is handled by the decoder arm. Can be one of the modes for `jax.image.resize` or 'conv'. Defaults to "linear".
            activation (str, optional): activation to use after each convolution. Defaults to 'leaky_relu'.
            output_activation (str, optional): activation of output. Defaults to 'sigmoid'.
            dropout_prob (float, optional): dropout probability. Defaults to 0.0.
        """
        if len(dec_channels) != len(enc_channels) - 1:
            raise ValueError("decoder must be 1 element shorter than encoder")
        keys = jax.random.split(key, len(enc_channels) + len(dec_channels) + 1)
        # setup the encoding pathway
        self.encoder_layers = list()
        for in_chan, out_chan, ekey in zip(
            [in_channels] + enc_channels[:-1],
            enc_channels,
            keys[: len(enc_channels)],
        ):
            single_conv = in_chan == in_channels
            self.encoder_layers.append(
                PartialConvBlock(
                    num_spatial_dims,
                    single_conv,
                    in_chan,
                    out_chan,
                    kernel_size,
                    stride,
                    padding,
                    dilation,
                    groups,
                    use_bias,
                    padding_mode,
                    dtype,
                    fft_conv=False,
                    fft_apply_channelwise=False,
                    activation=activation,
                    key=ekey,
                )
            )
        self.maxpool_layer = PartialMaxPool(num_spatial_dims, 2, 2)
        # setup the decoding pathway
        concat_chan = reversed([in_channels] + enc_channels[:-2])
        dec_in_chan = [
            c + p
            for c, p in zip(concat_chan, [enc_channels[-1]] + dec_channels)
        ]
        dec_keys = keys[
            len(enc_channels) : len(enc_channels) + len(dec_channels)
        ]
        self.decoder_layers = list()
        for in_chan, out_chan, dkey in zip(dec_in_chan, dec_channels, dec_keys):
            self.decoder_layers.append(
                UpBlock(
                    num_spatial_dims,
                    in_chan,
                    out_chan,
                    upsampling_mode,
                    kernel_size,
                    stride,
                    padding,
                    dilation,
                    groups,
                    use_bias,
                    padding_mode,
                    dtype,
                    activation,
                    dropout_prob,
                    key=dkey,
                )
            )
        # setup the output convolution & activations
        self.output_dropout = eqx.nn.Dropout(dropout_prob)
        # the last decoder channels could be a single int or a tuple of ints,
        # account for this here to make sure an int gets passed to the Conv layer
        if isinstance(dec_channels[-1], int):
            last_dec = dec_channels[-1]
        else:
            last_dec = dec_channels[-1][1]
        self.output_conv = eqx.nn.Conv(
            num_spatial_dims,
            last_dec,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            use_bias,
            padding_mode,
            dtype,
            key=keys[-1],
        )
        if output_activation == "sigmoid":
            self.output_activation = jax.nn.sigmoid
        elif output_activation == "relu":
            self.output_activation = jax.nn.relu
        elif output_activation == "leaky_relu":
            self.output_activation = Partial(
                jax.nn.leaky_relu, negative_slope=0.1
            )
        else:
            raise ValueError("invalid output_activation")

    def __call__(self, x: Array, mask: Array, key: PRNGKeyArray) -> Array:
        """__call__ generate predictions for input array and its mask (forward pass through UNet).

        Args:
            x (Array): input array
            mask (Array): mask array
            key (PRNGKeyArray): PRNG key

        Returns:
            Array
        """
        # encoder layers
        encoder_depth = len(self.encoder_layers)
        intermediate_encodings = list()
        for idx, encoder_layer in enumerate(self.encoder_layers):
            if idx < encoder_depth - 1:
                intermediate_encodings.append(x)
            x, mask = encoder_layer(x, mask)
            if idx < encoder_depth - 1:
                x, mask = self.maxpool_layer(x, mask)
        # decoder
        dec_keys = jax.random.split(key, len(self.decoder_layers))
        for decoder_layer, dkey in zip(self.decoder_layers, dec_keys):
            x = decoder_layer(x, intermediate_encodings.pop(-1), dkey)
        # output layers
        _, out_key = jax.random.split(dec_keys[-1], 2)
        x = self.output_dropout(x, key=out_key)
        return self.output_activation(self.output_conv(x))


class AdditiveWhiteGaussianNoise(eqx.Module):
    """AdditiveWhiteGaussianNoise module to add white gaussian noise to inputs."""

    sigma: float = eqx.field(static=True)

    def __init__(
        self,
        sigma: float,
    ):
        """__init__ initialize the module.

        Args:
            sigma (float): std. deviation of Gaussian.
        """
        self.sigma = sigma

    def __call__(self, x: Array, key: PRNGKeyArray) -> Array:
        """__call__ adds Gaussian white noise to the input array.

        Args:
            x (Array): input array.
            key (PRNGKeyArray): PRNG key

        Returns:
            Array: input with added Gaussian white noise
        """
        noise = self.sigma * jax.random.normal(
            key, shape=x.shape, dtype=x.dtype
        )
        return x + noise


class SaltAndPepperNoise(eqx.Module):
    """SaltAndPepperNoise module to add salt & pepper impulse noise to inputs."""

    q_black: float = eqx.field(static=True)
    p_white: float = eqx.field(static=True)
    v_white: float = eqx.field(static=True)
    indep_channels: bool = eqx.field(static=True)
    drop_black: eqx.nn.Dropout
    drop_white: eqx.nn.Dropout

    def __init__(
        self,
        p_black: float,
        p_white: float,
        val_white: float,
        indep_channels: bool = False,
    ):
        """__init__ initialize the S&P noise generator.

        Args:
            p_black (float): probability of any pixel being set to black (0)
            p_white (float): probability of any pixel being set to white (max)
            val_white (float): value to set "white" pixels to.
            indep_channels (bool, optional): whether to apply noise independently to all channels or not. Defaults to False.
        """
        self.q_black = 1.0 - p_black
        self.p_white = p_white
        self.v_white = val_white
        self.drop_black = eqx.nn.Dropout(p_black)
        self.drop_white = eqx.nn.Dropout(1.0 - p_white)
        self.indep_channels = indep_channels

    def __call__(self, x: Array, key: PRNGKeyArray) -> Array:
        """__call__ add S&P noise to the input array.

        Args:
            x (Array): input array.
            key (PRNGKeyArray): PRNG key

        Returns:
            Array: noisy array
        """
        white_renorm = self.p_white * self.v_white
        if self.indep_channels:
            # do dropout with energy normalization
            m_black = (
                self.drop_black(jax.numpy.ones_like(x), key=key) * self.q_black
            )
            m_white = self.drop_white(x, key=key) * white_renorm
        else:
            m_black = jax.numpy.expand_dims(
                self.drop_black(jax.numpy.ones_like(x[0, ...]), key=key)
                * self.q_black,
                0,
            )
            m_white = jax.numpy.expand_dims(
                self.drop_white(x[0, ...], key=key) * white_renorm, 0
            )
        return jax.numpy.clip((x + m_white) * m_black, min=0, max=self.v_white)


def train(
    model: PartialUNet,
    image: Array | None,
    optim,
    masker: BernoulliMaskMaker,
    steps: int,
    augment_flips: bool,
    verbose: bool,
    key: PRNGKeyArray,
) -> PartialUNet:
    """Train a PartialUNet model to denoise input images, using the method described in Self2Self.

    Args:
        model (PartialUNet): the model, a PartialUNet
        image (Array | None): noisy image, to be trained to denoise.
        optim (_type_): the optimizer (from `optax`)
        masker (BernoulliMaskMaker): module to do the Bernoulli masking of the input array.
        steps (int): number of training iterations to do
        augment_flips (bool): also do (probabilistic) vertical & horizontal flips to input during training.
        verbose (bool): show progress bar during training, counting down # of steps.
        key (PRNGKeyArray): PRNG key

    Returns:
        PartialUNet: trained model
    """
    opt_state = optim.init(eqx.filter(model, eqx.is_array))

    def _loss(model: PartialUNet, image: Array, mask: Array, key: PRNGKeyArray):
        pred = jax.vmap(model)(image * mask, mask, key)
        return loss_s2s(image, pred, mask, "l2")

    _loss = eqx.filter_jit(_loss)

    @eqx.filter_jit
    def _make_step(
        model: PartialUNet, x: Array, opt_state: PyTree, key: PRNGKeyArray
    ):
        key_mask, key_loss = jax.random.split(key, 2)
        # batch-ify the keys
        key_mask = jax.random.split(key_mask, x.shape[0])
        key_loss = jax.random.split(key_loss, x.shape[0])
        # do
        if augment_flips:
            do_flip = numpy.random.rand(2) > 0.5
            if do_flip[0]:
                x = x[:, ::-1, :]
            if do_flip[1]:
                x = x[..., ::-1]
        mask = jax.vmap(masker)(x, key=key_mask)
        loss_val, grads = eqx.filter_value_and_grad(_loss)(
            model, x, mask, key_loss
        )
        updates, opt_state = optim.update(
            grads, opt_state, eqx.filter(model, eqx.is_array)
        )
        model = eqx.apply_updates(model, updates)
        return model, opt_state, loss_val

    losses = []
    with (
        tqdm(range(steps), desc="S2S Training") if verbose else range(steps)
    ) as pbar:
        for _ in pbar:
            _, key = jax.random.split(key)
            model, opt_state, train_loss = _make_step(
                model, image, opt_state, key
            )
            losses.append(train_loss)
            if verbose:
                pbar.set_postfix({"loss": round(train_loss * 100, 2)})
    return model, losses


def test(
    model: PartialUNet,
    image: Array,
    masker: BernoulliMaskMaker,
    n_samples: int = 50,
    batched: bool = False,
    *,
    key: PRNGKeyArray,
) -> Array:
    """Generate model prediction by sampling and averaging many masked instances (see Self2Self paper for description).

    Args:
        model (PartialUNet): the (presumably trained) model
        image (Array): image to denoise
        masker (BernoulliMaskMaker): masking module for Bernoulli-sampled masks
        key (PRNGKeyArray): PRNG key
        n_samples (int, optional): number of samples to take and average. Defaults to 50.
        batched (bool, optional): whether to do all the samples in a single, batch-style or loop through and calculate running parameters to save memory. Defaults to False.

    Raises:
        NotImplementedError: batched=False not implemented, yet.

    Returns:
        Array
    """
    key_model, key_mask = jax.random.split(key, 2)
    if batched:
        key_mask = jax.random.split(key_mask, n_samples)
        mask = jax.lax.stop_gradient(jax.vmap(masker)(image, key=key_mask))
        image = jax.lax.stop_gradient(
            jax.numpy.expand_dims(image, 0).repeat(n_samples, axis=0)
        )
        key_model = jax.random.split(key_model, n_samples)
        pred = jax.vmap(model)(image, mask, key_model)
        return jax.numpy.mean(pred, axis=0)
    else:
        raise NotImplementedError("todo")
