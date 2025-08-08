# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""StyleGAN2 modules"""

import math
import torch
from torch import nn
from torch.nn import functional as F

from nvidia_tao_ds.skin_tone.models.psp.stylegan2.op import FusedLeakyReLU, fused_leaky_relu, upfirdn2d


def make_kernel(k):
    """Upsampling kernel"""
    k = torch.tensor(k, dtype=torch.float32)

    if k.ndim == 1:
        k = k[None, :] * k[:, None]

    k /= k.sum()

    return k


class Upsample(nn.Module):
    """Upsample module"""

    def __init__(self, kernel, factor=2):
        """Initialize Upsampling module"""
        super().__init__()

        self.factor = factor
        kernel = make_kernel(kernel) * (factor**2)
        self.register_buffer("kernel", kernel)

        p = kernel.shape[0] - factor

        pad0 = (p + 1) // 2 + factor - 1
        pad1 = p // 2

        self.pad = (pad0, pad1)

    def forward(self, x):
        """Foward pass"""
        out = upfirdn2d(x, self.kernel, up=self.factor, down=1, pad=self.pad)

        return out


class Blur(nn.Module):
    """Blur module"""

    def __init__(self, kernel, pad, upsample_factor=1):
        """Initialize Blur module"""
        super().__init__()

        kernel = make_kernel(kernel)

        if upsample_factor > 1:
            kernel = kernel * (upsample_factor**2)

        self.register_buffer("kernel", kernel)

        self.pad = pad

    def forward(self, x):
        """Foward pass"""
        out = upfirdn2d(x, self.kernel, pad=self.pad)

        return out


class EqualLinear(nn.Module):
    """Equal Linear module"""

    def __init__(
        self, in_dim, out_dim, bias=True, bias_init=0, lr_mul=1, activation=None
    ):
        """Initialize EqualLiner module"""
        super().__init__()

        self.weight = nn.Parameter(torch.randn(out_dim, in_dim).div_(lr_mul))

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_dim).fill_(bias_init))

        else:
            self.bias = None

        self.activation = activation

        self.scale = (1 / math.sqrt(in_dim)) * lr_mul
        self.lr_mul = lr_mul

    def forward(self, x):
        """Forward pass"""
        if self.activation:
            out = F.linear(x, self.weight * self.scale)
            out = fused_leaky_relu(out, self.bias * self.lr_mul)

        else:
            out = F.linear(
                x, self.weight * self.scale, bias=self.bias * self.lr_mul
            )

        return out

    def __repr__(self):
        """Object represnetation"""
        return (
            f"{self.__class__.__name__}({self.weight.shape[1]}, {self.weight.shape[0]})"
        )


class ModulatedConv2d(nn.Module):
    """ModulatedConv2d module"""

    def __init__(
        self,
        in_channel,
        out_channel,
        kernel_size,
        style_dim,
        demodulate=True,
        upsample=False,
        blur_kernel=[1, 3, 3, 1],
    ):
        """Initialize ModulatedConv2d module"""
        super().__init__()

        self.eps = 1e-8
        self.kernel_size = kernel_size
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.upsample = upsample

        if upsample:
            factor = 2
            p = (len(blur_kernel) - factor) - (kernel_size - 1)
            pad0 = (p + 1) // 2 + factor - 1
            pad1 = p // 2 + 1

            self.blur = Blur(blur_kernel, pad=(pad0, pad1), upsample_factor=factor)

        fan_in = in_channel * kernel_size**2
        self.scale = 1 / math.sqrt(fan_in)
        self.padding = kernel_size // 2

        self.weight = nn.Parameter(
            torch.randn(1, out_channel, in_channel, kernel_size, kernel_size)
        )

        self.modulation = EqualLinear(style_dim, in_channel, bias_init=1)

        self.demodulate = demodulate

    def __repr__(self):
        """Representation string"""
        return (
            f"{self.__class__.__name__}({self.in_channel}, {self.out_channel}, {self.kernel_size}, "
            f"upsample={self.upsample}, downsample={self.downsample})"
        )

    def forward(self, x, style, is_stylespace=False):
        """Forward pass"""
        batch, in_channel, height, width = x.shape

        weight = self.weight

        if not is_stylespace:
            style = self.modulation(style)
        style = style.view(batch, 1, in_channel, 1, 1)
        weight = self.scale * weight * style

        if self.demodulate:
            demod = torch.rsqrt(weight.pow(2).sum([2, 3, 4]) + 1e-8)
            weight = weight * demod.view(batch, self.out_channel, 1, 1, 1)

        weight = weight.view(
            batch * self.out_channel, in_channel, self.kernel_size, self.kernel_size
        )

        if self.upsample:
            x = x.view(1, batch * in_channel, height, width)
            weight = weight.view(
                batch, self.out_channel, in_channel, self.kernel_size, self.kernel_size
            )
            weight = weight.transpose(1, 2).reshape(
                batch * in_channel, self.out_channel, self.kernel_size, self.kernel_size
            )
            out = F.conv_transpose2d(x, weight, padding=0, stride=2, groups=batch)
            _, _, height, width = out.shape
            out = out.view(batch, self.out_channel, height, width)
            out = self.blur(out)

        else:
            padding = self.padding
            x = x.view(1, batch * in_channel, height, width)
            out = F.conv2d(x, weight, padding=padding, groups=batch)
            _, _, height, width = out.shape
            out = out.view(batch, self.out_channel, height, width)

        return out


class NoiseInjection(nn.Module):
    """Noise Injection module"""

    def __init__(self):
        """Initialize NoiseInjection module"""
        super().__init__()

        self.weight = nn.Parameter(torch.zeros(1))

    def forward(self, image, noise=None):
        """Forward pass"""
        if noise is None:
            batch, _, height, width = image.shape
            noise = image.new_empty(batch, 1, height, width).normal_()

        return image + self.weight * noise


class ConstantInput(nn.Module):
    """Constant Input module"""

    def __init__(self, channel, size=4):
        """Initialize ConstantInput module"""
        super().__init__()

        self.input = nn.Parameter(torch.randn(1, channel, size, size))

    def forward(self, x):
        """Forward pass"""
        batch = x.shape[0]
        out = self.input.repeat(batch, 1, 1, 1)

        return out


class StyledConv(nn.Module):
    """Styled Conv module"""

    def __init__(
        self,
        in_channel,
        out_channel,
        kernel_size,
        style_dim,
        upsample=False,
        blur_kernel=[1, 3, 3, 1],
        demodulate=True,
    ):
        """Initialize StyledConv module"""
        super().__init__()

        self.conv = ModulatedConv2d(
            in_channel,
            out_channel,
            kernel_size,
            style_dim,
            upsample=upsample,
            blur_kernel=blur_kernel,
            demodulate=demodulate,
        )

        self.noise = NoiseInjection()
        # self.bias = nn.Parameter(torch.zeros(1, out_channel, 1, 1))
        # self.activate = ScaledLeakyReLU(0.2)
        self.activate = FusedLeakyReLU(out_channel)

    def forward(self, x, style, noise=None, is_stylespace=False):
        """Forward pass"""
        out = self.conv(x, style, is_stylespace)
        out = self.noise(out, noise=noise)
        # out = out + self.bias
        out = self.activate(out)

        return out


class ToRGB(nn.Module):
    """ToRGB module"""

    def __init__(self, in_channel, style_dim, upsample=True, blur_kernel=[1, 3, 3, 1]):
        """Initialize ToRGB module"""
        super().__init__()

        if upsample:
            self.upsample = Upsample(blur_kernel)

        self.conv = ModulatedConv2d(in_channel, 3, 1, style_dim, demodulate=False)
        self.bias = nn.Parameter(torch.zeros(1, 3, 1, 1))

    def forward(self, x, style, skip=None, is_stylespace=False):
        """Forward pass"""
        out = self.conv(x, style, is_stylespace)
        out = out + self.bias

        if skip is not None:
            skip = self.upsample(skip)

            out = out + skip

        return out


class Generator(nn.Module):
    """Generator module"""

    def __init__(
        self,
        size,
        style_dim,
        channel_multiplier=2,
        blur_kernel=[1, 3, 3, 1],
    ):
        """Initialize Generator"""
        super().__init__()

        self.size = size

        self.style_dim = style_dim

        self.channels = {
            4: 512,
            8: 512,
            16: 512,
            32: 512,
            64: 256 * channel_multiplier,
            128: 128 * channel_multiplier,
            256: 64 * channel_multiplier,
            512: 32 * channel_multiplier,
            1024: 16 * channel_multiplier,
        }

        self.input = ConstantInput(self.channels[4])
        self.conv1 = StyledConv(
            self.channels[4], self.channels[4], 3, style_dim, blur_kernel=blur_kernel
        )
        self.to_rgb1 = ToRGB(self.channels[4], style_dim, upsample=False)

        self.log_size = int(math.log(size, 2))
        self.num_layers = (self.log_size - 2) * 2 + 1

        self.convs = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        self.to_rgbs = nn.ModuleList()
        self.noises = nn.Module()

        in_channel = self.channels[4]

        for layer_idx in range(self.num_layers):
            res = (layer_idx + 5) // 2
            shape = [1, 1, 2**res, 2**res]
            self.noises.register_buffer(f"noise_{layer_idx}", torch.randn(*shape))

        for i in range(3, self.log_size + 1):
            out_channel = self.channels[2**i]

            self.convs.append(
                StyledConv(
                    in_channel,
                    out_channel,
                    3,
                    style_dim,
                    upsample=True,
                    blur_kernel=blur_kernel,
                )
            )

            self.convs.append(
                StyledConv(
                    out_channel, out_channel, 3, style_dim, blur_kernel=blur_kernel
                )
            )

            self.to_rgbs.append(ToRGB(out_channel, style_dim))

            in_channel = out_channel

        self.n_latent = self.log_size * 2 - 2

    def forward(
        self,
        styles,
        return_features=False,
        noise=None,
        is_stylespace=False,
        new_features=None,
        feature_scale=1.0,
        early_stop=None,
    ):
        """Forward pass"""
        to_rgb_stylespace = None

        if noise is None:
            noise = [
                getattr(self.noises, f"noise_{i}") for i in range(self.num_layers)
            ]

        latent = styles[0]

        def insert_feature(x, layer_idx):
            if new_features is not None and new_features[layer_idx] is not None:
                x = (1 - feature_scale) * x + feature_scale * new_features[layer_idx].type_as(x)
            return x

        outs = []
        out = self.input(latent)
        outs.append(out)

        out = self.conv1(out, styles[0].float() if is_stylespace else latent[:, 0], noise=noise[0], is_stylespace=is_stylespace)
        outs.append(out)

        skip = self.to_rgb1(out, to_rgb_stylespace[0].float() if is_stylespace else latent[:, 1], is_stylespace=is_stylespace)

        i = 1
        for conv1, conv2, noise1, noise2, to_rgb in zip(
            self.convs[::2], self.convs[1::2], noise[1::2], noise[2::2], self.to_rgbs
        ):
            out = insert_feature(out, i)
            out = conv1(out, styles[i].float() if is_stylespace else latent[:, i], noise=noise1, is_stylespace=is_stylespace)
            outs.append(out)

            out = insert_feature(out, i + 1)
            out = conv2(out, styles[i + 1].float() if is_stylespace else latent[:, i + 1], noise=noise2, is_stylespace=is_stylespace)
            outs.append(out)

            skip = to_rgb(out, to_rgb_stylespace[i // 2 + 1].float() if is_stylespace else latent[:, i + 2], skip, is_stylespace=is_stylespace)

            if early_stop is not None and skip.size(-1) == early_stop:
                break

            i += 2

        image = skip

        if return_features:
            return image, outs

        return image, None
