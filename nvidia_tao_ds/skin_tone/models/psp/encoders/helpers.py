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

"""PSP model building blocks"""

from collections import namedtuple
import torch.nn.functional as F
from torch.nn import (
    Conv2d,
    BatchNorm2d,
    PReLU,
    ReLU,
    Sigmoid,
    MaxPool2d,
    AdaptiveAvgPool2d,
    Sequential,
    Module,
)

"""
ArcFace implementation from [TreB1eN](https://github.com/TreB1eN/InsightFace_Pytorch)
"""


class Bottleneck(namedtuple("Block", ["in_channel", "depth", "stride"])):
    """A named tuple describing a ResNet block."""


def get_block(in_channel, depth, num_units, stride=2):
    """Create Bottleneck block"""
    return [Bottleneck(in_channel, depth, stride)] + [
        Bottleneck(depth, depth, 1) for i in range(num_units - 1)
    ]


def get_blocks(num_layers):
    """Instantiates blocks given num_layers

    Args:
        num_layers (int): Number of layers

    Returns:
        list: Bottleneck blocks
    """
    if num_layers == 50:
        blocks = [
            get_block(in_channel=64, depth=64, num_units=3),
            get_block(in_channel=64, depth=128, num_units=4),
            get_block(in_channel=128, depth=256, num_units=14),
            get_block(in_channel=256, depth=512, num_units=3),
        ]
    elif num_layers == 100:
        blocks = [
            get_block(in_channel=64, depth=64, num_units=3),
            get_block(in_channel=64, depth=128, num_units=13),
            get_block(in_channel=128, depth=256, num_units=30),
            get_block(in_channel=256, depth=512, num_units=3),
        ]
    elif num_layers == 152:
        blocks = [
            get_block(in_channel=64, depth=64, num_units=3),
            get_block(in_channel=64, depth=128, num_units=8),
            get_block(in_channel=128, depth=256, num_units=36),
            get_block(in_channel=256, depth=512, num_units=3),
        ]
    else:
        raise ValueError(
            f"Invalid number of layers: {num_layers}. Must be one of [50, 100, 152]"
        )
    return blocks


class SEModule(Module):
    """SEModule"""

    def __init__(self, channels, reduction):
        """Initialize SEModule

        Args:
            channels (int): Convolution layer channels
            reduction (int): Factor to reduce number of channels by
        """
        super().__init__()
        self.avg_pool = AdaptiveAvgPool2d(1)
        self.fc1 = Conv2d(
            channels, channels // reduction, kernel_size=1, padding=0, bias=False
        )
        self.relu = ReLU(inplace=True)
        self.fc2 = Conv2d(
            channels // reduction, channels, kernel_size=1, padding=0, bias=False
        )
        self.sigmoid = Sigmoid()

    def forward(self, x):
        """Forward pass"""
        module_input = x
        x = self.avg_pool(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return module_input * x


class bottleneck_IR_SE(Module):
    """Bottleneck_IR_SE module"""

    def __init__(self, in_channel, depth, stride):
        """Initialize bottleneck_IR_SE

        Args:
            in_channel (int): Convolution input channels
            depth (int): Convolution middle channels
            stride (int): Convolution stride
        """
        super().__init__()
        if in_channel == depth:
            self.shortcut_layer = MaxPool2d(1, stride)
        else:
            self.shortcut_layer = Sequential(
                Conv2d(in_channel, depth, (1, 1), stride, bias=False),
                BatchNorm2d(depth),
            )
        self.res_layer = Sequential(
            BatchNorm2d(in_channel),
            Conv2d(in_channel, depth, (3, 3), (1, 1), 1, bias=False),
            PReLU(depth),
            Conv2d(depth, depth, (3, 3), stride, 1, bias=False),
            BatchNorm2d(depth),
            SEModule(depth, 16),
        )

    def forward(self, x):
        """Forward pass"""
        shortcut = self.shortcut_layer(x)
        res = self.res_layer(x)
        return res + shortcut


def _upsample_add(x, y):
    """Upsample and add two feature maps.
    Args:
      x: (Variable) top feature map to be upsampled.
      y: (Variable) lateral feature map.
    Returns:
      (Variable) added feature map.
    Note in PyTorch, when input size is odd, the upsampled feature map
    with `F.upsample(..., scale_factor=2, mode='nearest')`
    maybe not equal to the lateral feature map size.
    e.g.
    original input size: [N,_,15,15] ->
    conv2d feature map size: [N,_,8,8] ->
    upsampled feature map size: [N,_,16,16]
    So we choose bilinear upsample which supports arbitrary output sizes.
    """
    _, _, H, W = y.size()
    return F.interpolate(x, size=(H, W), mode="bilinear", align_corners=True) + y
