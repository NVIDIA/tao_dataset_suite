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

"""Fused activation modules"""

import os

import torch
from torch import nn
from torch.autograd import Function
from torch.utils.cpp_extension import load

module_path = os.path.dirname(__file__)
fused = load(
    "fused",
    sources=[
        os.path.join(module_path, "fused_bias_act.cpp"),
        os.path.join(module_path, "fused_bias_act_kernel.cu"),
    ],
)


class FusedLeakyReLUFunction(Function):
    """FusedLeakyReLU function"""

    @staticmethod
    def forward(ctx, input, bias, negative_slope, scale):
        """Forward pass"""
        empty = input.new_empty(0)
        out = fused.fused_bias_act(input, bias, empty, 3, 0, negative_slope, scale)
        ctx.save_for_backward(out)
        ctx.negative_slope = negative_slope
        ctx.scale = scale

        return out


class FusedLeakyReLU(nn.Module):
    """FusedLeakyReLU module"""

    def __init__(self, channel, negative_slope=0.2, scale=2**0.5):
        """Initialize FusedLeakyReLU module"""
        super().__init__()

        self.bias = nn.Parameter(torch.zeros(channel))
        self.negative_slope = negative_slope
        self.scale = scale

    def forward(self, x):
        """Forward pass"""
        return fused_leaky_relu(x, self.bias, self.negative_slope, self.scale)


def fused_leaky_relu(x, bias, negative_slope=0.2, scale=2**0.5):
    """Apply FusedLeakyReLU function"""
    return FusedLeakyReLUFunction.apply(x, bias, negative_slope, scale)
