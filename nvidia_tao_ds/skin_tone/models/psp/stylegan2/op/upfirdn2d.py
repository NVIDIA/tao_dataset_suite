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

"""UpFirDn2d modules"""

import os

import torch
from torch.autograd import Function
from torch.utils.cpp_extension import load

module_path = os.path.dirname(__file__)
upfirdn2d_op = load(
    "upfirdn2d",
    sources=[
        os.path.join(module_path, "upfirdn2d.cpp"),
        os.path.join(module_path, "upfirdn2d_kernel.cu"),
    ],
)


class UpFirDn2d(Function):
    """Upsampling function"""

    @staticmethod
    def forward(ctx, x, kernel, up, down, pad):
        """Forward pass"""
        up_x, up_y = up
        down_x, down_y = down
        pad_x0, pad_x1, pad_y0, pad_y1 = pad

        kernel_h, kernel_w = kernel.shape
        _, channel, in_h, in_w = x.shape
        ctx.in_size = x.shape

        x = x.reshape(-1, in_h, in_w, 1)

        ctx.save_for_backward(kernel, torch.flip(kernel, [0, 1]))

        out_h = (in_h * up_y + pad_y0 + pad_y1 - kernel_h) // down_y + 1
        out_w = (in_w * up_x + pad_x0 + pad_x1 - kernel_w) // down_x + 1
        ctx.out_size = (out_h, out_w)

        ctx.up = (up_x, up_y)
        ctx.down = (down_x, down_y)
        ctx.pad = (pad_x0, pad_x1, pad_y0, pad_y1)

        g_pad_x0 = kernel_w - pad_x0 - 1
        g_pad_y0 = kernel_h - pad_y0 - 1
        g_pad_x1 = in_w * up_x - out_w * down_x + pad_x0 - up_x + 1
        g_pad_y1 = in_h * up_y - out_h * down_y + pad_y0 - up_y + 1

        ctx.g_pad = (g_pad_x0, g_pad_x1, g_pad_y0, g_pad_y1)

        out = upfirdn2d_op.upfirdn2d(
            x, kernel, up_x, up_y, down_x, down_y, pad_x0, pad_x1, pad_y0, pad_y1
        )
        # out = out.view(major, out_h, out_w, minor)
        out = out.view(-1, channel, out_h, out_w)

        return out


def upfirdn2d(input, kernel, up=1, down=1, pad=(0, 0)):
    """Apply upsample function"""
    out = UpFirDn2d.apply(
        input, kernel, (up, up), (down, down), (pad[0], pad[1], pad[0], pad[1])
    )

    return out
