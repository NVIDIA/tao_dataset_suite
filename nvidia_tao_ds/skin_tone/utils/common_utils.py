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

"""Common utils"""

from PIL import Image


def tensor2im(var):
    """Tensor to int8 image"""
    var = var.cpu().detach().transpose(0, 2).transpose(0, 1).numpy()
    var = (var + 1) / 2
    var[var < 0] = 0
    var[var > 1] = 1
    var = var * 255
    return Image.fromarray(var.astype("uint8"))


def get_keys(d, name, key="state_dict"):
    """Get keys in state dict"""
    if key in d:
        d = d[key]
    d_filt = {k[len(name) + 1:]: v for k, v in d.items() if k[: len(name) + 1] == name + '.'}
    return d_filt


def toogle_grad(model, flag=True):
    """Turn gradients on"""
    for p in model.parameters():
        p.requires_grad = flag
