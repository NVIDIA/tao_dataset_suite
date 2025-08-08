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

"""PSP encoders"""

from enum import Enum
import math
import numpy as np

import torch
from torch import nn
from torch.nn import Conv2d, BatchNorm2d, PReLU, Sequential, Module

from nvidia_tao_ds.skin_tone.models.psp.encoders.feature_resnet import iresnet50
from nvidia_tao_ds.skin_tone.models.psp.encoders.helpers import (
    get_blocks,
    bottleneck_IR_SE,
    _upsample_add
)
from nvidia_tao_ds.skin_tone.models.psp.stylegan2.model import EqualLinear


class Inverter(nn.Module):
    """Inverter module"""

    def __init__(self, n_styles=18, opts=None):
        """Initialize Inverter"""
        super().__init__()

        self.fs_backbone = FSLikeBackbone(opts=opts, n_styles=n_styles)
        self.fuser = ContentLayerDeepFast(6, 1024, 512)


class FSLikeBackbone(nn.Module):
    """FSLikeBackbone module"""

    def __init__(self, n_styles=18, opts=None):
        """Initialize FSLikeBackbone"""
        super().__init__()

        resnet50 = iresnet50()
        resnet50.load_state_dict(torch.load(opts.arcface_model_path))

        self.conv = nn.Sequential(*list(resnet50.children())[:3])

        # define layers
        self.block_1 = list(resnet50.children())[3]  # 15-18
        self.block_2 = list(resnet50.children())[4]  # 10-14
        self.block_3 = list(resnet50.children())[5]  # 5-9
        self.block_4 = list(resnet50.children())[6]  # 1-4

        # replace stride in conv to increase predicted dimensionality
        state = self.block_3.state_dict()
        self.block_3[0].conv2 = torch.nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.block_3[0].downsample[0] = torch.nn.Conv2d(128, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.block_3.load_state_dict(state, strict=True)

        self.content_layer = nn.Sequential(
            nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.PReLU(num_parameters=512),
            nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )

        self.avg_pool = nn.AdaptiveAvgPool2d((3, 3))

        self.styles = nn.ModuleList()
        for _ in range(n_styles):
            self.styles.append(nn.Linear(960 * 9, 512))

    def apply_head(self, x):
        """Apply Linear head layers"""
        latents = []
        for i in range(len(self.styles)):
            latents.append(self.styles[i](x))
        out = torch.stack(latents, dim=1)
        return out

    def forward(self, x):
        """Forward pass"""
        features = []

        x = self.conv(x)
        x = self.block_1(x)
        features.append(self.avg_pool(x))
        x = self.block_2(x)
        features.append(self.avg_pool(x))
        x = self.block_3(x)
        features.append(self.avg_pool(x))

        content = self.content_layer(x)
        x = self.block_4(x)
        features.append(self.avg_pool(x))

        x = torch.cat(features, dim=1)
        x = x.view(x.size(0), -1)

        return self.apply_head(x), content


class ContentLayerDeepFast(nn.Module):
    """ContentLayerDeepFast module"""

    def __init__(self, length=4, inp=1024, out=512):
        """Initialize ContentLayerDeepFast"""
        super().__init__()
        self.body = nn.ModuleList([])
        self.conv = nn.Conv2d(inp, out, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0))
        for _ in range(length - 1):
            self.body.append(FeatureEncoderBlock(out, out))

    def forward(self, x):
        """Forward pass"""
        x = self.conv(x)
        for _, block in enumerate(self.body):
            x = x + block(x)
        return x


class FeatureEncoderBlock(nn.Module):
    """FeatureEncoderBlock module"""

    def __init__(self, inp=1024, out=512):
        """Initialize FetaureEncoderBlock"""
        super().__init__()
        self.body = nn.Sequential(
            nn.BatchNorm2d(inp, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.Conv2d(inp, out, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            nn.BatchNorm2d(out, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.PReLU(num_parameters=out),
            nn.Conv2d(out, out, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            nn.BatchNorm2d(out, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )

    def forward(self, x):
        """Forward pass"""
        return self.body(x)


class GradualStyleBlock(Module):
    """GradualStyleBlock"""

    def __init__(self, in_c, out_c, spatial, norm=False):
        """Initialize GradualStyleBlock"""
        super().__init__()
        self.out_c = out_c
        self.spatial = spatial
        self.norm = norm
        num_pools = int(np.log2(spatial))
        modules = []
        modules += [
            Conv2d(in_c, out_c, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(),
        ]
        for _ in range(num_pools - 1):
            modules += [
                Conv2d(out_c, out_c, kernel_size=3, stride=2, padding=1),
                nn.LeakyReLU(),
            ]
        self.convs = nn.Sequential(*modules)
        self.linear = EqualLinear(out_c, out_c, lr_mul=1)

    def forward(self, x):
        """Forward pass"""
        x = self.convs(x)
        x = x.view(-1, self.out_c)
        x = self.linear(x)
        return x


class ProgressiveStage(Enum):
    """Stages of model"""

    WTraining = 0
    Delta1Training = 1
    Delta2Training = 2
    Delta3Training = 3
    Delta4Training = 4
    Delta5Training = 5
    Delta6Training = 6
    Delta7Training = 7
    Delta8Training = 8
    Delta9Training = 9
    Delta10Training = 10
    Delta11Training = 11
    Delta12Training = 12
    Delta13Training = 13
    Delta14Training = 14
    Delta15Training = 15
    Delta16Training = 16
    Delta17Training = 17
    Inference = 18


class Encoder4Editing(Module):
    """Encoder4Editing (e4e) module"""

    def __init__(self, num_layers, mode="ir", opts=None):
        """Initialize Encoder4Editing (e4e)"""
        super().__init__()
        assert num_layers in [50, 100, 152], "num_layers should be 50,100, or 152"
        assert mode in ["ir", "ir_se"], "mode should be ir or ir_se"
        blocks = get_blocks(num_layers)
        unit_module = None
        if mode == "ir_se":
            unit_module = bottleneck_IR_SE
        self.input_layer = Sequential(
            Conv2d(3, 64, (3, 3), 1, 1, bias=False), BatchNorm2d(64), PReLU(64)
        )
        modules = []
        for block in blocks:
            for bottleneck in block:
                modules.append(
                    unit_module(
                        bottleneck.in_channel, bottleneck.depth, bottleneck.stride
                    )
                )
        self.body = Sequential(*modules)

        self.styles = nn.ModuleList()
        log_size = int(math.log(opts.stylegan_size, 2))
        self.style_count = 2 * log_size - 2
        self.coarse_ind = 3
        self.middle_ind = 7

        for i in range(self.style_count):
            if i < self.coarse_ind:
                style = GradualStyleBlock(512, 512, 16)
            elif i < self.middle_ind:
                style = GradualStyleBlock(512, 512, 32)
            else:
                style = GradualStyleBlock(512, 512, 64)
            self.styles.append(style)

        self.latlayer1 = nn.Conv2d(256, 512, kernel_size=1, stride=1, padding=0)
        self.latlayer2 = nn.Conv2d(128, 512, kernel_size=1, stride=1, padding=0)

        self.progressive_stage = ProgressiveStage.Inference

    def forward(self, x, return_feat=False):
        """Forward pass"""
        x = self.input_layer(x)

        modulelist = list(self.body._modules.values())
        c1, c2, c3 = None, None, None
        for i, l in enumerate(modulelist):
            x = l(x)
            if i == 6:
                c1 = x
            elif i == 20:
                c2 = x
            elif i == 23:
                c3 = x

        # Infer main W and duplicate it
        w0 = self.styles[0](c3)
        w = w0.repeat(self.style_count, 1, 1).permute(1, 0, 2)
        stage = self.progressive_stage.value
        features = c3
        for i in range(1, min(stage + 1, self.style_count)):  # Infer additional deltas
            if i == self.coarse_ind:
                p2 = _upsample_add(c3, self.latlayer1(c2))  # FPN's middle features
                features = p2
            elif i == self.middle_ind:
                p1 = _upsample_add(p2, self.latlayer2(c1))  # FPN's fine features
                features = p1
            delta_i = self.styles[i](features)
            w[:, i] += delta_i
        if return_feat:
            return w, features
        return w
