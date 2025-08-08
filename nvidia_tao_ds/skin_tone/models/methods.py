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

"""FSE Full Method"""

import torch
import torch.nn.functional as F
from torch import nn

from nvidia_tao_ds.skin_tone.models.psp.encoders import psp_encoders
from nvidia_tao_ds.skin_tone.models.psp.stylegan2.model import Generator
from nvidia_tao_ds.skin_tone.utils.class_registry import ClassRegistry
from nvidia_tao_ds.skin_tone.utils.common_utils import get_keys, toogle_grad
from nvidia_tao_ds.skin_tone.utils.paths import DefaultPaths
from argparse import Namespace


methods_registry = ClassRegistry()


@methods_registry.add_to_registry("fse_full", stop_args=("self", "checkpoint_path"))
class FSEFull(nn.Module):
    """FSEFull module"""

    def __init__(self,
                 device="cuda:0",
                 checkpoint_path=None):
        """Initialize FSEFull

        Args:
            device (str): Device for module
            checkpoint_path (str): Path to checkpoitn weights
        """
        super().__init__()
        self.opts = {
            "device": device,
            "checkpoint_path": checkpoint_path,
            "stylegan_size": 1024
        }
        self.opts.update(DefaultPaths)
        self.opts = Namespace(**self.opts)

        self.device = device

        self.encoder = self.set_encoder()
        self.decoder = Generator(self.opts.stylegan_size, 512)
        self.latent_avg = None

        self.pool = torch.nn.AdaptiveAvgPool2d((256, 256))
        self.load_weights()

    def set_encoder(self):
        """Instantiates and returns PSP encoder"""
        self.inverter = psp_encoders.Inverter(opts=self.opts, n_styles=18)
        self.e4e_encoder = psp_encoders.Encoder4Editing(50, "ir_se", self.opts)
        feat_editor = psp_encoders.ContentLayerDeepFast(6, 1024, 512)
        return feat_editor  # trainable part

    def load_weights(self):
        """Load pre-trained weights into encoder, inverter, decoder, and e4e_encoder"""
        if self.opts.checkpoint_path != "":
            print(f"Loading from checkpoint: {self.opts.checkpoint_path}")
            ckpt = torch.load(self.opts.checkpoint_path, map_location="cpu")
            self.encoder.load_state_dict(get_keys(ckpt, "encoder"), strict=True)
            self.inverter.load_state_dict(get_keys(ckpt, "inverter"), strict=True)

        self.inverter = self.inverter.eval().to(self.device)
        toogle_grad(self.inverter, False)

        print("Loading Decoder from", self.opts.stylegan_weights)
        ckpt = torch.load(self.opts.stylegan_weights)
        self.decoder.load_state_dict(ckpt["g_ema"], strict=False)
        self.latent_avg = ckpt['latent_avg'].to(self.device)
        self.decoder = self.decoder.eval().to(self.device)
        toogle_grad(self.decoder, False)

        print("Loading E4E from", self.opts.e4e_path)
        ckpt = torch.load(self.opts.e4e_path, map_location="cpu")
        self.e4e_encoder.load_state_dict(get_keys(ckpt, "encoder"), strict=True)
        self.e4e_encoder = self.e4e_encoder.eval().to(self.device)
        toogle_grad(self.e4e_encoder, False)

    def forward(self, x, return_latents=False, n_iter=1e5):
        """Forward pass"""
        x = F.interpolate(x, size=(256, 256), mode="bilinear", align_corners=False)

        with torch.no_grad():
            w_recon, predicted_feat = self.inverter.fs_backbone(x)
            w_recon = w_recon + self.latent_avg

            _, w_feats = self.decoder(
                [w_recon],
                return_features=True,
                is_stylespace=False,
                early_stop=64
            )

            w_feat = w_feats[9]  # bs x 512 x 64 x 64

            fused_feat = self.inverter.fuser(torch.cat([predicted_feat, w_feat], dim=1))
            delta = torch.zeros_like(fused_feat)  # inversion case

        edited_feat = self.encoder(torch.cat([fused_feat, delta], dim=1))
        feats = [None] * 9 + [edited_feat] + [None] * (17 - 9)

        images, _ = self.decoder(
            [w_recon],
            return_features=True,
            new_features=feats,
            feature_scale=min(1.0, 0.0001 * n_iter),
            is_stylespace=False,
        )

        if return_latents:
            if not self.encoder.training:
                fused_feat = fused_feat.cpu()
                predicted_feat = predicted_feat.cpu()
            return images, w_recon, fused_feat, predicted_feat
        return images
