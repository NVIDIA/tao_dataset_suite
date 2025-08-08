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

"""Latent Editor"""

import os

import torch

dirname = os.path.dirname(__file__)


class LatentEditor:
    """Latent space editor"""

    def __init__(self, domain="human_faces"):
        """Initialize LatentEditor"""
        self.domain = domain
        self.dir_prefix = f"{dirname}/../editings/interfacegan_directions"

        if self.domain == "human_faces":
            self.interfacegan_directions = {
                "age": os.path.join(self.dir_prefix, "age.pt"),
                "smile": os.path.join(self.dir_prefix, "smile.pt"),
                "rotation": os.path.join(self.dir_prefix, "rotation.pt"),
                # @seanf: added
                "lum": os.path.join(self.dir_prefix, "lum_skin_boundary.pt"),
                "hue": os.path.join(self.dir_prefix, "hue_skin_boundary.pt")
            }
            self.interfacegan_tensors = {
                name: torch.load(path).cuda()  # is there a discrepancy between .npy and .pt files
                for name, path in self.interfacegan_directions.items()
            }

    def get_interface_gan_edits(self, start_w, factors, direction):
        """Edit input by moving in latent space"""
        latents_to_display = []
        for factor in factors:
            tensor_direction = self.interfacegan_tensors[direction]
            edited_latent = start_w + factor / 2 * tensor_direction  # why divided by 2?
            latents_to_display.append(edited_latent)
        return latents_to_display
