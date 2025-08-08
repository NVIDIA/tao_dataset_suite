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

"""Base Runner"""

from nvidia_tao_ds.skin_tone.editings.latent_editor import LatentEditor
from nvidia_tao_ds.skin_tone.models.methods import methods_registry


class BaseRunner:
    """Base Runner for editing"""

    def __init__(self, ckpt_path=None):
        """Instantiate BaseRunner"""
        self.ckpt_path = ckpt_path

    def setup(self):
        """Setup env and models"""
        self.device = "cuda:0"
        self.latent_editor = LatentEditor("human_faces")
        self.method = methods_registry["fse_full"](
            checkpoint_path=self.ckpt_path
        ).to(self.device)

    def get_edited_latent(self, original_latent, editing_name, editing_degrees, original_image=None):
        """Edit and return latents"""
        if editing_name in self.latent_editor.interfacegan_directions:
            edited_latents = (
                self.latent_editor.get_interface_gan_edits(
                    original_latent, editing_degrees, editing_name
                ))
        else:
            raise ValueError(f'Edit name {editing_name} is not available')
        return edited_latents
