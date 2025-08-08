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

"""Interface Runner"""

import torch
import torch.nn.functional as F

from nvidia_tao_ds.skin_tone.utils.class_registry import ClassRegistry
from nvidia_tao_ds.skin_tone.runners.base_runner import BaseRunner


inference_runner_registry = ClassRegistry()


@inference_runner_registry.add_to_registry(name="fse_inference_runner")
class FSEInferenceRunner(BaseRunner):
    """Inference Runner for editing in latent space"""

    def _run_on_batch(self, inputs):
        """Encode images in latent space"""
        images, w_recon, fused_feat, predicted_feat = self.method(inputs, return_latents=True)

        x = F.interpolate(inputs, size=(256, 256), mode="bilinear", align_corners=False)
        w_e4e = self.method.e4e_encoder(x)
        w_e4e = w_e4e + self.method.latent_avg

        result_batch = {
            "latents": w_recon,
            "fused_feat": fused_feat,
            "predicted_feat": predicted_feat,
            "w_e4e": w_e4e,
            "inputs": inputs.cpu()
        }

        return images, result_batch

    def _run_editing_on_batch(self, method_res_batch, editing_name, editing_degrees, mask=None):
        """Edit batch of images"""
        orig_latents = method_res_batch["latents"]

        edited_images = []
        n_iter = 1e5

        for i, latent in enumerate(orig_latents):
            edited_latents = self.get_edited_latent(
                latent.unsqueeze(0),
                editing_name,
                editing_degrees,
                method_res_batch["inputs"][i].unsqueeze(0)
            )

            w_e4e = method_res_batch["w_e4e"][i].unsqueeze(0)
            edited_w_e4e = self.get_edited_latent(
                w_e4e,
                editing_name,
                editing_degrees,
                method_res_batch["inputs"][i].unsqueeze(0)
            )

            is_stylespace = isinstance(edited_latents, tuple)
            if not is_stylespace:
                edited_latents = torch.cat(edited_latents, dim=0).unsqueeze(0)
                edited_w_e4e = torch.cat(edited_w_e4e, dim=0).unsqueeze(0)

            w_e4e = w_e4e.repeat(len(editing_degrees), 1, 1)  # bs = len(editing_degrees)
            # w_latent = latent.unsqueeze(0).repeat(len(editing_degrees), 1, 1)

            _, fs_x = self.method.decoder(
                [w_e4e],
                return_features=True,
                early_stop=64
            )

            _, fs_y = self.method.decoder(
                edited_w_e4e,
                is_stylespace=is_stylespace,
                return_features=True,
                early_stop=64
            )

            delta = fs_x[9] - fs_y[9]

            fused_feat = method_res_batch["fused_feat"][i].to(self.device)
            fused_feat = fused_feat.repeat(len(editing_degrees), 1, 1, 1)

            if mask is not None:
                delta_mask = mask[i][0].unsqueeze(0).repeat(512, 1, 1).unsqueeze(0)  # add an additional dimension to delta_mask
                # this just smoothens the edges and then transforms it to be size 64 x 64 from 512 x 512
                delta_mask = F.interpolate(delta_mask, size=(64, 64), mode="bilinear", align_corners=True)
                delta = delta * (1 - delta_mask)  # zero out everything that needs to be zeroed out

            edited_feat = fused_feat

            edit_features = [None] * 9 + [edited_feat] + [None] * (17 - 9)

            image_edits, _ = self.method.decoder(
                edited_latents,
                new_features=edit_features,
                feature_scale=min(1.0, 0.0001 * n_iter),
                is_stylespace=is_stylespace,
            )

            edited_images.append(image_edits)
        edited_images = torch.stack(edited_images)

        return edited_images  # : torch.tensor(batch_size x len(editing_degrees) x 1024 x 1024)
