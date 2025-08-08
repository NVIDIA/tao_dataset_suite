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

"""Simple Runner"""

import torch.nn.functional as F
from torchvision import transforms
import numpy as np

from PIL import Image
from pathlib import Path

from nvidia_tao_ds.skin_tone.utils.common_utils import tensor2im
from nvidia_tao_ds.skin_tone.utils.dataset_utils import transforms_registry
from nvidia_tao_ds.skin_tone.runners.inference_runners import FSEInferenceRunner

import cv2


class SimpleRunner:
    """Simple Runner for editing"""

    def __init__(
        self,
        editor_ckpt_pth: str,
    ):
        """Initializes SimpleRunner

        Args:
            editor_ckpt_pth (str): Path to pre-trained editor
        """
        self.inference_runner = FSEInferenceRunner(editor_ckpt_pth)
        self.inference_runner.setup()
        self.inference_runner.method.eval()
        self.inference_runner.method.decoder = self.inference_runner.method.decoder.float()

    def edit(
        self,
        orig_img_pth: str,
        editing_name: str,
        edited_power: float,
        save_pth: str,
        use_mask: bool = False,
        mask_path: str = None,
    ):
        """Edit image

        Args:
            orig_img_pth (str): Path to input image
            editing_name (str): Attribute to edit
            edited_power (float): Strength of edit (move further in latent space)
            save_pth (str): Output image path
            use_mask (bool, optional): Only edit masked part of image. Defaults to False.
            mask_path (str, optional): Path to mask. Defaults to None.
        """
        save_pth = Path(save_pth)
        save_pth_dir = save_pth.parents[0]
        save_pth_dir.mkdir(parents=True, exist_ok=True)

        if use_mask and mask_path is not None:
            # print(f"Use mask from {mask_path}")
            mask = Image.open(mask_path).convert("RGB")
            lips = np.isin(np.asarray(mask), [11, 12])
            brows = np.isin(np.asarray(mask), [6, 7])
            mask = np.isin(np.asarray(mask), [1, 2, 8, 9, 15, 16, 17]).astype(np.float32)
            mask[lips] = 0.2
            mask[brows] = 0.3

            # TODO add a switch here, if the mask was generated use this:
            # mask = np.abs(255 - np.array(mask))

            transform = transforms.ToTensor()
            mask = cv2.GaussianBlur(mask, (15, 15), 10)  # smoothed

            mask = transform(mask)
            mask = mask.unsqueeze(0).to(self.inference_runner.device)

        orig_img = Image.open(orig_img_pth).convert("RGB")
        transform_dict = transforms_registry["face_1024"]().get_transforms()  # changed from 1024, check /datasets/transforms.py
        orig_img = transform_dict["test"](orig_img).unsqueeze(0)

        device = self.inference_runner.device
        inv_images, inversion_results = self.inference_runner._run_on_batch(orig_img.to(device))
        edited_image = self.inference_runner._run_editing_on_batch(
            method_res_batch=inversion_results,
            editing_name=editing_name,
            editing_degrees=[edited_power],
            mask=mask,
        )

        mask = F.interpolate(mask, size=(inv_images[0].shape[1], inv_images[0].shape[2]), mode="bilinear", align_corners=True)

        edited_image[0][0] = mask * edited_image[0][0] + (1 - mask) * inv_images[0]  # overlay the image
        # TODO: we'd need to do this for each image

        edited_image = tensor2im(edited_image[0][0].cpu())
        edited_image.save(save_pth)
        return edited_image

    def available_editings(self):
        """Return editable attributes"""
        edits_types = []
        for field in dir(self.inference_runner.latent_editor):
            if "directions" in field.split("_"):
                edits_types.append(field)

        print("This code handles the following editing directions for following methods:")
        # available_directions = {}
        for edit_type in edits_types:
            print(edit_type + ":")
            edit_type_directions = getattr(self.inference_runner.latent_editor, edit_type, None).keys()
            for direction in edit_type_directions:
                print("\t" + direction)
        print(GLOBAL_DIRECTIONS_DESC)


GLOBAL_DIRECTIONS_DESC = """
You can alse use directions from text prompts via StyleClip Global Mapper (https://arxiv.org/abs/2103.17249).
Such directions look as follows: "styleclip_global_{neutral prompt}_{target prompt}_{disentanglement}" where
neutral prompt -- some neutral description of the original image (e.g. "a face")
target prompt -- text that contains the desired edit (e.g. "a smilling face")
disentanglement -- positive number, the more this attribute - the more related attributes will also be changed (e.g.
for grey hair editing, wrinkle, skin colour and glasses may also be edited)

Example: "styleclip_global_face with hair_face with black hair_0.18"

More information about the purpose of directions and their approximate power range can be found in available_directions.txt.
"""
