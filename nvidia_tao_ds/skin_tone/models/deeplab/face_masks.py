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

"""Generate DeepLab face masks"""

import os
import numpy as np
import torch
from PIL import Image

from nvidia_tao_ds.skin_tone.models.deeplab import deeplab
from nvidia_tao_ds.skin_tone.utils.dataset_utils import DefaultDataset


def get_face_masks(cfg):
    """Generate face masks"""
    assert torch.cuda.is_available()
    torch.backends.cudnn.benchmark = True
    model_fname = '/pretrained/deeplab_model.pth'
    # add this as an argument
    assert os.path.isdir(cfg.dataset.input_dir)
    mask_dir = os.path.join(cfg.results_dir, 'masks')
    os.makedirs(mask_dir, exist_ok=True)
    dataset = DefaultDataset(cfg.dataset.input_dir, crop_size=cfg.dataset.image_size)

    model = getattr(deeplab, 'resnet101')(
        pretrained=True,
        num_classes=len(dataset.CLASSES),
        num_groups=32,
        weight_std=True,
        beta=False)

    model = model.cuda()
    model.eval()

    checkpoint = torch.load(model_fname)
    state_dict = {k[7:]: v for k, v in checkpoint['state_dict'].items() if 'tracked' not in k}
    model.load_state_dict(state_dict)
    for i in range(len(dataset)):
        inputs = dataset[i]
        inputs = inputs.cuda()
        outputs = model(inputs.unsqueeze(0))
        _, pred = torch.max(outputs, 1)
        pred = pred.data.cpu().numpy().squeeze().astype(np.uint8)
        imname = os.path.basename(dataset.images[i])
        mask_pred = Image.fromarray(pred)
        mask_pred = mask_pred.resize((cfg.dataset.image_size, cfg.dataset.image_size), Image.NEAREST)
        # TODO: add something here so that if it doesn't detect a face, it'll just move on and no save something empty
        # can we in the future just use some form of skin segmentation?
        # try:
        #     mask_pred.save(dataset.images[i].replace(imname, 'parsings/' + imname[:-4]+'.png'))
        # except FileNotFoundError:
        #     os.makedirs(os.path.join(os.path.dirname(dataset.images[i]), 'parsings'))
        #     mask_pred.save(dataset.images[i].replace(imname, 'parsings/' + imname[:-4]+'.png'))
        mask_pred.save(os.path.join(mask_dir, imname[:-4] + '.jpg'))
    return mask_dir
