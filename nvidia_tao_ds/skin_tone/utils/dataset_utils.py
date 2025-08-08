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

"""Dataset utils"""


import os
from PIL import Image
from torch.utils import data
from torchvision import transforms

from nvidia_tao_ds.skin_tone.utils.class_registry import ClassRegistry

transforms_registry = ClassRegistry()


class FaceTransforms(object):
    """Facial Transforms"""

    def __init__(self):
        """Initialize FaceTransforms"""
        super().__init__()
        self.image_size = None

    def get_transforms(self):
        """Returns transform dicts for train and test"""
        transforms_dict = {
            "train": transforms.Compose(
                [
                    transforms.Resize(self.image_size),
                    transforms.RandomHorizontalFlip(0.5),
                    transforms.ToTensor(),
                    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
                ]
            ),
            "test": transforms.Compose(
                [
                    transforms.Resize(self.image_size),
                    transforms.ToTensor(),
                    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
                ]
            )
        }
        return transforms_dict


@transforms_registry.add_to_registry(name="face_256")
class Face256Transforms(FaceTransforms):
    """Facial Transforms for 256x256 images"""

    def __init__(self):
        """Initializes FaceTransforms for 256x256 images"""
        super().__init__()
        self.image_size = (256, 256)


@transforms_registry.add_to_registry(name="face_1024")
class Face1024Transforms(FaceTransforms):
    """Facial Transforms for 1024x1024 images"""

    def __init__(self):
        """Initializes FaceTransforms for 1024x1024 images"""
        super().__init__()
        self.image_size = (1024, 1024)


class DefaultDataset(data.Dataset):
    """Default Dataset for images"""

    CLASSES = ['background', 'skin', 'nose', 'eye_g', 'l_eye', 'r_eye', 'l_brow', 'r_brow', 'l_ear', 'r_ear', 'mouth', 'u_lip', 'l_lip', 'hair', 'hat', 'ear_r', 'neck_l', 'neck', 'cloth']

    def __init__(self, root, transform=None, crop_size=None):
        """Initialize DefaultDataset"""
        self.root = root
        self.transform = transform
        self.crop_size = crop_size

        directory = next(os.walk(self.root))
        self.images = [os.path.join(self.root, file) for file in os.listdir(self.root) if file.endswith('.jpg')]
        # print("imgs: ", self.images)
        # this goes through and gets images from all subdirectories but does not get anything from the root directory itself
        subdirs = directory[1]  # quick trick to get all subdirectories
        for subdir in subdirs:
            curr_images = [os.path.join(self.root, subdir, file) for file in os.listdir(os.path.join(self.root, subdir)) if file.endswith('.jpg')]
            self.images += curr_images

    def __getitem__(self, index):
        """Get transformed image"""
        _img = Image.open(self.images[index]).convert('RGB')
        _img = _img.resize((self.crop_size + 1, self.crop_size + 1), Image.BILINEAR)  # Why 513? Just remove this. Set crop_size to 256?
        _img = preprocess_image(_img)

        if self.transform is not None:
            _img = self.transform(_img)

        return _img

    def __len__(self):
        """Return length of dataset"""
        return len(self.images)


def preprocess_image(image):
    """Normalize image"""
    data_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    image = data_transforms(image)

    return image
