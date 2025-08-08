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

"""Pre-trained model paths"""

from dataclasses import dataclass, fields


models_dir = "/pretrained/"


@dataclass
class DefaultPathsClass:
    """Paths for pre-trained models"""

    e4e_path: str = models_dir + "e4e_ffhq_encode.pt"  # "restyle_e4e_ffhq.pt", "e4e_ffhq_encode.pt"
    stylegan_weights: str = models_dir + "stylegan2-ffhq-config-f.pt"
    arcface_model_path: str = models_dir + "iresnet50-7f187506.pth"

    def __iter__(self):
        """Yield path"""
        for field in fields(self):
            yield field.name, getattr(self, field.name)


DefaultPaths = DefaultPathsClass()
