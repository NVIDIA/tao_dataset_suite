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

"""Mapping among different annotation format."""

from nvidia_tao_ds.annotations.conversion.kitti_to_coco import convert_kitti_to_coco
from nvidia_tao_ds.annotations.conversion.coco_to_kitti import convert_coco_to_kitti
from nvidia_tao_ds.annotations.conversion.coco_to_odvg import convert_coco_to_odvg
from nvidia_tao_ds.annotations.conversion.coco_to_contiguous import convert_coco_to_contiguous
from nvidia_tao_ds.annotations.conversion.odvg_to_coco import convert_odvg_to_coco
from nvidia_tao_ds.annotations.conversion.aicity_to_ovpkl import convert_aicity_to_ovpkl


CONVERSION_MAPPING = {
    "coco": {
        "kitti": convert_coco_to_kitti,
        "odvg": convert_coco_to_odvg,
        "coco": convert_coco_to_contiguous,
    },
    "kitti": {
        "coco": convert_kitti_to_coco
    },
    "odvg": {
        "coco": convert_odvg_to_coco
    },
    "aicity": {
        "ovpkl": convert_aicity_to_ovpkl
    }
}
