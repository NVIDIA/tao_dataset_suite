# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
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

"""Utility functions for TAO augment."""

from nvidia_tao_ds.augmentation.dataloader.coco_callable import CocoInputCallable
from nvidia_tao_ds.augmentation.dataloader.kitti_callable import KittiInputCallable
from nvidia_tao_ds.augmentation.pipeline.sharded_pipeline import (
    build_coco_pipeline,
    build_kitti_pipeline,
)

callable_dict = {
    'kitti': KittiInputCallable,
    'coco': CocoInputCallable
}

pipeline_dict = {
    'kitti': build_kitti_pipeline,
    'coco': build_coco_pipeline
}
