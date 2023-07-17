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

"""Utilities to handle different data formats."""


class DataFormat:
    """Class data format"""

    pass


class CocoData(DataFormat):
    """COCO data format"""

    def __init__(self, ann_file, image_dir=None) -> None:
        """Initialize a COCO object.

        Args:
            ann_file(str): path to annotation file.
            image_dir(str): path to image dir.

        """
        self.ann_file = ann_file
        self.image_dir = image_dir


class KittiData(DataFormat):
    """KITTI data format"""

    def __init__(self, image_dir=None, label_dir=None) -> None:
        """Initialize a KITTI object.

        Args:
            image_dir(str): path to image dir.
            label_dir(str): path to label dir.

        """
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.image_paths = None
        self.label_paths = None
