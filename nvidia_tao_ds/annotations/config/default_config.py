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

"""Default config file."""

from dataclasses import dataclass
from omegaconf import MISSING
from typing import Optional


@dataclass
class DataConfig:
    """Dataset configuration template."""

    input_format: str = "KITTI"
    output_format: str = "COCO"
    output_dir: str = MISSING


@dataclass
class KITTIConfig:
    """Dataset configuration template."""

    image_dir: str = MISSING
    label_dir: str = MISSING
    project: Optional[str] = None
    mapping: Optional[str] = None


@dataclass
class COCOConfig:
    """Dataset configuration template."""

    ann_file: str = MISSING


@dataclass
class ExperimentConfig:
    """Experiment configuration template."""

    data: DataConfig = DataConfig()
    kitti: KITTIConfig = KITTIConfig()
    coco: COCOConfig = COCOConfig()
    results_dir: Optional[str] = None
