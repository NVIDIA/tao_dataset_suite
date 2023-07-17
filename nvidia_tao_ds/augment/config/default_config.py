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

from typing import List, Optional, Union
from dataclasses import dataclass, field
from omegaconf import MISSING


@dataclass
class RotationAIConfig:
    """Rotation configuration template."""

    enabled: bool = False
    gt_cache: str = ''


@dataclass
class RotationConfig:
    """Rotation configuration template."""

    angle: List[float] = field(default_factory=lambda: [0])
    units: str = "degrees"
    refine_box: RotationAIConfig = RotationAIConfig()


@dataclass
class ShearConfig:
    """Rotation configuration template."""

    shear_ratio_x: List[float] = field(default_factory=lambda: [0])
    shear_ratio_y: List[float] = field(default_factory=lambda: [0])


@dataclass
class FlipConfig:
    """Flip configuration template."""

    flip_horizontal: bool = False
    flip_vertical: bool = False


@dataclass
class TranslationConfig:
    """Translation configuration template."""

    translate_x: List[int] = field(default_factory=lambda: [0])
    translate_y: List[int] = field(default_factory=lambda: [0])


@dataclass
class HueConfig:
    """Hue configuration template."""

    hue_rotation_angle: List[float] = field(default_factory=lambda: [0])


@dataclass
class SaturationConfig:
    """Saturation configuration template."""

    saturation_shift: List[float] = field(default_factory=lambda: [1])


@dataclass
class ContrastConfig:
    """Contrast configuration template."""

    contrast: List[float] = field(default_factory=lambda: [0])
    center: List[float] = field(default_factory=lambda: [127])


@dataclass
class BrightnessConfig:
    """Contrast configuration template."""

    offset: List[float] = field(default_factory=lambda: [0])


@dataclass
class SpatialAugmentationConfig:
    """Spatial augmentation configuration template."""

    rotation: RotationConfig = RotationConfig()
    shear: ShearConfig = ShearConfig()
    flip: FlipConfig = FlipConfig()
    translation: TranslationConfig = TranslationConfig()


@dataclass
class ColorAugmentationConfig:
    """Color augmentation configuration template."""

    hue: HueConfig = HueConfig()
    saturation: SaturationConfig = SaturationConfig()
    contrast: ContrastConfig = ContrastConfig()
    brightness: BrightnessConfig = BrightnessConfig()


@dataclass
class KernelFilterConfig:
    """Blur configuration template."""

    std: List[float] = field(default_factory=lambda: [0.1])
    size: List[float] = field(default_factory=lambda: [3])


@dataclass
class DataConfig:
    """Dataset configuration template."""

    dataset_type: str = 'coco'
    output_image_width: Union[int, None] = None
    output_image_height: Union[int, None] = None
    image_dir: str = MISSING
    ann_path: str = MISSING
    output_dataset: str = MISSING
    batch_size: int = 8
    include_masks: bool = False


@dataclass
class AugmentConfig:
    """Experiment configuration template."""

    random_seed: int = 42
    num_gpus: int = 1
    gpu_ids: List[int] = field(default_factory=lambda: [0])
    data: DataConfig = DataConfig()
    color_aug: ColorAugmentationConfig = ColorAugmentationConfig()
    spatial_aug: SpatialAugmentationConfig = SpatialAugmentationConfig()
    blur_aug: KernelFilterConfig = KernelFilterConfig()
    results_dir: Optional[str] = None
