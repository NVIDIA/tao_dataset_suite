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
from typing import Optional
from omegaconf import MISSING

from nvidia_tao_ds.config_utils.default_config_utils import (
    STR_FIELD,
    BOOL_FIELD,
    DATACLASS_FIELD,
)


@dataclass
class DataConfig:
    """Dataset configuration template."""

    input_format: str = STR_FIELD(value="KITTI")
    output_format: str = STR_FIELD(value="COCO")


@dataclass
class KITTIConfig:
    """Dataset configuration template."""

    image_dir: str = STR_FIELD(value=MISSING, default_value="<specify image directory>")
    label_dir: str = STR_FIELD(
        value=MISSING, default_value="<specify labels directory>"
    )
    project: Optional[str] = STR_FIELD(None, default_value="annotations")
    mapping: Optional[str] = STR_FIELD(None)
    no_skip: bool = BOOL_FIELD(value=False)
    preserve_hierarchy: bool = BOOL_FIELD(value=False)


@dataclass
class COCOConfig:
    """Dataset configuration template."""

    ann_file: str = STR_FIELD(
        value=MISSING, default_value="<specify path to annotation file>"
    )
    refine_box: bool = BOOL_FIELD(value=False)
    use_all_categories: bool = BOOL_FIELD(value=False)
    add_background: bool = BOOL_FIELD(
        value=False,
        default_value=False,
        description="Flag to add background to the class list, so as to make other classes, 1-indexed.",
    )


@dataclass
class ODVGConfig:
    """Dataset configuration template."""

    ann_file: str = STR_FIELD(
        value=MISSING, default_value="<specify path to annotation file>"
    )
    labelmap_file: Optional[str] = STR_FIELD(
        value=None, default_value="<specify path to labelmap file>"
    )


@dataclass
class ExperimentConfig:
    """Experiment configuration template."""

    data: DataConfig = DATACLASS_FIELD(DataConfig())
    kitti: KITTIConfig = DATACLASS_FIELD(KITTIConfig())
    coco: COCOConfig = DATACLASS_FIELD(COCOConfig())
    odvg: ODVGConfig = DATACLASS_FIELD(ODVGConfig())
    results_dir: Optional[str] = STR_FIELD(
        value="/results", default_value="/results"
    )
    verbose: bool = BOOL_FIELD(value=False)
