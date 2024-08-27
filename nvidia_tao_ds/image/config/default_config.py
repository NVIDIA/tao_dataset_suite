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

    image_dir: str = STR_FIELD(
        value=MISSING, default_value="images", description="Output image path"
    )


@dataclass
class ExperimentConfig:
    """Experiment configuration template."""

    data: DataConfig = DATACLASS_FIELD(
        DataConfig(), description="Input data parameters"
    )
    in_place: Optional[bool] = BOOL_FIELD(
        True, default_value=False, description="If correction needs to be done inplace"
    )
    results_dir: str = STR_FIELD(value="/results", default_value="/results")
