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
from typing import Optional, List


@dataclass
class DataConfig:
    """Dataset configuration template."""

    format: str = "COCO"
    annotations: List[str] = MISSING
    same_categories: bool = True


@dataclass
class MergeConfig:
    """Experiment configuration template."""

    data: DataConfig = DataConfig()
    results_dir: Optional[str] = None
