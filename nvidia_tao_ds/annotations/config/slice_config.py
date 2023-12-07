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

from dataclasses import dataclass, field
from omegaconf import MISSING
from typing import Optional, List, Any


@dataclass
class DataConfig:
    """Dataset configuration template."""

    format: str = 'COCO'
    annotation_file: str = MISSING


@dataclass
class FilterConfig:
    """Dataset configuration template."""

    mode: str = "random"  # category, number
    reuse_categories: bool = True
    dump_remainder: bool = False
    split: Any = 0.25
    num_samples: int = 100
    included_categories: List[str] = field(default_factory=lambda: [])
    excluded_categories: List[str] = field(default_factory=lambda: [])
    re_patterns: List[str] = field(default_factory=lambda: [])


@dataclass
class SliceConfig:
    """Experiment configuration template."""

    data: DataConfig = DataConfig()
    filter: FilterConfig = FilterConfig()
    results_dir: Optional[str] = None
