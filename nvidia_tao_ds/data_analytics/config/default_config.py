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
from typing import Optional, List, Dict
import multiprocessing


@dataclass
class DataConfig:
    """Dataset configuration template."""

    input_format: str = MISSING
    output_dir: str = MISSING
    kpi_sources: Optional[List[Dict[str, str]]] = MISSING
    image_dir: str = MISSING
    ann_path: str = MISSING
    mapping: str = MISSING


@dataclass
class GraphConfig:
    """Graph configuration template."""

    height: int = 15
    width: int = 15
    show_all: bool = False
    generate_summary_and_graph: bool = True


@dataclass
class WandbConfig:
    """Wandb configuration template."""

    project: Optional[str] = None
    entity: Optional[str] = None
    save_code: bool = False
    name: Optional[str] = None
    notes: Optional[str] = None
    tags: Optional[List] = None
    visualize: bool = False


@dataclass
class VisualizeConfig:
    """Visualize configuration template."""

    platform: str = "local"
    tag: Optional[str] = None


@dataclass
class KpiConfig:
    """KPI configuration template."""

    iou_threshold: float = 0.5
    filter: bool = False
    num_recall_points: int = 11
    ignore_sqwidth: int = 0
    conf_threshold: float = 0.5
    is_internal: bool = False


@dataclass
class ImageConfig:
    """Image configuration template."""

    generate_image_with_bounding_box: bool = False
    sample_size: int = 100


@dataclass
class ExperimentConfig:
    """Experiment configuration template."""

    workers: int = multiprocessing.cpu_count()
    data: DataConfig = DataConfig()
    image: ImageConfig = ImageConfig()
    apply_correction: bool = False
    graph: GraphConfig = GraphConfig()
    wandb: WandbConfig = WandbConfig()
    visualize: VisualizeConfig = VisualizeConfig()
    kpi: KpiConfig = KpiConfig()
    results_dir: Optional[str] = None
