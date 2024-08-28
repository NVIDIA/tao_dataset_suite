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

from typing import List, Optional, Dict
from dataclasses import dataclass
from nvidia_tao_ds.config_utils.default_config_utils import (
    STR_FIELD,
    INT_FIELD,
    BOOL_FIELD,
    LIST_FIELD,
    DATACLASS_FIELD,
)

from nvidia_tao_ds.auto_label.config.mal_config import MALInferenceExpConfig, MALEvalExpConfig, MALDatasetConfig, MALModelConfig, MALTrainExpConfig
from nvidia_tao_ds.auto_label.config.dataset import GDINOAugmentationConfig
from nvidia_tao_ds.auto_label.config.model import GDINOModelConfig
from nvidia_tao_ds.auto_label.config.train import GDINOTrainExpConfig


@dataclass
class MALConfig:
    """MAL config."""

    dataset: MALDatasetConfig = DATACLASS_FIELD(
        MALDatasetConfig(),
        description="Configuration parameters for MAL dataset"
    )
    train: MALTrainExpConfig = DATACLASS_FIELD(
        MALTrainExpConfig(),
        description="Configuration parameters for MAL train"
    )
    model: MALModelConfig = DATACLASS_FIELD(
        MALModelConfig(),
        description="Configuration parameters for MAL model"
    )
    inference: MALInferenceExpConfig = DATACLASS_FIELD(
        MALInferenceExpConfig(),
        description="Configuration parameters for MAL inference"
    )
    evaluate: MALEvalExpConfig = DATACLASS_FIELD(
        MALEvalExpConfig(),
        description="Configuration parameters for MAL evaluation"
    )
    checkpoint: Optional[str] = STR_FIELD(
        None,
        default_value="",
        description="MAL model checkpoint path",
    )
    results_dir: Optional[str] = STR_FIELD(
        value=None,
        default_value="",
        description="Result directory",
    )


@dataclass
class GDINOConfig:
    """Grounding DINO config."""

    @dataclass
    class GDINODataConfig:
        """DINO dataset config used for auto-labeling."""

        image_dir: Optional[str] = STR_FIELD(
            None,
            default_value="",
            description="Image root directory",
        )
        noun_chunk_path: Optional[str] = STR_FIELD(
            value=None,
            default_value=""
        )
        class_names: Optional[List[str]] = LIST_FIELD(
            arrList=[],
            description="List of classes to run auto-labeling"
        )
        augmentation: GDINOAugmentationConfig = DATACLASS_FIELD(
            GDINOAugmentationConfig(),
            description="Configuration parameters for Grounding DINO augmenation"
        )

    train: GDINOTrainExpConfig = DATACLASS_FIELD(
        GDINOTrainExpConfig(),
        description="Configuration parameters for Grounding DINO train"
    )
    model: GDINOModelConfig = DATACLASS_FIELD(
        GDINOModelConfig(),
        description="Configuration parameters for Grounding DINO model"
    )
    dataset: GDINODataConfig = DATACLASS_FIELD(
        GDINODataConfig(),
        description="Configuration parameters for Grounding DINO dataset"
    )

    checkpoint: Optional[str] = STR_FIELD(
        None,
        default_value="",
        description="Grounding model checkpoint path",
    )

    results_dir: Optional[str] = STR_FIELD(
        value=None,
        default_value="",
        description="Result directory",
    )

    iteration_scheduler: List[Dict[str, float]] = LIST_FIELD(
        arrList=[{"conf_threshold": 0.5, "nms_threshold": 0.0}],
        default_values=[{"conf_threshold": 0.5, "nms_threshold": 0.0}],
        description="""The list of iteration schedule. Default is one iteration with confidence threshold of 0.5.
                    Next iteration eliminates classes/noun chunks that have been already detected."""
    )
    visualize: bool = BOOL_FIELD(
        value=True,
        default_value=True,
        description="Flag to enable visualization of bounding boxes."
    )


@dataclass
class ExperimentConfig:
    """Experiment configuration template."""

    gpu_ids: List[int] = LIST_FIELD(
        arrList=[0],
        default_value=[0],
        description="Indices of GPUs to use"
    )
    num_gpus: int = INT_FIELD(value=1,
                              default_value=1,
                              description="Number of GPUs to use")
    batch_size: int = INT_FIELD(value=4,
                                default_value=4,
                                valid_min=1,
                                description="Batch size")
    num_workers: int = INT_FIELD(value=8,
                                 default_value=8,
                                 valid_min=1,
                                 description="Number of workers for dataloader")

    autolabel_type: str = STR_FIELD(
        value="mal",
        default_value="mal",
        description="Type of auto-labeling to run",
        valid_options="mal,grounding_dino"
    )

    mal: MALConfig = DATACLASS_FIELD(
        MALConfig(),
        description="Configuration parameters for MAL"
    )
    grounding_dino: GDINOConfig = DATACLASS_FIELD(
        GDINOConfig(),
        description="Configuration parameters for Grounding DINO"
    )

    results_dir: str = STR_FIELD(
        value="/results",
        default_value="/results",
        description="Result directory",
    )

    def __post_init__(self):
        """assertion check."""
        assert self.autolabel_type in ["mal", "grounding_dino"], f"Invalid option encountered. {self.autolabel_type}"
