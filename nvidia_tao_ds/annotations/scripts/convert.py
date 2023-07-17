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

"""Entrypoint script to run TAO data format conversion."""

import os
import sys
from nvidia_tao_ds.annotations.coco_to_kitti import convert_coco_to_kitti
from nvidia_tao_ds.annotations.config.default_config import ExperimentConfig
from nvidia_tao_ds.annotations.kitti_to_coco import convert_kitti_to_coco
from nvidia_tao_ds.core.decorators import monitor_status
from nvidia_tao_ds.core.hydra.hydra_runner import hydra_runner


@hydra_runner(
    config_path=os.path.join(os.path.dirname(os.path.abspath(__file__)), "../experiment_specs"),
    config_name="annotations", schema=ExperimentConfig
)
def main(cfg: ExperimentConfig) -> None:
    """Wrapper function for format conversion."""
    cfg.results_dir = cfg.results_dir or cfg.data.output_dir
    run_conversion(cfg=cfg)


@monitor_status(mode='Annotation conversion')
def run_conversion(cfg: ExperimentConfig):
    """TAO annotation convert wrapper."""
    try:
        if cfg.data.input_format == "KITTI" and cfg.data.output_format == "COCO":
            convert_kitti_to_coco(
                cfg.kitti.image_dir,
                cfg.kitti.label_dir,
                cfg.data.output_dir,
                cfg.kitti.mapping,
                cfg.kitti.project)
        elif cfg.data.input_format == "COCO" and cfg.data.output_format == "KITTI":
            convert_coco_to_kitti(
                cfg.coco.ann_file,
                cfg.data.output_dir)
        else:
            print("Unsupported format")
    except KeyboardInterrupt as e:
        print(f"Interrupting data conversion with error: {e}")
        sys.exit()
    except RuntimeError as e:
        print(f"Data conversion run failed with error: {e}")
        sys.exit()


if __name__ == '__main__':
    main()
