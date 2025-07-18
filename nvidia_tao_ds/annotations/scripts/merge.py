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

"""Entrypoint script to run COCO annotation merger."""

import os
import sys

from nvidia_tao_core.config.annotations.merge_config import MergeConfig
from nvidia_tao_ds.annotations.merger import COCOMerger, ODVGMerger
from nvidia_tao_ds.core.decorators import monitor_status
from nvidia_tao_ds.core.hydra.hydra_runner import hydra_runner


@hydra_runner(
    config_path=os.path.join(os.path.dirname(os.path.abspath(__file__)), "../experiment_specs"),
    config_name="merge", schema=MergeConfig
)
def main(cfg: MergeConfig) -> None:
    """Wrapper function for format conversion."""
    run_conversion(cfg=cfg)


@monitor_status(mode='Annotation merging')
def run_conversion(cfg: MergeConfig):
    """TAO annotation convert wrapper."""
    try:
        if cfg.data.format.lower() == 'coco':
            output_path = os.path.join(cfg.results_dir, 'output.json')
            merger = COCOMerger(cfg.data.annotations)
            merger.merge(output_path)
        elif cfg.data.format.lower() == 'odvg':
            output_path = os.path.join(cfg.results_dir, 'merged.json')
            merger = ODVGMerger(cfg.data.annotations)
            merger.merge(output_path)
    except KeyboardInterrupt as e:
        print(f"Interrupting annotation merging with error: {e}")
        sys.exit()
    except RuntimeError as e:
        print(f"Annotation merging run failed with error: {e}")
        sys.exit()


if __name__ == '__main__':
    main()
