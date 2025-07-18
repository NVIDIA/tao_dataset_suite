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

"""Entrypoint script to run COCO annotation slicer."""

import os
import sys

from nvidia_tao_core.config.annotations.slice_config import SliceConfig
from nvidia_tao_ds.annotations.slicer import builder
from nvidia_tao_ds.core.decorators import monitor_status
from nvidia_tao_ds.core.hydra.hydra_runner import hydra_runner


@hydra_runner(
    config_path=os.path.join(os.path.dirname(os.path.abspath(__file__)), "../experiment_specs"),
    config_name="slice", schema=SliceConfig
)
def main(cfg: SliceConfig) -> None:
    """Wrapper function for format conversion."""
    run_conversion(cfg=cfg)


@monitor_status(mode='Annotation slicing')
def run_conversion(cfg: SliceConfig):
    """TAO annotation convert wrapper."""
    try:
        if cfg.data.format == 'COCO':
            slicer = builder(cfg)
            slicer.slice(cfg.results_dir, cfg.filter)
    except KeyboardInterrupt as e:
        print(f"Interrupting annotation slicing with error: {e}")
        sys.exit()
    except RuntimeError as e:
        print(f"Annotation slicing run failed with error: {e}")
        sys.exit()


if __name__ == '__main__':
    main()
