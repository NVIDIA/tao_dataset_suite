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

"""Generate annotations using existing AI models in TAO Toolkit"""

import os

from nvidia_tao_core.config.auto_label.default_config import ExperimentConfig
from nvidia_tao_ds.core.hydra.hydra_runner import hydra_runner
from nvidia_tao_ds.core.decorators import monitor_status

from nvidia_tao_ds.auto_label.grounding_dino.inference import run_grounding_inference
from nvidia_tao_ds.auto_label.mal.inference import run_mal_inference


@monitor_status(mode='Auto-label')
def run_experiment(cfg, results_dir=None):
    """Start the inference."""
    os.makedirs(results_dir, exist_ok=True)
    if cfg.autolabel_type == "grounding_dino":
        run_grounding_inference(cfg, results_dir)
    elif cfg.autolabel_type == "mal":
        run_mal_inference(cfg, results_dir)
    else:
        raise NotImplementedError(f"{cfg.autolabel_type}")


spec_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# Load experiment specification, additially using schema for validation/retrieving the default values.
# --config_path and --config_name will be provided by the entrypoint script.
@hydra_runner(
    config_path=os.path.join(spec_root, "experiment_specs"), config_name="generate", schema=ExperimentConfig
)
def main(cfg: ExperimentConfig) -> None:
    """Run the inference process."""
    run_experiment(cfg,
                   results_dir=cfg.results_dir)


if __name__ == "__main__":
    main()
