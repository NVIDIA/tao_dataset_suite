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

"""Entrypoint script to run TAO skin augmentation."""


import os
import shutil
from tqdm import tqdm

from nvidia_tao_core.config.skin_tone.default_config import ExperimentConfig
from nvidia_tao_ds.core.decorators import monitor_status
from nvidia_tao_ds.core.hydra.hydra_runner import hydra_runner
from nvidia_tao_ds.skin_tone.runners.simple_runner import SimpleRunner
from nvidia_tao_ds.skin_tone.models.deeplab.face_masks import get_face_masks


@monitor_status(mode='skin_tone')
def run_experiment(cfg):
    """Start the inference."""
    required_ptms = ['e4e_ffhq_encode.pt', 'iresnet50-7f187506.pth', 'sfe_editor_light.pt',
                     'stylegan2-ffhq-config-f.pt', 'R-101-GN-WS.pth.tar', 'deeplab_model.pth']

    for ptm in required_ptms:
        assert os.path.exists(f'/pretrained/{ptm}'), f"{ptm} not found in /pretrained directory"

    runner = SimpleRunner(
        editor_ckpt_pth="/pretrained/sfe_editor_light.pt"
    )
    # print(runner.available_editings())

    mask_dir = get_face_masks(cfg)

    for (root, _, files) in os.walk(cfg.dataset.input_dir):
        for input_file in tqdm(files):
            fpath = os.path.join(root, input_file)
            output_path = os.path.join(cfg.results_dir, 'output', input_file)
            mask_path = os.path.join(mask_dir, input_file)
            # Lum edit
            runner.edit(
                orig_img_pth=fpath,
                editing_name="lum",
                edited_power=cfg.color_aug.lum.offset,  # factor
                save_pth=output_path,
                use_mask=True,  # automatic masks seem to include the lips and eyes
                mask_path=mask_path,  # this currently has all of them, we only want where it equals 1, can we remove all others from mask? Zero them all out?
            )
            # Hue edit on lum-edited image
            runner.edit(
                orig_img_pth=output_path,
                editing_name="hue",
                edited_power=cfg.color_aug.hue.angle,  # factor
                save_pth=output_path,
                use_mask=True,
                mask_path=mask_path,
            )

    shutil.rmtree(mask_dir, ignore_errors=True)


spec_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# Load experiment specification, additially using schema for validation/retrieving the default values.
# --config_path and --config_name will be provided by the entrypoint script.
@hydra_runner(
    config_path=os.path.join(spec_root, "experiment_specs"), config_name="augment", schema=ExperimentConfig
)
def main(cfg: ExperimentConfig) -> None:
    """Run the inference process."""
    run_experiment(cfg)


if __name__ == "__main__":
    main()
