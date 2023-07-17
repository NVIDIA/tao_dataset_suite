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

"""Utils for configuration."""


def update_config(cfg):
    """Update Hydra config."""
    # mask threshold
    if len(cfg.train.mask_thres) == 1:
        # this means to repeat the same threshold three times
        # all scale objects are sharing the same threshold
        cfg.train.mask_thres = [cfg.train.mask_thres[0] for _ in range(3)]
    assert len(cfg.train.mask_thres) == 3

    # frozen_stages
    if len(cfg.model.frozen_stages) == 1:
        cfg.model.frozen_stages = [0, cfg.model.frozen_stages[0]]
    assert len(cfg.model.frozen_stages) == 2
    assert len(cfg.train.margin_rate) == 2
    return cfg
