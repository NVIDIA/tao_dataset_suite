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

"""Common decorators used in TAO Toolkit."""

from functools import wraps
import os

import nvidia_tao_ds.core.logging.logging as status_logging
from nvidia_tao_ds.core.mlops.wandb import alert


def monitor_status(name='Data-services', mode='analyze'):
    """Status monitoring decorator."""
    def inner(runner):
        @wraps(runner)
        def _func(cfg, **kwargs):
            # set up status logger
            os.makedirs(cfg.results_dir, exist_ok=True)

            status_file = os.path.join(cfg.results_dir, "status.json")
            status_logging.set_status_logger(
                status_logging.StatusLogger(
                    filename=status_file,
                    verbosity=1,
                    append=True
                )
            )
            s_logger = status_logging.get_status_logger()
            try:
                s_logger.write(
                    status_level=status_logging.Status.STARTED,
                    message=f"Starting {name} {mode}."
                )
                alert(
                    title=f'{mode.capitalize()} started',
                    text=f'{mode.capitalize()} {name} has started',
                    level=0
                )
                runner(cfg, **kwargs)
                s_logger.write(
                    status_level=status_logging.Status.RUNNING,
                    message=f"{mode.capitalize()} finished successfully."
                )
                if os.getenv("CLOUD_BASED") == "True":
                    s_logger.write(
                        status_level=status_logging.Status.RUNNING,
                        message="Job artifacts in results dir are being uploaded to the cloud"
                    )
            except (KeyboardInterrupt, SystemError):
                status_logging.get_status_logger().write(
                    message=f"{mode.capitalize()} was interrupted",
                    verbosity_level=status_logging.Verbosity.INFO,
                    status_level=status_logging.Status.FAILURE
                )
                alert(
                    title=f'{mode.capitalize()} stopped',
                    text=f'{mode.capitalize()} was interrupted',
                    level=1
                )
            except Exception as e:
                status_logging.get_status_logger().write(
                    message=str(e),
                    status_level=status_logging.Status.FAILURE
                )
                alert(
                    title=f'{mode.capitalize()} failed',
                    text=str(e),
                    level=2
                )
                raise e

        return _func
    return inner
