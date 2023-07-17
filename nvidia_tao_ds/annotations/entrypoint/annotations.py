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

"""Define entrypoint to run tasks for annotation."""

import argparse

from nvidia_tao_ds.core.entrypoint.entrypoint import get_subtasks, launch
from nvidia_tao_ds.annotations import scripts


def main():
    """Main entrypoint wrapper."""
    # Create parser for a given task.
    parser = argparse.ArgumentParser(
        "annotation",
        add_help=True,
        description="Annotation entrypoint"
    )

    # Build list of subtasks by inspecting the scripts package.
    subtasks = get_subtasks(scripts)

    # Parse the arguments and launch the subtask.
    launch(
        parser, subtasks, task="annotation"
    )


if __name__ == '__main__':
    main()
