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

"""Define routines required for the entrypoint."""

import importlib
import os
import pkgutil
import subprocess
import sys
from time import time

from nvidia_tao_ds.core.telemetry.nvml_utils import get_device_details
from nvidia_tao_ds.core.telemetry.telemetry import send_telemetry_data


def get_subtasks(package):
    """Get supported subtasks for a given task.

    This function lists out the tasks in in the .scripts folder.

    Returns:
        subtasks (dict): Dictionary of files.

    """
    module_path = package.__path__
    modules = {}

    # Collect modules dynamically.
    for _, task, is_package in pkgutil.walk_packages(module_path):
        if is_package:
            continue
        module_name = package.__name__ + '.' + task
        module_details = {
            "module_name": module_name,
            "runner_path": os.path.abspath(importlib.import_module(module_name).__file__),
        }
        modules[task] = module_details

    return modules


def launch(parser, subtasks, multigpu_support=['generate'], task="tao_ds"):
    """CLI function that executes subtasks.

    Args:
        parser: Created parser object for a given task.
        subtasks: list of subtasks for a given task.
    """
    # Subtasks for a given model.
    parser.add_argument(
        'subtask', default='train', choices=subtasks.keys(), help="Subtask for a given task/model.",
    )
    # Add standard TAO arguments.
    parser.add_argument(
        "-r",
        "--results_dir",
        help="Path to a folder where the experiment outputs should be written. (DEFAULT: ./)",
        required=False,
    )
    parser.add_argument(
        "-e",
        "--experiment_spec_file",
        help="Path to the experiment spec file.",
        required=True)
    parser.add_argument(
        "-g",
        "--gpus",
        help="Number of GPUs or gpu index to use.",
        type=str,
        default=None
    )

    # Parse the arguments.
    args, unknown_args = parser.parse_known_args()
    process_passed = True

    script_args = ""
    # Check for whether the experiment spec file exists.
    if not os.path.exists(args.experiment_spec_file):
        raise FileNotFoundError(
            f"Experiment spec file wasn't found at {args.experiment_spec_file}"
        )
    path, name = os.path.split(args.experiment_spec_file)
    if path != "":
        script_args += f" --config-path {os.path.realpath(path)}"
    script_args += f" --config-name {name}"

    if args.results_dir:
        script_args += " results_dir=" + args.results_dir
    if args.gpus and args.subtask in multigpu_support:
        try:
            script_args += f" gpu_ids=[{','.join([str(i) for i in range(int(args.gpus))])}]"
        except ValueError:
            script_args += f" gpu_ids={args.gpus}"

    script = subtasks[args.subtask]["runner_path"]

    # Pass unknown args to call
    unknown_args_as_str = " ".join(unknown_args)
    # Create a system call.
    call = "python " + script + script_args + " " + unknown_args_as_str

    start = time()
    try:
        # Run the script.
        subprocess.check_call(call, shell=True, stdout=sys.stdout, stderr=sys.stdout)
    except (KeyboardInterrupt, SystemExit) as e:
        print("Command was interrupted due to ", e)
        process_passed = True
    except subprocess.CalledProcessError as e:
        if e.output is not None:
            print(e.output)
        process_passed = False
    end = time()
    time_lapsed = int(end - start)

    try:
        gpu_data = list()
        for device in get_device_details():
            gpu_data.append(device.get_config())
        print("Sending telemetry data.")
        send_telemetry_data(
            task,
            args.subtask,
            gpu_data,
            num_gpus=args.gpus,
            time_lapsed=time_lapsed,
            pass_status=process_passed
        )
    except Exception as e:
        print("Telemetry data couldn't be sent, but the command ran successfully.")
        print(f"[Error]: {e}")
        pass

    if not process_passed:
        print("Execution status: FAIL")
        exit(1)  # returning non zero return code from the process.

    print("Execution status: PASS")