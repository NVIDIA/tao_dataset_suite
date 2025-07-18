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

import re
import ast
import importlib
import logging
import os
import pkgutil
import shlex
import subprocess
import sys
from time import time
from contextlib import contextmanager

import yaml

from nvidia_tao_core.telemetry.telemetry import send_telemetry_data
from nvidia_tao_core.telemetry.nvml import get_device_details


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


def command_line_parser(parser, subtasks):
    """Build command line parser."""
    # Subtasks for a given model.
    parser.add_argument(
        'subtask', default='generate', choices=subtasks.keys(), help="Subtask for a given task/model.",
    )
    parser.add_argument(
        "-e",
        "--experiment_spec_file",
        help="Path to the experiment spec file.",
        required=True)

    args, unknown_args = parser.parse_known_args()
    return args, unknown_args


@contextmanager
def dual_output(log_file=None):
    """Context manager to handle dual output redirection for subprocess.

    Args:
    - log_file (str, optional): Path to the log file. If provided, output will be
      redirected to both sys.stdout and the specified log file. If not provided,
      output will only go to sys.stdout.

    Yields:
    - stdout_target (file object): Target for stdout output (sys.stdout or log file).
    - log_target (file object or None): Target for log file output, or None if log_file
      is not provided.
    """
    if log_file:
        with open(log_file, 'a') as f:
            yield sys.stdout, f
    else:
        yield sys.stdout, None


def launch(args, unknown_args, subtasks, multigpu_support=['generate'], network="tao_ds"):
    """CLI function that executes subtasks.

    Args:
        parser: Created parser object for a given task.
        subtasks: list of subtasks for a given task.
    """
    # Parse the arguments.
    process_passed = True

    script_args = ""
    # Check for whether the experiment spec file exists.
    if not os.path.exists(args["experiment_spec_file"]):
        raise FileNotFoundError(
            f'Experiment spec file wasn not found at {args["experiment_spec_file"]}'
        )
    path, name = os.path.split(args["experiment_spec_file"])
    if path != "":
        script_args += f" --config-path {os.path.realpath(path)}"
    script_args += f" --config-name {name}"

    # This enables a results_dir arg to be passed from the microservice side,
    # but there is no --results_dir cmdline arg. Instead, the spec field must be used
    if "results_dir" in args:
        script_args += " results_dir=" + args["results_dir"]

    script = subtasks[args['subtask']]["runner_path"]

    log_file = ""
    if os.getenv('JOB_ID'):
        logs_dir = os.getenv('TAO_MICROSERVICES_TTY_LOG', '/results')
        log_file = f"{logs_dir}/{os.getenv('JOB_ID')}/microservices_log.txt"

    # Pass unknown args to call
    unknown_args_as_str = " ".join(unknown_args)

    # Precedence these settings: cmdline > specfile > default
    overrides = ["num_gpus", "gpu_ids", "cuda_blocking"]
    num_gpus = 1
    gpu_ids = [0]
    launch_cuda_blocking = False
    if args["subtask"] in multigpu_support:
        # Parsing cmdline override
        if any(arg in unknown_args_as_str for arg in overrides):
            if "num_gpus" in unknown_args_as_str:
                num_gpus = int(
                    unknown_args_as_str.split('num_gpus=')[1].split()[0]
                )
            if "gpu_ids" in unknown_args_as_str:
                gpu_ids = ast.literal_eval(
                    unknown_args_as_str.split('gpu_ids=')[1].split()[0]
                )
            if "cuda_blocking" in unknown_args_as_str:
                launch_cuda_blocking = ast.literal_eval(
                    unknown_args_as_str.split('cuda_blocking=')[1].split()[0]
                )
        # If no cmdline override, look at specfile
        else:
            with open(args["experiment_spec_file"], 'r') as spec:
                exp_config = yaml.safe_load(spec)
                if 'num_gpus' in exp_config:
                    num_gpus = exp_config['num_gpus']
                if 'gpu_ids' in exp_config:
                    gpu_ids = exp_config['gpu_ids']
                if "cuda_blocking" in exp_config:
                    launch_cuda_blocking = exp_config['cuda_blocking']

    if num_gpus != len(gpu_ids):
        logging.info(f"The number of GPUs ({num_gpus}) must be the same as the number of GPU indices ({gpu_ids}) provided.")
        num_gpus = max(num_gpus, len(gpu_ids))
        gpu_ids = list(range(num_gpus)) if len(gpu_ids) != num_gpus else gpu_ids
        logging.info(f"Using GPUs {gpu_ids} (total {num_gpus})")

    num_gpus_available = str(subprocess.check_output(["nvidia-smi", "-L"])).count("UUID")
    assert num_gpus <= num_gpus_available, (
        "Checking for valid GPU ids and num_gpus."
    )

    # All future logic will look at this envvar for guidance on which devices to use
    os.environ["TAO_VISIBLE_DEVICES"] = str(gpu_ids)[1:-1]

    # Create a system call.
    call = "python " + script + script_args + " " + unknown_args_as_str
    if network == "augmentation":
        env_variables = ""
        if launch_cuda_blocking:
            env_variables += " CUDA_LAUNCH_BLOCKING=1"
        mpi_command = ""
        if num_gpus > 1:
            mpi_command = f'mpirun -np {num_gpus} --oversubscribe --bind-to none --allow-run-as-root -mca pml ob1 -mca btl ^openib'
        # Augment uses MPI, which uses all visible devices
        os.environ["CUDA_VISIBLE_DEVICES"] = os.environ["TAO_VISIBLE_DEVICES"]
        call = f"{mpi_command} bash -c \'{env_variables} {call}\'"

    process_passed = False
    start = time()
    progress_bar_pattern = re.compile(r"Epoch \d+: \s*\d+%|\[.*\]")

    try:
        # Run the script.
        with dual_output(log_file) as (stdout_target, log_target):
            proc = subprocess.Popen(
                shlex.split(call),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                bufsize=1,  # Line-buffered
                universal_newlines=True  # Text mode
            )
            last_progress_bar_line = None

            for line in proc.stdout:
                # Check if the line contains \r or matches the progress bar pattern
                if '\r' in line or progress_bar_pattern.search(line):
                    last_progress_bar_line = line.strip()
                    # Print the progress bar line to the terminal
                    stdout_target.write('\r' + last_progress_bar_line)
                    stdout_target.flush()
                else:
                    # Write the final progress bar line to the log file before a new log line
                    if last_progress_bar_line:
                        if log_target:
                            log_target.write(last_progress_bar_line + '\n')
                            log_target.flush()
                        last_progress_bar_line = None
                    stdout_target.write(line)
                    stdout_target.flush()
                    if log_target:
                        log_target.write(line)
                        log_target.flush()

            proc.wait()  # Wait for the process to complete
            # Write the final progress bar line after process completion
            if last_progress_bar_line and log_target:
                log_target.write(last_progress_bar_line + '\n')
                log_target.flush()
            if proc.returncode == 0:
                process_passed = True

    except (KeyboardInterrupt, SystemExit) as e:
        logging.info("Command was interrupted due to ", e)
        process_passed = True
    except subprocess.CalledProcessError as e:
        if e.output is not None:
            logging.info(e.output)
        process_passed = False

    end = time()
    time_lapsed = int(end - start)

    try:
        gpu_data = list()
        for device in get_device_details():
            gpu_data.append(device.get_config())
        print("Sending telemetry data.")
        send_telemetry_data(
            network,
            args["subtask"],
            gpu_data,
            num_gpus=num_gpus,
            time_lapsed=time_lapsed,
            pass_status=process_passed
        )
    except Exception as e:
        print("Telemetry data couldn't be sent, but the command ran successfully.")
        print(f"[Error]: {e}")
        pass

    if not process_passed:
        print("Execution status: FAIL")
        return False

    print("Execution status: PASS")
    return True
