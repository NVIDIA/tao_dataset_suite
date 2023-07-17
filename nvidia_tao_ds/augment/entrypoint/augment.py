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

"""Define entrypoint to run tasks for augmentation."""

import argparse
import importlib
import os
import pkgutil
import subprocess
import sys
from time import time

from nvidia_tao_ds.augment import scripts
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
        'subtask', default='generate', choices=subtasks.keys(), help="Subtask for augmentation.",
    )
    # Add standard TAO arguments.
    parser.add_argument(
        "-e",
        "--experiment_spec_file",
        help="Path to the experiment spec file.",
        required=True)
    parser.add_argument(
        "--gpu_ids",
        help="GPU index to use.",
        type=str,
        default=None
    )
    parser.add_argument(
        "--num_gpus",
        help="Number of GPUs to use.",
        type=int,
        default=1
    )
    parser.add_argument(
        "-o",
        "--output_specs_dir",
        help="Path to a target folder where experiment spec files will be downloaded.",
        default=None
    )
    parser.add_argument(
        "--mpirun_arg",
        type=str,
        default="-x NCCL_IB_HCA=mlx5_4,mlx5_6,mlx5_8,mlx5_10 -x NCCL_SOCKET_IFNAME=^lo,docker",
        help="Arguments for the mpirun command to run multi-node."
    )
    parser.add_argument(
        "--launch_cuda_blocking",
        action="store_true",
        default=False,
        help="Debug flag to add CUDA_LAUNCH_BLOCKING=1 to the command calls."
    )

    # Parse the arguments.
    args, unknown_args = parser.parse_known_args()

    script_args = ""
    # Process spec file for all commands except the one for getting spec files ;)
    if args.subtask not in ["download_specs", "pitch_stats"]:
        # Make sure the user provides spec file.
        if args.experiment_spec_file is None:
            print(f"ERROR: The subtask `{args.subtask}` requires the following argument: -e/--experiment_spec_file")
            sys.exit()

        # Make sure the file exists!
        if not os.path.exists(args.experiment_spec_file):
            print(f"ERROR: The indicated experiment spec file `{args.experiment_spec_file}` doesn't exist!")
            sys.exit()

        # Split spec file_path into config path and config name.
        path, name = os.path.split(args.experiment_spec_file)
        if path != '':
            script_args += " --config-path " + os.path.realpath(path)
        script_args += " --config-name " + name
        # Find relevant module and pass args.

    mpi_command = ""
    gpu_ids = args.gpu_ids
    num_gpus = args.num_gpus
    if gpu_ids is None:
        gpu_ids = range(num_gpus)
    else:
        gpu_ids = eval(args.gpu_ids)
        num_gpus = len(gpu_ids)

    launch_cuda_blocking = args.launch_cuda_blocking
    assert num_gpus > 0, "At least 1 GPU required to run any task."

    if num_gpus > 1:
        if args.subtask not in multigpu_support:
            raise NotImplementedError(
                f"This {args['subtask']} doesn't support multiGPU. Please set --num_gpus 1"
            )
        mpi_command = f'mpirun -np {num_gpus} --oversubscribe --bind-to none --allow-run-as-root -mca pml ob1 -mca btl ^openib'

    if args.subtask in multigpu_support:
        if not args.gpu_ids:
            script_args += f" gpu_ids=[{','.join([str(i) for i in range(num_gpus)])}]"
            script_args += f" num_gpus={num_gpus}"
        else:
            script_args += f" gpu_ids=[{','.join([str(i) for i in (eval(args.gpu_ids))])}]"
            script_args += f" num_gpus={len(gpu_ids)}"

    script = subtasks[args.subtask]["runner_path"]

    # Pass unknown args to call
    unknown_args_as_str = " ".join(unknown_args)
    task_command = f"python {script} {script_args} {unknown_args_as_str}"
    env_variables = ""
    env_variables += set_gpu_info_single_node(num_gpus, gpu_ids)
    if launch_cuda_blocking:
        task_command = f"CUDA_LAUNCH_BLOCKING=1 {task_command}"
    run_command = f"{mpi_command} bash -c \'{env_variables} {task_command}\'"

    start = time()
    process_passed = True
    try:
        subprocess.check_call(
            run_command,
            shell=True,
            stdout=sys.stdout,
            stderr=sys.stderr
        )
    except (KeyboardInterrupt, SystemExit):
        print("Command was interrupted.")
    except subprocess.CalledProcessError as e:
        if e.output is not None:
            print(f"TAO Toolkit task: {args['subtask']} failed with error:\n{e.output}")
        process_passed = False
    end = time()
    time_lapsed = int(end - start)

    try:
        gpu_data = []
        for device in get_device_details():
            gpu_data.append(device.get_config())
        print("Sending telemetry data.")
        send_telemetry_data(
            task,
            args.subtask,
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
        sys.exit(1)  # returning non zero return code from the process.

    print("Execution status: PASS")


def check_valid_gpus(num_gpus, gpu_ids):
    """Check if the number of GPU's called and IDs are valid.

    This function scans the machine using the nvidia-smi routine to find the
    number of GPU's and matches the id's and num_gpu's accordingly.

    Once validated, it finally also sets the CUDA_VISIBLE_DEVICES env variable.

    Args:
        num_gpus (int): Number of GPUs alloted by the user for the job.
        gpu_ids (list(int)): List of GPU indices used by the user.

    Returns:
        No explicit returns
    """
    # Ensure the gpu_ids are all different, and sorted
    gpu_ids = sorted(list(set(gpu_ids)))
    assert num_gpus > 0, "At least 1 GPU required to run any task."
    num_gpus_available = str(subprocess.check_output(["nvidia-smi", "-L"])).count("UUID")
    max_id = max(gpu_ids)
    assert min(gpu_ids) >= 0, (
        "GPU ids cannot be negative."
    )
    assert len(gpu_ids) == num_gpus, (
        f"The number of GPUs ({gpu_ids}) must be the same as the number of GPU indices"
        f" ({num_gpus}) provided."
    )
    assert max_id < num_gpus_available and num_gpus <= num_gpus_available, (
        "Checking for valid GPU ids and num_gpus."
    )
    cuda_visible_devices = ",".join([str(idx) for idx in gpu_ids])
    os.environ['CUDA_VISIBLE_DEVICES'] = cuda_visible_devices


def set_gpu_info_single_node(num_gpus, gpu_ids):
    """Set gpu environment variable for single node."""
    check_valid_gpus(num_gpus, gpu_ids)

    env_variable = ""
    visible_devices = os.getenv("CUDA_VISIBLE_DEVICES", None)
    if visible_devices is not None:
        env_variable = f" CUDA_VISIBLE_DEVICES={visible_devices}"
    return env_variable


def main():
    """Main entrypoint wrapper."""
    # Create parser for a given task.
    parser = argparse.ArgumentParser(
        "augmentation",
        add_help=True,
        description="TAO Toolkit entrypoint for MAL"
    )

    # Build list of subtasks by inspecting the scripts package.
    subtasks = get_subtasks(scripts)

    # Parse the arguments and launch the subtask.
    launch(
        parser, subtasks, task="augment"
    )


if __name__ == '__main__':
    main()
