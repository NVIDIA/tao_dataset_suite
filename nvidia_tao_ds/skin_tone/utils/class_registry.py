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

"""Class registry"""

import inspect
import typing
import omegaconf
import dataclasses


class ClassRegistry:
    """Class Registry for editing methods"""

    def __init__(self):
        """Initializes ClassRegistry"""
        self.classes = {}
        self.args = {}
        self.arg_keys = None

    def __getitem__(self, item):
        """Returns item in registry"""
        return self.classes[item]

    def make_dataclass_from_init(self, func, name, stop_args):
        """Generates dataclass"""
        args = inspect.signature(func).parameters
        args = [
            (k, typing.Any, omegaconf.MISSING)
            if v.default is inspect.Parameter.empty
            else (k, typing.Optional[typing.Any], None)
            if v.default is None
            else (
                k,
                type(v.default),
                v.default,
            )
            for k, v in args.items()
        ]
        args = [arg for arg in args if arg[0] not in stop_args]
        return dataclasses.make_dataclass(name, args)

    def add_to_registry(
        self, name, stop_args=("self", "args", "kwargs")
    ):
        """Add class to registry"""
        def add_class_by_name(cls):
            """Add class to registry fn"""
            self.classes[name] = cls
            self.args[name] = self.make_dataclass_from_init(
                cls.__init__, name, stop_args
            )
            return cls

        return add_class_by_name
