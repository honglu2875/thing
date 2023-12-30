# Copyright 2023 Honglu Fan (https://github.com/honglu2875).
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
import dataclasses


@dataclasses.dataclass(kw_only=True)
class Arguments:
    @classmethod
    def from_args(cls, *args):
        """
        Merge dicts and dataclasses. The latter has higher priority.
        """
        field_names = set(cls.__dataclass_fields__.keys())
        collected_args = {}
        for arg in args:
            if arg is None:
                continue

            if dataclasses.is_dataclass(arg):
                arg = dataclasses.asdict(arg)
            if not isinstance(arg, dict):
                raise TypeError(
                    f"Arguments must be either a dataclass or a dict, got {type(arg)}"
                )
            for key in arg:
                if key in field_names:
                    collected_args[key] = arg[key]
        return cls(**collected_args)


@dataclasses.dataclass
class ServicerArguments(Arguments):
    port: int = 2875
    max_size: int = 0
    max_byte_per_item: int = 0


@dataclasses.dataclass
class ServerArguments(Arguments):
    # todo: finish this
    dummy: str = ""
