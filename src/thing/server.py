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
from typing import Optional

from thing.argument import ServerArguments, ServicerArguments
from thing.servicer import Servicer
from thing.store import Store


class Server:
    """
    A Server object wraps around a Servicer to receive payloads, while also managing
    the received objects, metadata, logs, etc.
    """

    def __init__(
        self,
        server_args: Optional[ServerArguments] = None,
        servicer_args: Optional[ServicerArguments] = None,
        **kwargs,
    ):
        servicer_args = ServicerArguments.from_args(servicer_args, kwargs)
        self.servicer = Servicer(**servicer_args)

        self.server_args = ServerArguments.from_args(server_args, kwargs)
        self.store = Store()
        ...

    def start(self):
        self.servicer.start()

    def close(self):
        self.servicer.close()

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
