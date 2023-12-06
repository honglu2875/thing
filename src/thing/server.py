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
import logging
from threading import Thread
from typing import Optional

from thing.argument import ServerArguments, ServicerArguments
from thing.servicer import Servicer, _exit
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
        self.servicer = Servicer(**dataclasses.asdict(servicer_args))

        server_args = ServerArguments.from_args(server_args, kwargs)
        self.server_args = ServerArguments.from_args(
            dataclasses.asdict(server_args), kwargs
        )
        self.store = Store()
        self.logger = logging.getLogger(__name__)
        self._retrieve_thread = None
        self._stopped = False

    def _retrieve_obj(self):
        while not self._stopped:
            obj = self.servicer.get_tensor(timeout=None)

            if obj is None:
                break
            self.store.add(obj)

    def start(self):
        self.servicer.start()
        self._retrieve_thread = Thread(target=self._retrieve_obj)
        self._retrieve_thread.daemon = True  # so that it exits with the main thread
        self._retrieve_thread.start()

    def close(self):
        self._stopped = True
        self.servicer.close()

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def __del__(self):
        self.close()
