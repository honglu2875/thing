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
import threading
from concurrent.futures import Future, ThreadPoolExecutor
from threading import Thread
from typing import Callable, Optional

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
        with ThreadPoolExecutor(3) as pool:
            target_fns = [
                self.servicer.get_tensor,
                self.servicer.get_string,
                self.servicer.get_pytree,
            ]
            futures: list[Optional[Future]] = [None] * len(target_fns)
            try:
                while not self._stopped and not threading._SHUTTING_DOWN:
                    # There is a tricky deadlock problem without the `threading._SHUTTING_DOWN`:
                    #   - Data queues are created first.
                    #   - By creating the thread pool, another thread along with a simple queue is created.
                    #   - The target_fns inside thread pool wait for the original data queues.
                    #   - When the program exits, queues are closed in *REVERSE* order!
                    #   - The simple queue of the thread pool is trying to close first,
                    #     but the target_fns are waiting for the original data queues to close
                    #   - deadlock!
                    # `threading._SHUTTING_DOWN` solves it because it is flipped to True *BEFORE* trying
                    # to close the queues in reverse order.
                    for i, fn in enumerate(target_fns):
                        if futures[i] is None and not threading._SHUTTING_DOWN:
                            futures[i] = pool.submit(fn)

                    for i, future in enumerate(futures):
                        if future is not None and future.done():
                            obj = future.result()
                            futures[i] = None
                        else:
                            continue

                        if (
                            obj is None
                        ):  # only ever return None when the servicer is closed
                            return
                        self.store.add(obj)
                        futures[i] = None
            finally:
                self.servicer.close()
                for future in futures:
                    if future is not None:
                        future.cancel()

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
