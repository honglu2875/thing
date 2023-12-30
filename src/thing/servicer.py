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
import logging
import sys
import time
from concurrent import futures
from queue import Queue
from typing import Any, Optional, Union

import grpc

from thing import thing_pb2, thing_pb2_grpc
from thing.type import PyTreeObject, StringObject, TensorObject
from thing.utils import (reconstruct_pytree_object, reconstruct_string_object,
                         reconstruct_tensor_object)

_exit = object()  # a sentinel object to indicate the end of the queue


class Servicer(thing_pb2_grpc.ThingServicer):
    """
    The main class for receiving the data payloads from `thing.client`.
    Exposed APIs:
        - `get_array` to retrieve the array payload (whatever next in the queue that is complete).
        - `get_byte` (todo: I haven't quite figured out what to use it for)
        - `start` and `close` to start and close the server.
        - `__enter__` and `__exit__` to use it as a context manager.
    """

    def __init__(
        self,
        port: int = 2875,
        max_size: int = 0,
        max_byte_per_item: int = 0,
        logger: Optional[logging.Logger] = None,
    ):
        """
        Args:
            max_size: the maximum size of the queue. 0 if the queue is unbounded.
            max_byte_per_item: the maximum bytes of each item in the queue. 0 if item sizes are unbounded.
        """
        super().__init__()
        self.port = port
        self.max_size = max_size
        self.max_byte_per_item = max_byte_per_item
        self.logger = logger or logging.getLogger(__name__)
        self._byte_queue = Queue(max_size)
        self._array_queue = Queue(max_size)
        self._string_queue = Queue(max_size)
        self._pytree_queue = Queue(max_size)

        self._incomplete_chunks = {}  # save incomplete chunks of arrays

        self._id_to_client_addr = {}  # save the client address for each array id
        self._id_to_timestamp = {}  # save the timestamp of the latest chunk for each id

        self._server = None  # the gRPC server instance
        self._blocked = False

    def start(self):
        if self._server is not None:
            raise RuntimeError("Server is already started.")
        self._server = grpc.server(futures.ThreadPoolExecutor())
        thing_pb2_grpc.add_ThingServicer_to_server(self, self._server)
        self._server.add_insecure_port(f"[::]:{self.port}")
        self._server.start()

    def close(self):
        self._array_queue.put(_exit)
        self._string_queue.put(_exit)
        self._pytree_queue.put(_exit)
        self._blocked = True
        if self._server is not None:
            self._server.stop(0)

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def __del__(self):
        self.close()

    def _check_valid(self, request):
        if self._blocked:
            return False
        if (
            self.max_byte_per_item
            and sys.getsizeof(request.data) > self.max_byte_per_item
        ):
            return False
        return True

    def _check_register_request(self, request, context):
        if not self._check_valid(request):
            return thing_pb2.Response(status=thing_pb2.STATUS.FAILURE)

        if request.id is None:
            self.logger.warning("Received an array without an id. Ignoring.")
            return thing_pb2.Response(status=thing_pb2.STATUS.FAILURE)

        if (
            request.id in self._id_to_client_addr
            and self._id_to_client_addr[request.id] != context.peer()
        ):
            self.logger.warning(
                f"Received an array with id {request.id} from a different client. Ignoring."
            )
            return thing_pb2.Response(status=thing_pb2.STATUS.FAILURE)

        if request.id not in self._id_to_client_addr:
            self._id_to_client_addr[request.id] = context.peer()

        self._id_to_timestamp[request.id] = time.time()

    def _catch(self, request, context, queue):
        self._check_register_request(request, context)

        queue.put(request)
        return thing_pb2.Response(status=thing_pb2.STATUS.SUCCESS)

    def CatchByte(self, request, context):
        # todo: have not exposed this yet on client
        if not self._check_valid(request):
            return thing_pb2.Response(status=thing_pb2.STATUS.FAILURE)

        self._byte_queue.put(request.data)
        return thing_pb2.Response(status=thing_pb2.STATUS.SUCCESS)

    def CatchArray(self, request, context):
        return self._catch(request, context, self._array_queue)

    def CatchString(self, request, context):
        return self._catch(request, context, self._string_queue)

    def CatchPyTree(self, request, context):
        return self._catch(request, context, self._pytree_queue)

    def HealthCheck(self, request, context):
        return thing_pb2.Response(status=thing_pb2.STATUS.SUCCESS)

    def get_byte(self):
        return self._byte_queue.get()

    def get_tensor(self, timeout: Optional[float] = None) -> Optional[TensorObject]:
        # Keep getting chunks until we received the first complete payload.
        # Incomplete payloads keep getting saved in `self._incomplete_chunks`.
        while True:
            array_payload: thing_pb2.Array = self._array_queue.get(timeout=timeout)
            if array_payload is _exit:
                self.logger.info("The array queue received exit signal. Exiting.")
                return None

            current_chunks = self._incomplete_chunks.get(array_payload.id, [])
            current_chunks.append(array_payload)

            if array_payload.num_chunks > len(current_chunks):
                self._incomplete_chunks[array_payload.id] = current_chunks
                continue
            break

        if array_payload.id in self._incomplete_chunks:
            del self._incomplete_chunks[array_payload.id]

        return reconstruct_tensor_object(
            current_chunks,
            client_addr=self._id_to_client_addr.get(array_payload.id, "unknown"),
            timestamp=self._id_to_timestamp.get(array_payload.id, 0),
        )

    def get_string(self, timeout: Optional[float] = None) -> Optional[StringObject]:
        obj: thing_pb2.String = self._string_queue.get(timeout=timeout)
        if obj is _exit:
            self.logger.info("The string queue received exit signal. Exiting.")
            return None
        return reconstruct_string_object(
            obj,
            client_addr=self._id_to_client_addr.get(obj.id, "unknown"),
            timestamp=self._id_to_timestamp.get(obj.id, 0),
        )

    def get_pytree(self, timeout: Optional[float] = None) -> Optional[PyTreeObject]:
        obj = self._pytree_queue.get(timeout=timeout)
        if obj is _exit:
            self.logger.info("The pytree queue received exit signal. Exiting.")
            return None
        return reconstruct_pytree_object(
            obj,
            client_addr=self._id_to_client_addr.get(obj.id, "unknown"),
            timestamp=self._id_to_timestamp.get(obj.id, 0),
        )
