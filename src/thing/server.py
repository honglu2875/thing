import logging
import sys
from concurrent import futures
from queue import Queue
from typing import Optional

import grpc

from thing import thing_pb2, thing_pb2_grpc
from thing.utils import reconstruct_array


class Server(thing_pb2_grpc.ThingServicer):
    """
    The main class for receiving the data payloads from `thing.client`.
    A few important notes:
        - The instance is designed to be created using the `with` statement so that grpc server can be
          opened and closed properly. I do not see a use case requiring multiple instances on the server,
          so I would rather not to over-engineer it.
        - The queue in the server is thread-safe. But the queue will not respect the order (as separate
          chunks of payload can arrive in any order)
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

        self._incomplete_chunks = {}  # save incomplete chunks of arrays

        self._server = None  # the gRPC server instance
        self._blocked = False

    def __enter__(self):
        self.server = grpc.server(futures.ThreadPoolExecutor())
        thing_pb2_grpc.add_ThingServicer_to_server(self, self.server)
        self.server.add_insecure_port(f"[::]:{self.port}")
        self.server.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._blocked = True
        if self._server is not None:
            self._server.stop(0)

    def _check_valid(self, request):
        if self._blocked:
            return False
        if (
            self.max_byte_per_item
            and sys.getsizeof(request.data) > self.max_byte_per_item
        ):
            return False
        return True

    def CatchByte(self, request, context):
        if not self._check_valid(request):
            return thing_pb2.Response(status=thing_pb2.STATUS.FAILURE)

        self._byte_queue.put(request.data)
        return thing_pb2.Response(status=thing_pb2.STATUS.SUCCESS)

    def CatchArray(self, request, context):
        if not self._check_valid(request):
            return thing_pb2.Response(status=thing_pb2.STATUS.FAILURE)

        self._array_queue.put(request)
        return thing_pb2.Response(status=thing_pb2.STATUS.SUCCESS)

    def HealthCheck(self, request, context):
        return thing_pb2.Response(status=thing_pb2.STATUS.SUCCESS)

    def get_byte(self):
        return self._byte_queue.get()

    def get_array(self, timeout: float = 5.0):
        while True:
            array_payload = self._array_queue.get(timeout=timeout)
            incomplete_chunks = self._incomplete_chunks.get(array_payload.id, [])
            incomplete_chunks.append(array_payload)

            if array_payload.num_chunks > len(incomplete_chunks):
                self._incomplete_chunks[array_payload.id] = incomplete_chunks
                continue
            break

        if array_payload.id in self._incomplete_chunks:
            del self._incomplete_chunks[array_payload.id]

        return reconstruct_array(incomplete_chunks)
