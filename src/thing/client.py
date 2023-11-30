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
import contextlib
import ctypes
import logging
import os
import threading
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Optional

import grpc
import numpy as np

from thing import thing_pb2, thing_pb2_grpc
from thing.utils import _get_default_logger, _numpy_dtypes


class ThingClient:
    """
    The main class used in the data-fetching client.
    """

    def __init__(
        self,
        server_url: str,
        server_port: str | int,
        logger: logging.Logger = None,
        chunk_size: int = 32 * 1024,
    ):
        self._server_url = server_url
        self._server_port = str(server_port)
        self._logger = logger or _get_default_logger()
        self._chunk_size = chunk_size

        self.server_available = None
        self._thread_pool = ThreadPoolExecutor()
        self._id_counter = 0
        self._id_lock = threading.Lock()

    def _test_server(self) -> bool:
        with self._set_channel_stub() as stub:
            server = self._server_url + ":" + self._server_port
            try:
                response = stub.HealthCheck(thing_pb2.HealthCheckRequest())
                assert response.status == thing_pb2.STATUS.SUCCESS
                self._logger.info(f"Server {server} is available.")
                return True
            except:
                self._logger.error(
                    f"Server {server} is not available. All functions will become no-ops."
                )
                return False

    @contextlib.contextmanager
    def _set_channel_stub(self):
        _channel = grpc.insecure_channel(self._server_url + ":" + self._server_port)
        _stub = thing_pb2_grpc.ThingStub(_channel)
        yield _stub

    def _catch_numpy(self, array, framework=thing_pb2.FRAMEWORK.NUMPY) -> bool:
        with self._id_lock:  # avoid clashing ids
            id = self._id_counter
            self._id_counter += 1

        if array.dtype.name in _numpy_dtypes:
            dtype = _numpy_dtypes[array.dtype.name]
        else:
            return False

        # Use ctypes instead of `array.tobytes()` to avoid creating a copy
        data = (ctypes.c_char * array.nbytes).from_address(array.ctypes.data).raw

        with self._set_channel_stub() as stub:
            for i in range(0, len(data), self._chunk_size):
                response = stub.CatchArray(
                    thing_pb2.CatchArrayRequest(
                        id=id,
                        shape=array.shape,
                        dtype=dtype,
                        framework=framework,
                        data=data[i : i + self._chunk_size],
                        chunk_id=i // self._chunk_size,
                        num_chunks=(len(data) + self._chunk_size - 1)
                        // self._chunk_size,
                    )
                )
                if response.status != thing_pb2.STATUS.SUCCESS:
                    return False

        return True

    def _catch_torch(self, array) -> bool:
        # We have to unfortunately detach and offload the array to CPU which may cause a sync
        array = array.detach().cpu().numpy(force=False)  # force=False to avoid a copy
        return self._catch_numpy(array, framework=thing_pb2.FRAMEWORK.TORCH)

    def _catch_jax(self, array) -> bool:
        array = np.array(array, copy=False)
        return self._catch_numpy(array, framework=thing_pb2.FRAMEWORK.JAX)

    def catch(self, array) -> Optional[Future]:
        """
        Catch an array.
        Args:
            array: the array to be caught. It can be
                - a numpy array
                - a torch tensor
                - a jax array
                We however do not import any of these libraries to avoid overhead.
        Returns:
            The future object for the request. It will be up to the user to
            decide whether to block on the future or not.
            If the server is not available, it will return None.
        """
        if self.server_available is None:  # Ping the server on the first call
            self.server_available = self._test_server()

        if not self.server_available:
            return None

        # Sacrifice type-check robustness for performance
        if (
            str(array.__class__) == "<class 'numpy.ndarray'>"
        ):  # Numpy array naming should be stable enough
            future = self._thread_pool.submit(self._catch_numpy, array)
        elif (
            str(array.__class__) == "<class 'torch.Tensor'>"
        ):  # Torch tensor naming should be stable enough
            future = self._thread_pool.submit(self._catch_torch, array)
        elif "ArrayImpl" in str(
            array.__class__
        ):  # May not be robust since JAX makes changes frequently
            future = self._thread_pool.submit(self._catch_jax, array)
        else:
            raise NotImplementedError(f"Unsupported array type {array.__class__}")

        return future

    def __del__(self):
        if self._thread_pool:
            self._thread_pool.shutdown(wait=True)
