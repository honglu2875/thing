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
from thing.utils import _numpy_dtypes, _validate_server_name


class ThingClient:
    """
    The main class used in the data-fetching client. "Thing" moves as silent and as non-invasive
    as possible to retrieve those live tensors for you.
    A few important notes:
      - On the first call of `catch`, a health-check message will be sent. If unavailable,
        the functionality will become permanent no-op for the life-cycle of the client object.
        This design is to protect the cases when you still have these debugging codes with `catch`
        but do not intend to spin up a server.
      - The rpc channel and stub are created and destroyed on each call of `catch`. I am not aware
        of a big over-head over this.
      - `catch` returns a future object. You can either invoke `Future.result()` in the code to block,
        or just forget about it so that it runs asynchronously.
    """

    def __init__(
        self,
        server_addr: str,
        server_port: str | int,
        logger: Optional[logging.Logger] = None,
        chunk_size: int = 32 * 1024,
    ):
        self._server_addr = server_addr
        self._server_port = str(server_port)
        self._logger = logger or logging.getLogger(__name__)
        self._chunk_size = chunk_size

        self.server_available = None
        self._thread_pool = ThreadPoolExecutor()
        self._id_counter = 0
        self._id_lock = threading.Lock()

    def _test_server(self) -> bool:
        with self._set_channel_stub() as stub:
            server = self._server_addr + ":" + self._server_port
            try:
                response = stub.HealthCheck(thing_pb2.HealthCheckRequest())
                assert response.status == thing_pb2.STATUS.SUCCESS
                self._logger.info(f"Server {server} is available.")
                return True
            except grpc.RpcError:
                self._logger.error(
                    f"Server {server} is not available. All functions will become no-ops."
                )
                return False
            except Exception as e:
                self._logger.error(f"Unknown error when testing server: {e}")
                return False

    @contextlib.contextmanager
    def _set_channel_stub(self, server: Optional[str] = None):
        server = server or self._server_addr + ":" + self._server_port
        _channel = grpc.insecure_channel(server)
        _stub = thing_pb2_grpc.ThingStub(_channel)
        yield _stub

    def _catch_numpy(
        self,
        array,
        var_name: Optional[str] = None,
        framework=thing_pb2.FRAMEWORK.NUMPY,
        server: Optional[str] = None,
    ) -> bool:
        with self._id_lock:  # avoid clashing ids
            id = self._id_counter
            self._id_counter += 1

        if array.dtype.name in _numpy_dtypes:
            dtype = _numpy_dtypes[array.dtype.name]
        else:
            self._logger.error(f"Unsupported dtype {array.dtype}")
            return False

        # Use ctypes instead of `array.tobytes()` to avoid creating a copy
        data = (ctypes.c_char * array.nbytes).from_address(array.ctypes.data).raw

        try:
            with self._set_channel_stub(server) as stub:
                for i in range(0, len(data), self._chunk_size):
                    self._logger.info(
                        f"Sending tensor {var_name or '<noname>'} of shape {array.shape} and "
                        f"type {array.dtype.name}. "
                        f"Chunk {i // self._chunk_size} of {len(data) // self._chunk_size}."
                    )
                    response = stub.CatchArray(
                        thing_pb2.CatchArrayRequest(
                            id=id,
                            var_name=var_name,
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
        except Exception as e:
            self._logger.error(f"Error when sending array: {e}")
            return False

        return True

    def _catch_torch(
        self, array, var_name: Optional[str] = None, server: Optional[str] = None
    ) -> bool:
        # We have to unfortunately detach and offload the array to CPU which may cause a sync
        array = array.detach().cpu().numpy(force=False)  # force=False to avoid a copy
        return self._catch_numpy(
            array, var_name=var_name, framework=thing_pb2.FRAMEWORK.TORCH, server=server
        )

    def _catch_jax(
        self, array, var_name: Optional[str] = None, server: Optional[str] = None
    ) -> bool:
        array = np.array(array, copy=False)
        return self._catch_numpy(
            array, var_name=var_name, framework=thing_pb2.FRAMEWORK.JAX, server=server
        )

    def catch(
        self, array, var_name: Optional[str] = None, server: Optional[str] = None
    ) -> Optional[Future]:
        """
        Catch an array.
        Args:
            array: the array to be caught. It can be
                - a numpy array
                - a torch tensor
                - a jax array
                We however do not import any of these libraries to avoid overhead.
            server: a custom server address and port if different from default.
                Must be in the form of "[address]:[port]".
        Returns:
            The future object for the request. It will be up to the user to
            decide whether to block on the future or not.
            If the server is not available, it will return None.
        """
        server = _validate_server_name(server, self._logger)
        if (
            self.server_available is None and not server
        ):  # Ping the default server on the first call
            self.server_available = self._test_server()

        if not self.server_available and not server:
            return None

        # Sacrifice type-check robustness for performance
        if (
            str(array.__class__) == "<class 'numpy.ndarray'>"
        ):  # Numpy array naming should be stable enough
            future = self._thread_pool.submit(
                self._catch_numpy, array, var_name=var_name, server=server
            )
        elif (
            str(array.__class__) == "<class 'torch.Tensor'>"
        ):  # Torch tensor naming should be stable enough
            future = self._thread_pool.submit(
                self._catch_torch, array, var_name=var_name, server=server
            )
        elif "ArrayImpl" in str(
            array.__class__
        ):  # May not be robust since JAX makes changes frequently
            future = self._thread_pool.submit(
                self._catch_jax, array, var_name=var_name, server=server
            )
        else:
            self._logger.error(f"Unsupported array type {array.__class__}")
            return None

        return future

    @property
    def target_server(self):
        return self._server_addr + ":" + self._server_port

    @target_server.setter
    def target_server(self, target):
        self._server_addr = target.split(":")[0]
        self._server_port = target.split(":")[1]

    def __del__(self):
        if self._thread_pool:
            self._thread_pool.shutdown(wait=True)
