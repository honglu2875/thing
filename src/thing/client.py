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
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from numbers import Number
from typing import Any, Optional

import grpc
import numpy as np

from thing import thing_pb2, thing_pb2_grpc
from thing.type import Futures
from thing.utils import (_numpy_dtypes, _prepare_pytree_obj,
                         _validate_server_name, get_rand_id)


class ThingClient:
    """
    The main class used in the data-fetching client. "Thing" moves as silent and as non-invasive
    as possible to retrieve those live tensors for you.
    A few important notes:
      - On the first call of `catch`, a health-check message will be sent. If unavailable,
        the functionality will become permanently disabled for the life-cycle of the client object.
        This design is to protect the cases when you still have these debugging codes with `catch`
        but do not intend to spin up a server.
      - The rpc channel and stub are created and destroyed on each call of `catch`. I am not aware
        of a big over-head over this.
      - `catch` returns a future object. You can either invoke `Future.result()` in the code to block,
        or just forget about it so that it runs asynchronously.
    """

    _timeout: float = 5.0

    def __init__(
        self,
        server_addr: str,
        server_port: str | int,
        logger: Optional[logging.Logger] = None,
        chunk_size: int = 128 * 1024,
        logging_level: str = "ERROR",
    ):
        self._server_addr = server_addr
        self._server_port = str(server_port)
        self._logger = logger or logging.getLogger(__name__)
        self._logger.setLevel(logging_level)
        self._chunk_size = chunk_size

        self.server_available = None
        self._thread_pool = ThreadPoolExecutor()

        self._every_counter = defaultdict(
            lambda: 0
        )  # implement `.catch(..., every=k)` by keeping track of the counter

    def _test_server(self) -> bool:
        with self._set_channel_stub() as stub:
            server = self._server_addr + ":" + self._server_port
            try:
                response = stub.HealthCheck(
                    thing_pb2.HealthCheckRequest(), timeout=self._timeout
                )
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

    def _catch(
        self,
        idx: int,
        array: np.ndarray,
        name: Optional[str] = None,
        server: Optional[str] = None,
        framework=thing_pb2.FRAMEWORK.NUMPY,
    ) -> bool:
        """
        The core of array catching. It only receives a numpy array (already converted from torch or jax),
        and sends it to the server in bytes and in chunks without creating copies.
        """
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
                        f"Sending tensor {name or '<noname>'} of shape {array.shape} and "
                        f"type {array.dtype.name}. "
                        f"Chunk {i // self._chunk_size} of {len(data) // self._chunk_size}."
                    )
                    response = stub.CatchArray(
                        thing_pb2.CatchArrayRequest(
                            id=idx,
                            var_name=name,
                            shape=array.shape,
                            dtype=dtype,
                            framework=framework,
                            data=data[i : i + self._chunk_size],
                            chunk_id=i // self._chunk_size,
                            num_chunks=(len(data) + self._chunk_size - 1)
                            // self._chunk_size,
                        ),
                        timeout=self._timeout,
                    )
                    if response.status != thing_pb2.STATUS.SUCCESS:
                        return False
        except Exception as e:
            self._logger.error(f"Error when sending array: {e}")
            return False

        return True

    def _catch_string(
        self, idx: int, array, name: Optional[str] = None, server: Optional[str] = None
    ) -> bool:
        try:
            with self._set_channel_stub(server) as stub:
                response = stub.CatchString(
                    thing_pb2.CatchStringRequest(
                        id=idx,
                        var_name=name,
                        data=array,
                    ),
                    timeout=self._timeout,
                )
                if response.status != thing_pb2.STATUS.SUCCESS:
                    return False
        except Exception as e:
            self._logger.error(f"Error when sending string: {e}")
            return False

        return True

    def _catch_pytree_structure(self, root, server: Optional[str] = None) -> bool:
        try:
            with self._set_channel_stub(server) as stub:
                response = stub.CatchPyTree(
                    root,
                    timeout=self._timeout,
                )
                if response.status != thing_pb2.STATUS.SUCCESS:
                    return False
        except Exception as e:
            self._logger.error(f"Error when sending pytree structure: {e}")
            return False

        return True

    def _catch_torch(
        self, idx: int, array, name: Optional[str] = None, server: Optional[str] = None
    ) -> bool:
        # We have to unfortunately detach and offload the array to CPU which may cause a sync
        array = array.detach().cpu().numpy(force=False)  # force=False to avoid a copy
        return self._catch(
            idx, array, name=name, framework=thing_pb2.FRAMEWORK.TORCH, server=server
        )

    def _catch_jax(
        self, idx: int, array, name: Optional[str] = None, server: Optional[str] = None
    ) -> bool:
        array = np.array(array, copy=False)
        return self._catch(
            idx, array, name=name, framework=thing_pb2.FRAMEWORK.JAX, server=server
        )

    def _catch_numpy(
        self, idx: int, array, name: Optional[str] = None, server: Optional[str] = None
    ) -> bool:
        return self._catch(
            idx, array, name=name, framework=thing_pb2.FRAMEWORK.NUMPY, server=server
        )

    def _catch_tensors_and_strings(
        self,
        ids: list[int],
        objects: list,
        names: list[Optional[str]],
        server: Optional[str] = None,
    ) -> Futures:
        futures = []
        for idx, obj, name in zip(ids, objects, names):
            # For scalars, default to numpy array with shape ()
            if isinstance(obj, Number):
                obj = np.array(obj)

            # Sacrifice a little type-check robustness to avoid unnecessary imports
            if isinstance(obj, str):
                _fn = self._catch_string
            # Numpy array naming should be stable enough
            elif str(obj.__class__) == "<class 'numpy.ndarray'>":
                _fn = self._catch_numpy
            # Torch tensor naming should be stable enough
            elif str(obj.__class__) == "<class 'torch.Tensor'>":
                _fn = self._catch_torch
            # May not be robust since JAX makes changes frequently
            elif "ArrayImpl" in str(obj.__class__):
                _fn = self._catch_jax
            else:
                self._logger.error(f"Unsupported array type {obj.__class__}")
                continue

            futures.append(
                self._thread_pool.submit(_fn, idx, obj, name=name, server=server)
            )

        return Futures(futures)

    def catch(
        self,
        obj: Any,
        name: Optional[str] = None,
        server: Optional[str] = None,
        every: int = 1,
    ) -> Optional[Futures]:
        """
        Catch an array.
        Args:
            obj: the object to be caught. It can be
                - a numpy array
                - a torch tensor
                - a jax array
                - a list or tuple
                - a dict
                - a string
                We however do not import any of these libraries to avoid overhead.
            name: the name of the variable. If not provided, it will be None.
                In the case of None, the logger will refer to it as "<noname>".
            server: a custom server address and port if different from default.
                Must be in the form of "[address]:[port]".
            every: catch every `every`-th array. The array MUST have a name, or
                it will be ignored.
        Returns:
            The future object for the request. It will be up to the user to
            decide whether to block on the future or not.
            If the server is not available, it will return None.
        """
        if obj is None:
            return None
        server = _validate_server_name(server, self._logger)
        if (
            self.server_available is None and not server
        ):  # Ping the default server on the first call
            self.server_available = self._test_server()

        if not self.server_available and not server:
            return None

        if name is not None and every > 1:
            self._every_counter[name] += 1
            if self._every_counter[name] % every != 0:
                return None

        if obj is None or isinstance(obj, (tuple, list, dict)):
            root, id_to_leaves = _prepare_pytree_obj(obj, name=name)
            ids = []
            objects = []
            for k, v in id_to_leaves.items():
                ids.append(k)
                objects.append(v)
            # todo: how to assign names for intermediate nodes?
            names = [None] * len(ids)
            self._catch_tensors_and_strings(ids, objects, names, server=server)

            # Send the pytree object structure
            self._catch_pytree_structure(root, server=server)
        else:
            return self._catch_tensors_and_strings(
                [get_rand_id()], [obj], [name], server=server
            )

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
