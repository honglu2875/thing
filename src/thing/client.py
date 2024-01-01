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
import inspect
import logging
import re
from collections import defaultdict
from concurrent.futures import Future, ThreadPoolExecutor
from numbers import Number
from typing import Any, Callable, Optional

import grpc
import numpy as np

from thing import thing_pb2, thing_pb2_grpc
from thing.type import ArrayLike, Awaitable
from thing.utils import (_get_framework, _is_tensor, _numpy_dtypes,
                         _prepare_array, _prepare_pytree_obj, _prepare_string,
                         _to_bytes_no_copy, _to_numpy_no_copy,
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
        self._thread_pool = ThreadPoolExecutor()

        self._server_availability = {}

        # implement `.catch(..., every=k)` by keeping track of the counter
        self._every_counter = defaultdict(lambda: 0)

        # used for `self._parse_var_name`
        # e.g. `self._var_name_counter["v"] = 1, the next `v` from a different location
        #      will be called `v_2``
        self._var_name_counter = defaultdict(lambda: 0)
        self._scope_info_to_vars = defaultdict(dict)

    def _test_server(self, server: str) -> bool:
        with self._set_channel_stub(server) as stub:
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

    def _parse_var_name(self, prev_frame: inspect.FrameInfo) -> Optional[str]:
        """Parse the variable name from the trace info.

        Note that in different local scopes, variable names can easily collide.
        The way I do here is to use the previous function names and file location
        as the hash. If variables from different functions or defined in different
        files have the same name, we append "_1", "_2", ...
        Args:
            prev_frame (inspect.FrameInfo):
        Returns:
            a unique variable name based on the full trace of frame info.
            None if the previous frame is invalid.
        """
        # Assume the current scope was `self.catch`, we start from the second
        scope_info = f"{prev_frame.filename}-{prev_frame.function}"
        if not prev_frame.code_context:
            return None
        context = prev_frame.code_context[0]
        match = re.search(r"catch\( *([^,\) ]*)", context)
        if not match:
            return None
        root_name = match.group(1).replace(".", "_")

        if root_name not in self._scope_info_to_vars[scope_info]:
            # If the variable root name is new to the previous scope
            if self._var_name_counter[root_name] > 0:
                var_name = f"{root_name}_{self._var_name_counter[root_name]}"
            else:
                var_name = root_name
            self._var_name_counter[root_name] += 1
            self._scope_info_to_vars[scope_info][root_name] = var_name
        else:
            # If under the previous scope, a variable with the same root is already captured
            var_name = self._scope_info_to_vars[scope_info][root_name]

        return var_name

    def _catch_array(
        self,
        idx: int,
        array: ArrayLike,
        name: Optional[str] = None,
        server: Optional[str] = None,
    ) -> bool:
        """
        The core of array catching. It underlies the `_catch_numpy`, `_catch_torch`,
        and `_catch_jax` calls.

        It only receives a numpy array (already converted from torch or jax),
        and sends it to the server in bytes and in chunks without creating copies.
        """
        framework = _get_framework(array)
        array = _to_numpy_no_copy(array, framework)

        if array.dtype.name in _numpy_dtypes:
            dtype = _numpy_dtypes[array.dtype.name]
        else:
            self._logger.error(f"Unsupported dtype {array.dtype}")
            return False

        data = _to_bytes_no_copy(array)

        try:
            with self._set_channel_stub(server) as stub:
                num_chunks = (len(data) + self._chunk_size - 1) // self._chunk_size
                singleton = num_chunks == 1
                for i in range(0, len(data), self._chunk_size):
                    self._logger.info(
                        f"Sending tensor {name or '<noname>'} of shape {array.shape} and "
                        f"type {array.dtype.name}. "
                        f"Chunk {i // self._chunk_size} of {len(data) // self._chunk_size}."
                    )
                    response = stub.CatchArray(
                        _prepare_array(
                            data[i : i + self._chunk_size],
                            idx=idx,
                            name=name,
                            shape=array.shape,
                            dtype=dtype,
                            framework=framework,
                            chunk_id=None if singleton else i // self._chunk_size,
                            num_chunks=None if singleton else num_chunks,
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
        self, idx: int, string, name: Optional[str] = None, server: Optional[str] = None
    ) -> bool:
        try:
            with self._set_channel_stub(server) as stub:
                response = stub.CatchString(
                    _prepare_string(string, idx=idx, name=name),
                    timeout=self._timeout,
                )
                if response.status != thing_pb2.STATUS.SUCCESS:
                    return False
        except Exception as e:
            self._logger.error(f"Error when sending string: {e}")
            return False

        return True

    def _catch_pytree_schema(
        self,
        id: int,
        root: thing_pb2.PyTreeNode,
        name: Optional[str] = None,
        server: Optional[str] = None,
    ) -> bool:
        try:
            with self._set_channel_stub(server) as stub:
                response = stub.CatchPyTree(
                    root,
                    timeout=self._timeout,
                )
                if response.status != thing_pb2.STATUS.SUCCESS:
                    return False
        except Exception as e:
            self._logger.error(f"Error when sending pytree schema: {e}")
            return False
        return True

    def _async_catch_objects(
        self,
        ids: list[int],
        objects: list,
        names: list[Optional[str]],
        server: Optional[str] = None,
    ) -> bool:
        """Send a batch of objects asynchronously."""
        futures = []
        with ThreadPoolExecutor(len(ids)) as pool:
            for idx, obj, name in zip(ids, objects, names):
                # For scalars, default to numpy array with shape ()
                if isinstance(obj, Number):
                    obj = np.array(obj)

                if isinstance(obj, str):
                    _fn = self._catch_string
                elif isinstance(obj, thing_pb2.PyTreeNode):
                    _fn = self._catch_pytree_schema
                elif _is_tensor(obj):
                    _fn = self._catch_array
                else:
                    self._logger.error(f"Unsupported array type {obj.__class__}")
                    continue

                futures.append(pool.submit(_fn, idx, obj, name=name, server=server))

        return all(f.result() for f in futures)

    def _catch(
        self,
        obj: Any,
        name: Optional[str] = None,
        past_frame: Optional[Any] = None,
        server: Optional[str] = None,
        every: int = 1,
    ) -> bool:
        """The inner synchronous call of `.catch`"""
        # None will show up as empty string in the recipient server
        obj = "" if obj is None else obj

        server = _validate_server_name(server, self._logger) or self.target_server
        # Ping server on the first call
        if server not in self._server_availability:
            self._server_availability[server] = self._test_server(server)
        # If first ping was unsuccessful, make everything no-op to avoid overhead
        if not self._server_availability[server]:
            return None

        if past_frame is not None and name is None:
            # `inspect.getframeinfo` is expensive! We are fine if calls are sparse.
            # But ideally, specifying a name would avoid this.
            name = self._parse_var_name(inspect.getframeinfo(past_frame))
        # If parsing fails, we use the memory address as name
        name = name or f"p_{id(obj)}"

        self._every_counter[name] += 1
        if every > 1 and self._every_counter[name] % every != 0:
            return None

        if obj is None or isinstance(obj, (tuple, list, dict)):
            # Parse a tuple/list/dict as a PyTree object
            root, id_to_leaves = _prepare_pytree_obj(obj, name=name)
            ids, objects, names = [], [], []
            # Send all the leaves first
            for k, v in id_to_leaves.items():
                ids.append(k)
                objects.append(v)
                # Intermediate nodes do not have names
                names.append(None)
            # Send the pytree object structure
            ids.append(root.id)
            objects.append(root)
            names.append(None)
            return self._async_catch_objects(ids, objects, names, server=server)
        else:
            return self._async_catch_objects(
                [get_rand_id()], [obj], [name], server=server
            )

    def catch(
        self,
        obj: Any,
        name: Optional[str] = None,
        server: Optional[str] = None,
        every: int = 1,
    ) -> Awaitable:
        """
        Catch a supported object.

        Args:
            obj: the object to be caught. It can be
                - a numpy array
                - a torch tensor
                - a jax array
                - an int/float (will be converted to numpy)
                - a list or tuple
                - a dict
                - a string
                - None (will be converted to empty string)
                We however do not import any extra libraries to avoid overhead.
            name: the name of the variable. If not provided, it will:
                - Try to trace to the previous scope and attempt to capture the
                  variable name. Same variable names from different functions
                  will be separated by appending `_1`, `_2`, etc.
                - If failed, the variable name becomes `p_` followed by its pointer
                  address.
            server: a custom server address and port if different from default.
                Must be in the form of "[address]:[port]".
            every: catch every `every`-th array.
        Returns:
            An `Awaitable` object (see `thing.type`).
            Two types of usage:
              - `thing.catch(x).wait()` to wait for the execution successfulness.
              - `thing.catch(x)` to only send the payloads without waiting.
        """
        # Capture the previous frame before entering a thread pool
        # This incurs almost zero overhead.
        past_frame = inspect.currentframe().f_back
        # Note:
        #   There is an `inspect.getframeinfo` call inside `self._catch`. This incurs
        #   big overhead as it will perform an I/O. Must hide the latency inside a
        #   thread pool. Or, if the `name` argument is given, this will not be called.
        return Awaitable(
            self._thread_pool.submit(
                self._catch,
                obj,
                past_frame=past_frame,
                name=name,
                server=server,
                every=every,
            )
        )

    @property
    def target_server(self):
        return self._server_addr + ":" + self._server_port

    @target_server.setter
    def target_server(self, target):
        self._server_addr = target.split(":")[0]
        self._server_port = target.split(":")[1]

    def __del__(self):
        if self._thread_pool is not None:
            self._thread_pool.shutdown(wait=True)
