import logging
import os
import time
from datetime import datetime
from typing import Optional, Union

from rich.console import Console

from .argument import ServicerArguments
from .client import ThingClient as Client
from .server import Server
from .store import HistoryRecord
from .utils import _set_up_logger

_set_up_logger(__name__)

"""
Expose a default client for convenient usage. The overhead should be minimal as
the client instance takes up minimal resources if not used.

Environment variables:
  - `THING_SERVER` determines the server address (default to `localhost`)
  - `THING_PORT` determines the port (default to port 2875)
"""

# Client API
server_url = os.environ.get("THING_SERVER", "localhost")
port = os.environ.get("THING_PORT", 2875)
client = Client(server_addr=server_url, server_port=port)
catch = client.catch

# Server API
server: Optional[Server] = None


def serve():
    global server
    logger = logging.getLogger(__name__)

    if server is None:
        server = Server(port=port)
        try:
            server.start()
            logger.info(f"Server started at port {port}.")
        except Exception as e:
            logger.error(f"Failed to start server: {e}")
            server = None


def _show(log: HistoryRecord, console: Console):
    dt = datetime.fromtimestamp(log.timestamp)
    console.print(
        f"{dt} - Received tensor `{log.name or '[unnamed]'}` with id {log.id} "
        f"and shape {log.shape} from {log.client_addr}."
    )


def status():
    """
    The command-line frontend for checking the status of the default server.
    """
    global server

    if server is None:
        return "Server is not running."

    _NUM_LOGS = 3
    console = Console()
    _msg = [
        "Server is running...",
        "Press Ctrl+C to go back to the console (server will keep running).",
    ]
    with console.status(spinner="dots9", status="\n".join(_msg)):
        try:
            _logs = []
            for i, log in enumerate(server.store.get_history()):
                if i >= _NUM_LOGS:
                    break
                _logs.append(log)

            for log in _logs[::-1]:
                _show(log, console)

            while True:
                new_logs = []
                for log in server.store.get_history():
                    if _logs and id(log) == id(_logs[-1]):
                        break
                    new_logs.append(log)

                for log in new_logs[::-1]:
                    _show(log, console)
                    _logs.append(log)

                time.sleep(0.1)
        except KeyboardInterrupt:
            pass


def get(item: Union[int, str]):
    """
    Command-line frontend for getting tensors by either its name or its id.
    """
    global server

    if server is None:
        return "Server is not running."

    if isinstance(item, int):
        try:
            return server.store.get_tensor_by_id(item)
        except KeyError:
            return f"Item with id {item} does not exist."
    elif isinstance(item, str):
        try:
            return server.store.get_tensor_by_name(item)
        except KeyError:
            return f"Item with name {item} does not exist."
    else:
        raise TypeError(f"The argument must be either a string or an int, got {type(item)}.")


def summary():
    ...
    # TODO


__all__ = [
    "catch", "Client",
    "serve", "status", "get", "summary", "Server",
]
