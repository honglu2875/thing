import os
import secrets
from typing import Optional

from .argument import ServicerArguments
from .client import ThingClient as Client
from .interactive import (describe, get, get_all, help, ingest_all, serve,
                          status, summary)
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
  
On the package level, there are default instances of `Client` and `Server` objects,
so that both client and server can easily access by making `thing.<method>` calls.
"""


# Client API


def get_server_url() -> str:
    return os.environ.get("THING_SERVER", "localhost")


def get_port() -> str:
    return os.environ.get("THING_PORT", 2875)


client = Client(server_addr=get_server_url(), server_port=get_port())
catch = client.catch

# Server API
# Note: the interactive server API is implemented in `interactive.py`
server: Optional[Server] = None

__all__ = [
    # Client object
    "Client",
    # Client APIs
    "catch",
    "get_server_url",
    "get_port",
    # Server object
    "Server",
    # Server APIs
    "status",
    "get",
    "get_all",
    "summary",
    "serve",
    "ingest_all",
    "describe",
    "help",
]
