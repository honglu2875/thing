import os

from .client import ThingClient as Client
from .server import Server
from .utils import _set_up_logger

_set_up_logger(__name__)

"""
Expose a default client for convenient usage. The overhead should be minimal as
the client instance takes up minimal resources if not used.

Environment variables:
  - `THING_SERVER` determines the server address (default to `localhost`)
  - `THING_PORT` determines the port (default to port 2875)
"""

server_url = os.environ.get("THING_SERVER", "localhost")
port = os.environ.get("THING_PORT", 2875)
client = Client(server_addr=server_url, server_port=port)
catch = client.catch
__all__ = ["catch", "Server", "Client"]
