import logging
import os

from .argument import ServicerArguments
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

# Client API
server_url = os.environ.get("THING_SERVER", "localhost")
port = os.environ.get("THING_PORT", 2875)
client = Client(server_addr=server_url, server_port=port)
catch = client.catch


# Server API
server = None


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


__all__ = ["catch", "serve", "Server", "Client"]
