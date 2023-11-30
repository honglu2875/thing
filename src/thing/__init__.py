import os

from .client import ThingClient
from .server import Server

"""
determine the gRPC server based on the env vars `THING_SERVER` and `THING_PORT`
  - `THING_SERVER` determines the server url (default to `localhost`)
  - `THING_PORT` determines the port (default to port 2875)
"""
server_url = os.environ.get("THING_SERVER", "localhost")
port = os.environ.get("THING_PORT", 2875)
client = ThingClient(server_url=server_url, server_port=port)

catch = client.catch
__all__ = ["catch", "Server"]
