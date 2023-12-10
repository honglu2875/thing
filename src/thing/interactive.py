import logging
import sys
import time
from collections import OrderedDict
from datetime import datetime
from typing import Union

import rich
from rich.console import Console

import thing
from thing.server import Server
from thing.store import HistoryRecord

# Whether the note for `.get` has already been displayed once
_get_note_displayed: bool = False
_all = [
    ("serve", (), "start the server."),
    ("status", (), "show the status and the last a few logs."),
    ("get", ("id_or_name",), "get a tensor by its name or id."),
    ("get_all", ("name",), "get all tensors with the same name."),
    (
        "ingest_all",
        (),
        "ingest all named tensors directly into your python REPL session.",
    ),
    ("summary", (), "show a summary of recently received tensors."),
    ("describe", ("id_or_name",), "describe a tensor and its metadata in details."),
]


def status():
    """
    The command-line frontend for checking the status of the default server.
    """
    server = thing.server

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


def get(item: Union[int, str], index: int = 0):
    """
    Command-line frontend for getting tensors by either its name or its id.
    """
    server = thing.server
    global _get_note_displayed

    if server is None:
        rich.print("Server is not running.")
        return None

    if isinstance(item, int):
        return server.store.get_tensor_by_id(item)
    elif isinstance(item, str):
        hist_len = server.store.get_len(item)
        if not _get_note_displayed:
            if hist_len > 1:
                rich.print("Note:")
                rich.print(
                    f"  There are {hist_len} tensors with the same name `{item}`."
                )
                rich.print(f"  To get the i-th latest version, use `.get(name, i)`.")
        _get_note_displayed = True
        return server.store.get_tensor_by_name(item, index)
    else:
        raise TypeError(
            f"The argument must be either a string or an int, got {type(item)}."
        )


def get_all(name: str):
    """
    Command-line frontend for getting all tensors with the same name.
    """
    server = thing.server

    if server is None:
        rich.print("Server is not running.")
        return None

    return server.store.get_all_by_name(name)


def summary():
    server = thing.server

    if server is None:
        rich.print("Server is not running.")
        return None

    _received_tensors = OrderedDict()
    _received_noname_tensors = OrderedDict()
    _NUM_LOGS = float("inf")  # maybe limit the number of logs in the future
    # Scan _NUM_LOGS logs and write a summary
    for i, log in enumerate(server.store.get_history()):
        if i >= _NUM_LOGS:
            break
        if not log.name:
            _received_noname_tensors[log.id] = server.store.by_id(log.id)
        else:
            _received_tensors[log.name] = server.store.by_name(log.name)

    if _received_tensors:
        rich.print("Received tensors:")
        for name, tensor in _received_tensors.items():
            hist_len = server.store.get_len(name)
            if hist_len > 1:
                rich.print(f"  {name}: {tensor.shape} (latest, total {hist_len})")
            else:
                rich.print(f"  {name}: {tensor.shape}")

    if _received_noname_tensors:
        rich.print("Received unnamed tensors:")
        for idx, tensor in _received_noname_tensors.items():
            rich.print(f"  id={idx}: {tensor.shape}")


def _show(log: HistoryRecord, console: Console):
    dt = datetime.fromtimestamp(log.timestamp)
    console.print(
        f"{dt} - Received tensor `{log.name or '[unnamed]'}` with id {log.id} "
        f"and shape {log.shape} from {log.client_addr}."
    )


def serve():
    logger = logging.getLogger(__name__)

    if thing.server is None:
        port = thing.get_port()
        thing.server = Server(port=port)
        try:
            thing.server.start()
            logger.info(f"Server started at port {port}.")
        except Exception as e:
            logger.error(f"Failed to start server: {e}")
            thing.server = None


def ingest_all(overwrite=False):
    server = thing.server
    console = Console()

    if overwrite:
        rich.print(
            "Warning: this will write all named tensors directly into your `globals()`."
        )
        rich.print(
            "         You toggled the `overwrite` flag, so it will overwrite existing variables."
        )
        if (
            console.input(
                "Are you sure to ingest all tensors with overwriting? [y/N] "
            ).lower()
            != "y"
        ):
            return

    if server is not None:
        session_globals = sys._getframe(1).f_globals
        for name in server.store.get_names():
            if not overwrite and name in globals():
                rich.print(
                    f"Skipping tensor `{name}` because a variable under this name already exists."
                )
                continue
            try:
                tensor = server.store.get_tensor_by_name(name)
                session_globals[name] = tensor
            except Exception as e:
                rich.print(f"Failed to ingest tensor `{name}`: {e}")


def describe(item: Union[str, int]):
    """
    Command-line frontend for describing a tensor and its metadata.

    Args:
        item: the name or id of the tensor.
    """
    server = thing.server

    if isinstance(item, str):
        obj = server.store.by_name(item)
    elif isinstance(item, int):
        obj = server.store.by_id(item)
    else:
        raise TypeError(f"Unsupported type {type(item)}")

    rich.print(f"Name       : {obj.name}")
    rich.print(f"Id         : {obj.id}")
    rich.print(f"Shape      : {obj.shape}")
    rich.print(f"Received at: {datetime.fromtimestamp(obj.timestamp)}")
    rich.print(f"Client addr: {obj.client_addr}")
    receive_count = len(server.store.get_len(obj.name))
    if receive_count > 1:
        rich.print(f"Received {receive_count} times under the same name.")
        rich.print(
            "To retrieve a historical version, use `.get_tensor_by_name(name, index)`."
        )
        rich.print(
            "Example:\n"
            "    `.get_tensor_by_name('my_tensor', 0)` for the latest version.\n"
            "    `.get_tensor_by_name('my_tensor', 1)` for the second latest version."
        )


def help():
    """
    The helper for interactive sessions running as the server.
    """
    rich.print("Available commands for interactive Python REPL:")
    for name, args, desc in _all:
        rich.print(f"  .{name}({', '.join(args)}) - {desc}")
