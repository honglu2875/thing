import dataclasses
from collections import defaultdict
from datetime import datetime
from typing import Optional, Union

from thing.type import TensorObject


@dataclasses.dataclass
class HistoryRecord:
    """
    A record of the history of the store.
    """

    id: int
    name: Optional[str]
    shape: tuple
    timestamp: int
    client_addr: str


class Store:
    def __init__(self):
        self._items = {}
        self._name_to_id = defaultdict(list)
        self._history: list[HistoryRecord] = []

    def add(self, item: TensorObject):
        self._items[item.id] = item
        if item.name:
            self._name_to_id[item.name].append(item.id)
        self._history.append(
            HistoryRecord(
                id=item.id,
                name=item.name,
                shape=item.shape,
                timestamp=item.timestamp,
                client_addr=item.client_addr,
            )
        )

    def get_history(self):
        for log in self._history[::-1]:
            yield log

    def by_id(self, idx: int) -> TensorObject:
        if idx not in self._items:
            raise KeyError(f"Item with id {idx} does not exist.")
        return self._items[idx]

    def by_name(self, name: str, index: int = 0) -> TensorObject:
        if not self._name_to_id[name]:
            raise KeyError(f"Item with name {name} does not exist.")
        return self._items[self._name_to_id[name][-(1 + index)]]

    def get_tensor_by_id(self, idx: int):
        """
        Retrieve the tensor received under the given id.

        Args:
            idx: the id of the item.
        Returns:
            The corresponding tensor.
        """
        return self.by_id(idx).data

    def get_tensor_by_name(self, name: str, index: int = 0):
        """
        Retrieve the tensor received under the given name. If there are multiples, return the one with the
        given index (ordered by when they are added).

        Args:
            name: the name of the item.
            index: the index of the item under the given name.
        Returns:
            The corresponding tensor.
        """
        return self.by_name(name, index).data

    def get_all_by_name(self, name: str):
        """
        Retrieve all tensors received under the given name.

        Args:
            name: the name of the item.
        Returns:
            The corresponding tensors.
        """
        if not self._name_to_id[name]:
            raise KeyError(f"Item with name {name} does not exist.")
        return [self._items[idx].data for idx in self._name_to_id[name][::-1]]

    def get_len(self, name: str):
        """
        Get the number of historical tensors received under the given name.
        """
        if not self._name_to_id[name]:
            return 0
        return len(self._name_to_id[name])

    def describe(self, item: Union[str, int]):
        """
        Command-line frontend for describing a tensor and its metadata.

        Args:
            item: the name or id of the tensor.
        """
        import rich

        if isinstance(item, str):
            obj = self.by_name(item)
        elif isinstance(item, int):
            obj = self.by_id(item)
        else:
            raise TypeError(f"Unsupported type {type(item)}")

        rich.print(f"Name       : {obj.name}")
        rich.print(f"Id         : {obj.id}")
        rich.print(f"Shape      : {obj.shape}")
        rich.print(f"Received at: {datetime.fromtimestamp(obj.timestamp)}")
        rich.print(f"Client addr: {obj.client_addr}")
        if len(self._name_to_id[obj.name]) > 1:
            rich.print(
                f"Received {len(self._name_to_id[obj.name])} times under the same name."
            )
            rich.print(
                "To retrieve a historical version, use `.get_tensor_by_name(name, index)`."
            )
            rich.print(
                "Example:\n"
                "    `.get_tensor_by_name('my_tensor', 0)` for the latest version.\n"
                "    `.get_tensor_by_name('my_tensor', 1)` for the second latest version."
            )

    def __getitem__(self, item: Union[str, int]):
        if isinstance(item, str):  # if string, assume it's a name
            return self.get_tensor_by_name(item)
        elif isinstance(item, int):  # if int, assume it's an id
            return self.get_tensor_by_id(item)
        else:
            raise TypeError(f"Unsupported type {type(item)}")

    def __len__(self):
        return len(self._items)

    # todo: add support for arbitrary pytree objects
