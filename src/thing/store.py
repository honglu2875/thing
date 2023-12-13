import dataclasses
import logging
import time
from collections import defaultdict
from typing import Optional, Union

from thing import thing_pb2
from thing.type import Object, PyTreeObject, StringObject, TensorObject
from thing.utils import _reconstruct_pytree_obj


@dataclasses.dataclass
class HistoryRecord:
    """
    A record of the history of the store.
    """

    id: int
    type: str  # one of "tensor", "string", "pytree"
    name: Optional[str]
    shape: Optional[tuple]  # only for tensors
    timestamp: int
    client_addr: str

    def __post_init__(self):
        if self.type == "tensor":
            assert self.shape is not None
        elif self.type == "pytree":
            pass  # todo: add more metadata?
        elif self.type == "string":
            assert self.shape is None
        else:
            raise ValueError(f"Unsupported type {self.type}")


class Store:
    def __init__(self):
        self._items = {}
        self._name_to_id = defaultdict(list)
        self._history: list[HistoryRecord] = []
        self._logger = logging.getLogger(__name__)

    def _post_process(self, obj):
        if isinstance(obj, thing_pb2.PyTreeNode):
            try:
                obj = _reconstruct_pytree_obj(obj, self._items)
            except NameError:
                self._logger.error("The object has not been completely transmitted.")
        return obj

    @staticmethod
    def _get_type(item: TensorObject | StringObject | PyTreeObject):
        if isinstance(item, TensorObject):
            return "tensor"
        elif isinstance(item, StringObject):
            return "string"
        elif isinstance(item, PyTreeObject):
            return "pytree"

    def add(self, item: TensorObject | StringObject | PyTreeObject):
        self._items[item.id] = item
        if item.name:
            self._name_to_id[item.name].append(item.id)
        self._history.append(
            HistoryRecord(
                id=item.id,
                type=self._get_type(item),
                name=item.name,
                shape=item.shape if hasattr(item, "shape") else None,
                timestamp=item.timestamp,
                client_addr=item.client_addr,
            )
        )

    def get_history(self):
        for log in self._history[::-1]:
            yield log

    def by_id(self, idx: int) -> Object:
        if idx not in self._items:
            raise KeyError(f"Item with id {idx} does not exist.")
        return self._items[idx]

    def by_name(self, name: str, index: int = 0) -> Object:
        if not self._name_to_id[name]:
            raise KeyError(f"Item with name {name} does not exist.")
        return self._items[self._name_to_id[name][-(1 + index)]]

    def get_object_by_id(self, idx: int):
        """
        Retrieve the tensor received under the given id.

        Args:
            idx: the id of the item.
        Returns:
            The corresponding tensor.
        """
        return self._post_process(self.by_id(idx).data)

    def get_object_by_name(self, name: str, index: int = 0):
        """
        Retrieve the tensor received under the given name. If there are multiples, return the one with the
        given index (ordered by when they are added).

        Args:
            name: the name of the item.
            index: the index of the item under the given name.
        Returns:
            The corresponding tensor.
        """
        return self._post_process(self.by_name(name, index).data)

    def get_all_by_name(self, name: str):
        """
        Retrieve all tensors received under the given name.
        Ordered from the oldest to the latest.

        Args:
            name: the name of the item.
        Returns:
            The corresponding tensors.
        """
        if not self._name_to_id[name]:
            raise KeyError(f"Item with name {name} does not exist.")
        return [self._items[idx].data for idx in self._name_to_id[name]]

    def get_len(self, name: str):
        """
        Get the number of historical tensors received under the given name.
        """
        if not self._name_to_id[name]:
            return 0
        return len(self._name_to_id[name])

    def get_names(self):
        """
        Get the names of all tensors.
        """
        return list(self._name_to_id.keys())

    def __getitem__(self, item: Union[str, int]):
        if isinstance(item, str):  # if string, assume it's a name
            return self.get_object_by_name(item)
        elif isinstance(item, int):  # if int, assume it's an id
            return self.get_object_by_id(item)
        else:
            raise TypeError(f"Unsupported type {type(item)}")

    def __len__(self):
        return len(self._items)
