from collections import defaultdict

from thing.type import TensorObject


class Store:
    def __init__(self):
        self._items = {}
        self._name_to_id = defaultdict(list)

    def add_item(self, item: TensorObject):
        self._items[item.id] = item
        if item.name:
            self._name_to_id[item.name].append(item.id)

    def get_tensor_by_id(self, idx: int):
        return self._items[idx]

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
        if not self._name_to_id[name]:
            raise KeyError(f"Item with name {name} does not exist.")
        if index >= len(self._name_to_id[name]) or index < 0:
            raise IndexError(f"Index {index} is out of range for name {name}.")
        return self._items[self._name_to_id[name][-(1 + index)]]

    # todo: add support for arbitrary pytree objects
