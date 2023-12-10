from google.protobuf.internal import containers as _containers
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Iterable as _Iterable, Mapping as _Mapping, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class DTYPE(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = []
    INT8: _ClassVar[DTYPE]
    INT16: _ClassVar[DTYPE]
    INT32: _ClassVar[DTYPE]
    INT64: _ClassVar[DTYPE]
    UINT8: _ClassVar[DTYPE]
    UINT16: _ClassVar[DTYPE]
    UINT32: _ClassVar[DTYPE]
    UINT64: _ClassVar[DTYPE]
    FLOAT16: _ClassVar[DTYPE]
    FLOAT32: _ClassVar[DTYPE]
    FLOAT64: _ClassVar[DTYPE]
    BOOL: _ClassVar[DTYPE]

class FRAMEWORK(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = []
    NUMPY: _ClassVar[FRAMEWORK]
    TORCH: _ClassVar[FRAMEWORK]
    JAX: _ClassVar[FRAMEWORK]

class STATUS(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = []
    SUCCESS: _ClassVar[STATUS]
    FAILURE: _ClassVar[STATUS]

class NODE_TYPE(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = []
    LIST: _ClassVar[NODE_TYPE]
    TUPLE: _ClassVar[NODE_TYPE]
    DICT: _ClassVar[NODE_TYPE]
    TENSOR: _ClassVar[NODE_TYPE]
    STRING: _ClassVar[NODE_TYPE]
INT8: DTYPE
INT16: DTYPE
INT32: DTYPE
INT64: DTYPE
UINT8: DTYPE
UINT16: DTYPE
UINT32: DTYPE
UINT64: DTYPE
FLOAT16: DTYPE
FLOAT32: DTYPE
FLOAT64: DTYPE
BOOL: DTYPE
NUMPY: FRAMEWORK
TORCH: FRAMEWORK
JAX: FRAMEWORK
SUCCESS: STATUS
FAILURE: STATUS
LIST: NODE_TYPE
TUPLE: NODE_TYPE
DICT: NODE_TYPE
TENSOR: NODE_TYPE
STRING: NODE_TYPE

class CatchArrayRequest(_message.Message):
    __slots__ = ["id", "shape", "var_name", "dtype", "framework", "data", "chunk_id", "num_chunks"]
    ID_FIELD_NUMBER: _ClassVar[int]
    SHAPE_FIELD_NUMBER: _ClassVar[int]
    VAR_NAME_FIELD_NUMBER: _ClassVar[int]
    DTYPE_FIELD_NUMBER: _ClassVar[int]
    FRAMEWORK_FIELD_NUMBER: _ClassVar[int]
    DATA_FIELD_NUMBER: _ClassVar[int]
    CHUNK_ID_FIELD_NUMBER: _ClassVar[int]
    NUM_CHUNKS_FIELD_NUMBER: _ClassVar[int]
    id: int
    shape: _containers.RepeatedScalarFieldContainer[int]
    var_name: str
    dtype: DTYPE
    framework: FRAMEWORK
    data: bytes
    chunk_id: int
    num_chunks: int
    def __init__(self, id: _Optional[int] = ..., shape: _Optional[_Iterable[int]] = ..., var_name: _Optional[str] = ..., dtype: _Optional[_Union[DTYPE, str]] = ..., framework: _Optional[_Union[FRAMEWORK, str]] = ..., data: _Optional[bytes] = ..., chunk_id: _Optional[int] = ..., num_chunks: _Optional[int] = ...) -> None: ...

class CatchStringRequest(_message.Message):
    __slots__ = ["id", "var_name", "data"]
    ID_FIELD_NUMBER: _ClassVar[int]
    VAR_NAME_FIELD_NUMBER: _ClassVar[int]
    DATA_FIELD_NUMBER: _ClassVar[int]
    id: int
    var_name: str
    data: str
    def __init__(self, id: _Optional[int] = ..., var_name: _Optional[str] = ..., data: _Optional[str] = ...) -> None: ...

class PyTreeNode(_message.Message):
    __slots__ = ["id", "var_name", "node_type", "children", "key", "object_id"]
    ID_FIELD_NUMBER: _ClassVar[int]
    VAR_NAME_FIELD_NUMBER: _ClassVar[int]
    NODE_TYPE_FIELD_NUMBER: _ClassVar[int]
    CHILDREN_FIELD_NUMBER: _ClassVar[int]
    KEY_FIELD_NUMBER: _ClassVar[int]
    OBJECT_ID_FIELD_NUMBER: _ClassVar[int]
    id: int
    var_name: str
    node_type: NODE_TYPE
    children: _containers.RepeatedCompositeFieldContainer[PyTreeNode]
    key: str
    object_id: int
    def __init__(self, id: _Optional[int] = ..., var_name: _Optional[str] = ..., node_type: _Optional[_Union[NODE_TYPE, str]] = ..., children: _Optional[_Iterable[_Union[PyTreeNode, _Mapping]]] = ..., key: _Optional[str] = ..., object_id: _Optional[int] = ...) -> None: ...

class CatchByteRequest(_message.Message):
    __slots__ = ["data"]
    DATA_FIELD_NUMBER: _ClassVar[int]
    data: bytes
    def __init__(self, data: _Optional[bytes] = ...) -> None: ...

class HealthCheckRequest(_message.Message):
    __slots__ = []
    def __init__(self) -> None: ...

class Response(_message.Message):
    __slots__ = ["status"]
    STATUS_FIELD_NUMBER: _ClassVar[int]
    status: STATUS
    def __init__(self, status: _Optional[_Union[STATUS, str]] = ...) -> None: ...
