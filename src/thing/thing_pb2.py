# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: thing/thing.proto
# Protobuf Python Version: 4.25.0
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import symbol_database as _symbol_database
from google.protobuf.internal import builder as _builder
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n\x11thing/thing.proto\x12\x05thing\"\xda\x01\n\x11\x43\x61tchArrayRequest\x12\n\n\x02id\x18\x01 \x01(\x03\x12\r\n\x05shape\x18\x02 \x03(\x03\x12\x15\n\x08var_name\x18\x03 \x01(\tH\x00\x88\x01\x01\x12\x1b\n\x05\x64type\x18\x04 \x01(\x0e\x32\x0c.thing.DTYPE\x12#\n\tframework\x18\x05 \x01(\x0e\x32\x10.thing.FRAMEWORK\x12\x0c\n\x04\x64\x61ta\x18\x06 \x01(\x0c\x12\x15\n\x08\x63hunk_id\x18\x07 \x01(\rH\x01\x88\x01\x01\x12\x12\n\nnum_chunks\x18\x08 \x01(\rB\x0b\n\t_var_nameB\x0b\n\t_chunk_id\"R\n\x12\x43\x61tchStringRequest\x12\n\n\x02id\x18\x01 \x01(\x03\x12\x15\n\x08var_name\x18\x02 \x01(\tH\x00\x88\x01\x01\x12\x0c\n\x04\x64\x61ta\x18\x03 \x01(\tB\x0b\n\t_var_name\"\xd9\x01\n\nPyTreeNode\x12\n\n\x02id\x18\x01 \x01(\x03\x12\x15\n\x08var_name\x18\x02 \x01(\tH\x00\x88\x01\x01\x12(\n\tnode_type\x18\x03 \x01(\x0e\x32\x10.thing.NODE_TYPEH\x01\x88\x01\x01\x12#\n\x08\x63hildren\x18\x04 \x03(\x0b\x32\x11.thing.PyTreeNode\x12\x10\n\x03key\x18\x05 \x01(\tH\x02\x88\x01\x01\x12\x16\n\tobject_id\x18\x06 \x01(\x03H\x03\x88\x01\x01\x42\x0b\n\t_var_nameB\x0c\n\n_node_typeB\x06\n\x04_keyB\x0c\n\n_object_id\" \n\x10\x43\x61tchByteRequest\x12\x0c\n\x04\x64\x61ta\x18\x01 \x01(\x0c\"\x14\n\x12HealthCheckRequest\")\n\x08Response\x12\x1d\n\x06status\x18\x01 \x01(\x0e\x32\r.thing.STATUS*\x92\x01\n\x05\x44TYPE\x12\x08\n\x04INT8\x10\x00\x12\t\n\x05INT16\x10\x01\x12\t\n\x05INT32\x10\x02\x12\t\n\x05INT64\x10\x03\x12\t\n\x05UINT8\x10\x04\x12\n\n\x06UINT16\x10\x05\x12\n\n\x06UINT32\x10\x06\x12\n\n\x06UINT64\x10\x07\x12\x0b\n\x07\x46LOAT16\x10\x08\x12\x0b\n\x07\x46LOAT32\x10\t\x12\x0b\n\x07\x46LOAT64\x10\n\x12\x08\n\x04\x42OOL\x10\x0b**\n\tFRAMEWORK\x12\t\n\x05NUMPY\x10\x00\x12\t\n\x05TORCH\x10\x01\x12\x07\n\x03JAX\x10\x02*\"\n\x06STATUS\x12\x0b\n\x07SUCCESS\x10\x00\x12\x0b\n\x07\x46\x41ILURE\x10\x01*L\n\tNODE_TYPE\x12\x08\n\x04LIST\x10\x00\x12\t\n\x05TUPLE\x10\x01\x12\x08\n\x04\x44ICT\x10\x02\x12\n\n\x06TENSOR\x10\x03\x12\n\n\x06STRING\x10\x04\x12\x08\n\x04NONE\x10\x05\x32\xa0\x02\n\x05Thing\x12\x37\n\nCatchArray\x12\x18.thing.CatchArrayRequest\x1a\x0f.thing.Response\x12\x39\n\x0b\x43\x61tchString\x12\x19.thing.CatchStringRequest\x1a\x0f.thing.Response\x12\x35\n\tCatchByte\x12\x17.thing.CatchByteRequest\x1a\x0f.thing.Response\x12\x31\n\x0b\x43\x61tchPyTree\x12\x11.thing.PyTreeNode\x1a\x0f.thing.Response\x12\x39\n\x0bHealthCheck\x12\x19.thing.HealthCheckRequest\x1a\x0f.thing.Responseb\x06proto3')

_globals = globals()
_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, _globals)
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, 'thing.thing_pb2', _globals)
if _descriptor._USE_C_DESCRIPTORS == False:
  DESCRIPTOR._options = None
  _globals['_DTYPE']._serialized_start=653
  _globals['_DTYPE']._serialized_end=799
  _globals['_FRAMEWORK']._serialized_start=801
  _globals['_FRAMEWORK']._serialized_end=843
  _globals['_STATUS']._serialized_start=845
  _globals['_STATUS']._serialized_end=879
  _globals['_NODE_TYPE']._serialized_start=881
  _globals['_NODE_TYPE']._serialized_end=957
  _globals['_CATCHARRAYREQUEST']._serialized_start=29
  _globals['_CATCHARRAYREQUEST']._serialized_end=247
  _globals['_CATCHSTRINGREQUEST']._serialized_start=249
  _globals['_CATCHSTRINGREQUEST']._serialized_end=331
  _globals['_PYTREENODE']._serialized_start=334
  _globals['_PYTREENODE']._serialized_end=551
  _globals['_CATCHBYTEREQUEST']._serialized_start=553
  _globals['_CATCHBYTEREQUEST']._serialized_end=585
  _globals['_HEALTHCHECKREQUEST']._serialized_start=587
  _globals['_HEALTHCHECKREQUEST']._serialized_end=607
  _globals['_RESPONSE']._serialized_start=609
  _globals['_RESPONSE']._serialized_end=650
  _globals['_THING']._serialized_start=960
  _globals['_THING']._serialized_end=1248
# @@protoc_insertion_point(module_scope)
