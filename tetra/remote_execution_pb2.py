# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# NO CHECKED-IN PROTOBUF GENCODE
# source: remote_execution.proto
# Protobuf Python Version: 5.29.0
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import runtime_version as _runtime_version
from google.protobuf import symbol_database as _symbol_database
from google.protobuf.internal import builder as _builder
_runtime_version.ValidateProtobufRuntimeVersion(
    _runtime_version.Domain.PUBLIC,
    5,
    29,
    0,
    '',
    'remote_execution.proto'
)
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n\x16remote_execution.proto\x12\x05tetra\"\xc6\x01\n\x0f\x46unctionRequest\x12\x15\n\rfunction_name\x18\x01 \x01(\t\x12\x15\n\rfunction_code\x18\x02 \x01(\t\x12\x0c\n\x04\x61rgs\x18\x03 \x03(\t\x12\x32\n\x06kwargs\x18\x04 \x03(\x0b\x32\".tetra.FunctionRequest.KwargsEntry\x12\x14\n\x0c\x64\x65pendencies\x18\x05 \x03(\t\x1a-\n\x0bKwargsEntry\x12\x0b\n\x03key\x18\x01 \x01(\t\x12\r\n\x05value\x18\x02 \x01(\t:\x02\x38\x01\"B\n\x10\x46unctionResponse\x12\x0f\n\x07success\x18\x01 \x01(\x08\x12\x0e\n\x06result\x18\x02 \x01(\t\x12\r\n\x05\x65rror\x18\x03 \x01(\t2V\n\x0eRemoteExecutor\x12\x44\n\x0f\x45xecuteFunction\x12\x16.tetra.FunctionRequest\x1a\x17.tetra.FunctionResponse\"\x00\x62\x06proto3')

_globals = globals()
_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, _globals)
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, 'remote_execution_pb2', _globals)
if not _descriptor._USE_C_DESCRIPTORS:
  DESCRIPTOR._loaded_options = None
  _globals['_FUNCTIONREQUEST_KWARGSENTRY']._loaded_options = None
  _globals['_FUNCTIONREQUEST_KWARGSENTRY']._serialized_options = b'8\001'
  _globals['_FUNCTIONREQUEST']._serialized_start=34
  _globals['_FUNCTIONREQUEST']._serialized_end=232
  _globals['_FUNCTIONREQUEST_KWARGSENTRY']._serialized_start=187
  _globals['_FUNCTIONREQUEST_KWARGSENTRY']._serialized_end=232
  _globals['_FUNCTIONRESPONSE']._serialized_start=234
  _globals['_FUNCTIONRESPONSE']._serialized_end=300
  _globals['_REMOTEEXECUTOR']._serialized_start=302
  _globals['_REMOTEEXECUTOR']._serialized_end=388
# @@protoc_insertion_point(module_scope)
