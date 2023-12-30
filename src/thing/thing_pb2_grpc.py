# Generated by the gRPC Python protocol compiler plugin. DO NOT EDIT!
"""Client and server classes corresponding to protobuf-defined services."""
import grpc

from thing import thing_pb2 as thing_dot_thing__pb2


class ThingStub(object):
    """Missing associated documentation comment in .proto file."""

    def __init__(self, channel):
        """Constructor.

        Args:
            channel: A grpc.Channel.
        """
        self.CatchArray = channel.unary_unary(
                '/thing.Thing/CatchArray',
                request_serializer=thing_dot_thing__pb2.Array.SerializeToString,
                response_deserializer=thing_dot_thing__pb2.Response.FromString,
                )
        self.CatchString = channel.unary_unary(
                '/thing.Thing/CatchString',
                request_serializer=thing_dot_thing__pb2.String.SerializeToString,
                response_deserializer=thing_dot_thing__pb2.Response.FromString,
                )
        self.CatchByte = channel.unary_unary(
                '/thing.Thing/CatchByte',
                request_serializer=thing_dot_thing__pb2.Byte.SerializeToString,
                response_deserializer=thing_dot_thing__pb2.Response.FromString,
                )
        self.CatchPyTree = channel.unary_unary(
                '/thing.Thing/CatchPyTree',
                request_serializer=thing_dot_thing__pb2.PyTreeNode.SerializeToString,
                response_deserializer=thing_dot_thing__pb2.Response.FromString,
                )
        self.HealthCheck = channel.unary_unary(
                '/thing.Thing/HealthCheck',
                request_serializer=thing_dot_thing__pb2.HealthCheckRequest.SerializeToString,
                response_deserializer=thing_dot_thing__pb2.Response.FromString,
                )


class ThingServicer(object):
    """Missing associated documentation comment in .proto file."""

    def CatchArray(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def CatchString(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def CatchByte(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def CatchPyTree(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def HealthCheck(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')


def add_ThingServicer_to_server(servicer, server):
    rpc_method_handlers = {
            'CatchArray': grpc.unary_unary_rpc_method_handler(
                    servicer.CatchArray,
                    request_deserializer=thing_dot_thing__pb2.Array.FromString,
                    response_serializer=thing_dot_thing__pb2.Response.SerializeToString,
            ),
            'CatchString': grpc.unary_unary_rpc_method_handler(
                    servicer.CatchString,
                    request_deserializer=thing_dot_thing__pb2.String.FromString,
                    response_serializer=thing_dot_thing__pb2.Response.SerializeToString,
            ),
            'CatchByte': grpc.unary_unary_rpc_method_handler(
                    servicer.CatchByte,
                    request_deserializer=thing_dot_thing__pb2.Byte.FromString,
                    response_serializer=thing_dot_thing__pb2.Response.SerializeToString,
            ),
            'CatchPyTree': grpc.unary_unary_rpc_method_handler(
                    servicer.CatchPyTree,
                    request_deserializer=thing_dot_thing__pb2.PyTreeNode.FromString,
                    response_serializer=thing_dot_thing__pb2.Response.SerializeToString,
            ),
            'HealthCheck': grpc.unary_unary_rpc_method_handler(
                    servicer.HealthCheck,
                    request_deserializer=thing_dot_thing__pb2.HealthCheckRequest.FromString,
                    response_serializer=thing_dot_thing__pb2.Response.SerializeToString,
            ),
    }
    generic_handler = grpc.method_handlers_generic_handler(
            'thing.Thing', rpc_method_handlers)
    server.add_generic_rpc_handlers((generic_handler,))


 # This class is part of an EXPERIMENTAL API.
class Thing(object):
    """Missing associated documentation comment in .proto file."""

    @staticmethod
    def CatchArray(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/thing.Thing/CatchArray',
            thing_dot_thing__pb2.Array.SerializeToString,
            thing_dot_thing__pb2.Response.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def CatchString(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/thing.Thing/CatchString',
            thing_dot_thing__pb2.String.SerializeToString,
            thing_dot_thing__pb2.Response.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def CatchByte(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/thing.Thing/CatchByte',
            thing_dot_thing__pb2.Byte.SerializeToString,
            thing_dot_thing__pb2.Response.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def CatchPyTree(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/thing.Thing/CatchPyTree',
            thing_dot_thing__pb2.PyTreeNode.SerializeToString,
            thing_dot_thing__pb2.Response.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def HealthCheck(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/thing.Thing/HealthCheck',
            thing_dot_thing__pb2.HealthCheckRequest.SerializeToString,
            thing_dot_thing__pb2.Response.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)
