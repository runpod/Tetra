# server.py
import grpc.aio
from tetra import remote_execution_pb2
from tetra import remote_execution_pb2_grpc
import json
import asyncio

class RemoteExecutor(remote_execution_pb2_grpc.RemoteExecutorServicer):
    async def ExecuteFunction(self, request, context):
        try:
            namespace = {}
            exec(request.function_code, namespace)
            func = namespace[request.function_name]
            
            args = [json.loads(arg) for arg in request.args]
            kwargs = {k: json.loads(v) for k, v in request.kwargs.items()}
            
            result = func(*args, **kwargs)
            
            return remote_execution_pb2.FunctionResponse(
                result=json.dumps(result),
                success=True
            )
        except Exception as e:
            return remote_execution_pb2.FunctionResponse(
                success=False,
                error=str(e)
            )

async def serve():
    server = grpc.aio.server()
    remote_execution_pb2_grpc.add_RemoteExecutorServicer_to_server(
        RemoteExecutor(), server
    )
    server.add_insecure_port('[::]:50052')
    print(f"Starting the server on 50052...")
    await server.start()
    print(f"Server started on 50052")
    await server.wait_for_termination()

if __name__ == '__main__':
    asyncio.run(serve())