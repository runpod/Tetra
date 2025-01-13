# # import grpc 
# # from concurrent import futures
# # import remote_execution_pb2
# # import remote_execution_pb2_grpc
# # import json 



# # class RemoteExecutor(remote_execution_pb2_grpc.RemoteExecutorServicer):
# #     def ExecuteFunction(self, request, context):
# #         try:
# #             namespace = {}
# #             #execute the function code
# #             exec(request.function_code, namespace)
# #             #get the funtion by nam 
# #             function = namespace[request.function_name]

# #             #parse arguments
# #             args = [json.loads(arg) for arg in request.args]
# #             kwargs = {k: json.loads(v) for k, v in request.kwargs.items()}

# #             # execute the function
# #             result = function(*args, **kwargs)


# #             return remote_execution_pb2.FunctionResponse(success=True, result=json.dumps(result))
# #         except Exception as e:
# #             return remote_execution_pb2.FunctionResponse(success=False, error=str(e))
        
# # def serve():
# #     server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
# #     remote_execution_pb2_grpc.add_RemoteExecutorServicer_to_server(RemoteExecutor(), server)
# #     server.add_insecure_port("[::]:50051")
# #     server.start()
# #     server.wait_for_termination()



# # if __name__ == "__main__":
# #     serve()




# # server.py
# import grpc
# from concurrent import futures
# import remote_execution_pb2
# import remote_execution_pb2_grpc
# import json

# class RemoteExecutor(remote_execution_pb2_grpc.RemoteExecutorServicer):
#     def ExecuteFunction(self, request, context):
#         try:
#             # Create function namespace
#             namespace = {}
#             global_namespace = {}  # Add this for global functions like print
            
#             # Execute the function code in the namespace
#             exec(request.function_code, global_namespace, namespace)
            
#             # Get the function by name from the local namespace
#             if request.function_name not in namespace:
#                 raise NameError(f"Function '{request.function_name}' not found in the executed code")
            
#             func = namespace[request.function_name]
            
#             # Parse arguments
#             args = [json.loads(arg) for arg in request.args]
#             kwargs = {k: json.loads(v) for k, v in request.kwargs.items()}
            
#             # Execute function
#             result = func(*args, **kwargs)
            
#             return remote_execution_pb2.FunctionResponse(
#                 result=json.dumps(result),
#                 success=True
#             )
#         except Exception as e:
#             import traceback
#             error_msg = f"{str(e)}\n{traceback.format_exc()}"
#             return remote_execution_pb2.FunctionResponse(
#                 success=False,
#                 error=error_msg
#             )

# def serve():
#     server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
#     remote_execution_pb2_grpc.add_RemoteExecutorServicer_to_server(
#         RemoteExecutor(), server
#     )
#     server.add_insecure_port('[::]:50052')
#     print("Server starting on port 50052...")
#     server.start()
#     print("server started on 50052...")
#     server.wait_for_termination()

# if __name__ == '__main__':
#     serve()


# server.py
import grpc.aio
import remote_execution_pb2
import remote_execution_pb2_grpc
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