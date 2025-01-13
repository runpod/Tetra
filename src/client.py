import json
from typing import Optional
import remote_execution_pb2
import remote_execution_pb2_grpc
from typing import Union, List
import random
import grpc.aio  # Direct import of grpc.aio
from functools import wraps
import inspect
import json


def get_function_source(func):
    """Extract the function source code without the decorator"""
    source = inspect.getsource(func)
    
    # Get the function lines
    lines = source.split('\n')
    
    # Skip decorator line(s)
    while lines and (lines[0].strip().startswith('@') or not lines[0].strip()):
        lines = lines[1:]
    
    # Rebuild function source
    return '\n'.join(lines)



class RemoteExecutionClient:
    def __init__(self):
        self.servers = {}
        self.stubs = {}
        self.pools = {}
    
    async def add_server(self, name: str, address: str):
        """Register a new server"""
        self.servers[name] = address
        channel = grpc.aio.insecure_channel(address)
        self.stubs[name] = remote_execution_pb2_grpc.RemoteExecutorStub(channel)
    
    def create_pool(self, pool_name: str, server_names: List[str]):
        if not all(name in self.servers for name in server_names):
            raise ValueError("All servers must be registered first")
        self.pools[pool_name] = server_names
    
    def get_stub(self, server_spec: Union[str, List[str]], fallback: Union[None, str, List[str]] = None):
        if isinstance(server_spec, list):
            return self._get_pool_stub(server_spec)
        elif server_spec in self.pools:
            return self._get_pool_stub(self.pools[server_spec])
        elif server_spec in self.stubs:
            stub = self.stubs[server_spec]
            if fallback:
                return StubWithFallback(stub, self, fallback)
            return stub
        else:
            raise ValueError(f"Unknown server or pool: {server_spec}")
    
    def _get_pool_stub(self, server_names: List[str]):
        if not server_names:
            raise ValueError("Server pool is empty")
        server_name = random.choice(server_names)
        return self.stubs[server_name]

class StubWithFallback:
    def __init__(self, primary_stub, client, fallback_spec):
        self.primary_stub = primary_stub
        self.client = client
        self.fallback_spec = fallback_spec
    
    async def ExecuteFunction(self, request):
        try:
            return await self.primary_stub.ExecuteFunction(request)
        except Exception as e:
            print(f"Primary server failed: {e}, trying fallback...")
            fallback_stub = self.client.get_stub(self.fallback_spec)
            return await fallback_stub.ExecuteFunction(request)

def remote(server_spec: Union[str, List[str]], fallback: Union[None, str, List[str]] = None):
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            from client_manager import get_global_client
            global_client = get_global_client()
            
            source = get_function_source(func)
            
            request = remote_execution_pb2.FunctionRequest(
                function_name=func.__name__,
                function_code=source,
                args=[json.dumps(arg) for arg in args],
                kwargs={k: json.dumps(v) for k, v in kwargs.items()}
            )
            
            stub = global_client.get_stub(server_spec, fallback)
            
            try:
                response = await stub.ExecuteFunction(request)
                if response.success:
                    return json.loads(response.result)
                else:
                    raise Exception(f"Remote execution failed: {response.error}")
            except Exception as e:
                raise Exception(f"All execution attempts failed: {str(e)}")
                
        return wrapper
    return decorator