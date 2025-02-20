# import runpod
# import logging
# from tetra.client_manager import get_global_client

# logging.basicConfig(level=logging.INFO)



# async def setup_client():
#     client = get_global_client()
#     await client.add_server("server1", "localhost:50052")
#     # await client.add_server("server2", "205.196.17.34:8519")
#     client.create_pool("compute_pool", ["server1"])
#     return client







# class Runpod:
#     def __init__(self, api_key: str):
#         self.api_key = api_key
#         runpod.api_key = api_key

#     # This creates an endpoint.
#     async def deploy_endpoint(self, config: dict) -> None:
#         try:    
#             new_endpoint = await runpod.create_endpoint( # type: ignore
#                 name=config["name"],
#                 template=config["template"], # use Pre_built template for testing
#                 gpu_ids=config["gpu_ids"],
#                 env_vars=config["env_vars"],
#             )
#             logging.info(f"Endpoint created: {new_endpoint}")
#             return new_endpoint
#         except Exception as e:
#             raise e



# # What would flow be like.
# # when we get the client, with client we should also get.
# # Flow is simple what we do is we first take the input from the user for the API key.
# # then we use those API-key to deploy apps.
# async def create_and_return_endpoints(servers: list):
#     client = get_global_client()
#     rp = Runpod(api_key="1234")
#     servers = ["localhost:50052", "localhost:50053"]
#     for server in servers:
#         await client.add_server("server1", server)
    
#     pool = client.create_pool("compute_pool", servers)

#     return pool



    