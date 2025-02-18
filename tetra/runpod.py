import runpod
import logging

logging.basicConfig(level=logging.INFO)


class Runpod:
    def __init__(self, api_key: str):
        self.api_key = api_key
        runpod.api_key = api_key

    # This creates an endpoint.
    async def deploy_endpoint(self, config: dict) -> None:
        try:    
            new_endpoint = await runpod.create_endpoint( # type: ignore
                name=config["name"],
                template=config["template"], # use Pre_built template for testing
                gpu_ids=config["gpu_ids"],
                env_vars=config["env_vars"],
            )
            logging.info(f"Endpoint created: {new_endpoint}")
            
        except Exception as e:
            raise e



# What would flow be like.
# when we get the client, with client we should also get