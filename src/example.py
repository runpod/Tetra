# # example.py
# from client import remote
# from client_manager import get_global_client

# # Setup servers
# client = get_global_client()
# client.add_server("local", "localhost:50051")

# @remote("local")
# def add_numbers(a: int, b: int) -> int:
#     return a + b


# @remote("local")
# def subtract_numbers(a: int, b: int) -> int:
#     return a - b
# async def main():
#     try:
#         result = await add_numbers(10, 20)
#         result2 = await subtract_numbers(10, 20)

#         print(f"Result: {result}")
#         print(f"Result2: {result2}")
#     except Exception as e:
#         print(f"Error occurred: {e}")

# if __name__ == "__main__":
#     import asyncio
#     asyncio.run(main())



# example.py
from client import remote
from client_manager import get_global_client
import asyncio

async def setup_client():
    client = get_global_client()
    await client.add_server("server1", "localhost:50051")
    await client.add_server("server2", "localhost:50052")
    client.create_pool("compute_pool", ["server1", "server2"])
    return client

@remote("compute_pool")
def add_numbers(a: int, b: int) -> int:
    return a + b

@remote("server1", fallback="server2")
def multiply_numbers(a: int, b: int) -> int:
    return a * b

async def main():
    await setup_client()
    
    try:
        result1 = await add_numbers(5, 3)
        print(f"Pool execution result: {result1}")
        
        result2 = await multiply_numbers(4, 6)
        print(f"Fallback execution result: {result2}")
        
    except Exception as e:
        print(f"Error occurred: {e}")

if __name__ == "__main__":
    asyncio.run(main())