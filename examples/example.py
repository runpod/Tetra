import tetra
from tetra.client import remote
from tetra.client_manager import get_global_client
import asyncio

async def setup_client():
    client = get_global_client()
    await client.add_server("server1", "localhost:50052")
    # await client.add_server("server2", "205.196.17.34:8519")
    client.create_pool("compute_pool", ["server1"])
    return client

# Function that will run on local server
@remote("server1")
def preprocess_data(data: list) -> dict:
    """Initial data processing on local server"""
    return {
        "sum": sum(data),
        "length": len(data),
        "squared": [x**2 for x in data]
    }

# Function that will run on remote server
@remote("server1")
def advanced_analysis(preprocessed: dict) -> dict:
    """Complex analysis on remote server"""
    squared_sum = sum(preprocessed["squared"])
    mean = preprocessed["sum"] / preprocessed["length"]
    
    return {
        "mean": mean,
        "squared_sum": squared_sum,
        "original_sum": preprocessed["sum"],
        "sample_size": preprocessed["length"]
    }

# Function that uses the pool (will run on either server)
@remote("compute_pool")
def final_calculations(analysis: dict) -> dict:
    """Final calculations can run on any available server"""
    return {
        "final_score": analysis["squared_sum"] / analysis["sample_size"],
        "normalized_mean": analysis["mean"] / analysis["sample_size"],
        "metrics_computed": len(analysis)
    }

async def main():
    await setup_client()
    
    try:
        # Initial data
        input_data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        
        print("Starting distributed processing pipeline...")
        
        # Step 1: Preprocess on local server
        print("\nStep 1: Preprocessing on local server")
        preprocessed = await preprocess_data(input_data)
        print(f"Preprocessed results: {preprocessed}")
        
        # Step 2: Advanced analysis on remote server
        print("\nStep 2: Running advanced analysis on remote server")
        analysis_results = await advanced_analysis(preprocessed)
        print(f"Analysis results: {analysis_results}")
        
        # Step 3: Final calculations on any available server
        print("\nStep 3: Performing final calculations")
        final_results = await final_calculations(analysis_results)
        print(f"Final results: {final_results}")
        
        # Bonus: Parallel processing example
        print("\nBonus: Running parallel processing")
        data_chunks = [
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9]
        ]
        
        parallel_results = await asyncio.gather(
            *[preprocess_data(chunk) for chunk in data_chunks]
        )
        print(f"Parallel processing results: {parallel_results}")
        
    except Exception as e:
        print(f"Error occurred: {e}")

if __name__ == "__main__":
    asyncio.run(main())