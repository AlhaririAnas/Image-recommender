import time
from functools import wraps

# Global dictionary to store total execution time and call count per function
timing_data = {}

def timeit(func):
    """A decorator to measure and print the average execution time of a function."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time

        # Add or update function data in the timing dictionary
        if func.__name__ not in timing_data:
            timing_data[func.__name__] = {"total_time": 0, "calls": 0}

        timing_data[func.__name__]["total_time"] += execution_time
        timing_data[func.__name__]["calls"] += 1

        average_time = timing_data[func.__name__]["total_time"] / timing_data[func.__name__]["calls"]
        print(f"Average execution time after {timing_data[func.__name__]['calls']} calls of the function {func.__name__}: {average_time:.4f} seconds.")
        
        return result
    return wrapper
