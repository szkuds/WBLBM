import time
from functools import wraps
import jax.numpy as jnp

def time_function(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        # Block until computation completes (important for JAX)
        if hasattr(result, 'block_until_ready'):
            result.block_until_ready()
        elif isinstance(result, tuple):
            # Handle multiple return values
            for item in result:
                if hasattr(item, 'block_until_ready'):
                    item.block_until_ready()
        end = time.perf_counter()
        print(f"{func.__name__}: {end - start:.4f} seconds")
        return result
    return wrapper
