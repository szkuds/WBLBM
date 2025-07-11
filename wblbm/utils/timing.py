import time
from functools import wraps
import jax.numpy as jnp

TIMING_ENABLED = False  # Default: no timing print; set to True to enable


def time_function(enable_timing=TIMING_ENABLED):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if not enable_timing:
                return func(*args, **kwargs)
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

    return decorator
