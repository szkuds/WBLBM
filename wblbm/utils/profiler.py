import jax
from jax import profiler
import os

# Have not managed to get this to work, removed the profiling from the run class.

class JAXProfiler:
    def __init__(self, output_dir="./profiler_output", create_perfetto_link=True):
        self.output_dir = output_dir
        self.create_perfetto_link = create_perfetto_link
        os.makedirs(output_dir, exist_ok=True)

    def __enter__(self):
        # Configure profiler options for better traces
        options = jax.profiler.ProfileOptions()
        options.host_tracer_level = 2  # Include high-level program execution details
        options.python_tracer_level = 1  # Enable Python tracing

        profiler.start_trace(self.output_dir, profiler_options=options)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        profiler.stop_trace()
        if self.create_perfetto_link:
            print(f"Profiling completed! Trace saved to: {self.output_dir}")
            print("To view the trace, you can:")
            print("1. Use TensorBoard: tensorboard --logdir=" + self.output_dir)
            print("2. Or upload the .pb files to https://ui.perfetto.dev/")
