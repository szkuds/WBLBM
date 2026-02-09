import jax.numpy as jnp
from jax import jit
from wblbm.lattice.lattice import Lattice
from functools import partial
from wblbm.utils.timing import time_function, TIMING_ENABLED


class Streaming:
    """
    Callable class to perform the streaming step of the LBM.
    Supports configurable periodicity per axis.
    """

    def __init__(self, lattice: Lattice, bc_config: dict = None):
        self.c = lattice.c
        self.q = lattice.q

        if bc_config is None:
            self.periodic_x = True
            self.periodic_y = True
        else:
            # Only check actual edge keys, ignore params
            left_bc = bc_config.get('left', 'periodic')
            right_bc = bc_config.get('right', 'periodic')
            top_bc = bc_config.get('top', 'periodic')
            bottom_bc = bc_config.get('bottom', 'periodic')

            self.periodic_x = (left_bc == 'periodic' and right_bc == 'periodic')
            self.periodic_y = (top_bc == 'periodic' and bottom_bc == 'periodic')

    @time_function(enable_timing=TIMING_ENABLED)
    @partial(jit, static_argnums=(0,))
    def __call__(self, f):
        """
        Perform the streaming step of the LBM.
        Non-periodic axes use shift with zero-fill at boundaries.
        """
        for i in range(self.q):
            shift_x = int(self.c[0, i])
            shift_y = int(self.c[1, i])

            fi = f[:, :, i, 0]

            # X direction streaming
            if shift_x != 0:
                fi = jnp.roll(fi, shift_x, axis=0)
                if not self.periodic_x:
                    if shift_x > 0:
                        fi = fi.at[:shift_x, :].set(0.0)
                    else:
                        fi = fi.at[shift_x:, :].set(0.0)

            # Y direction streaming
            if shift_y != 0:
                fi = jnp.roll(fi, shift_y, axis=1)
                if not self.periodic_y:
                    if shift_y > 0:
                        fi = fi.at[:, :shift_y].set(0.0)
                    else:
                        fi = fi.at[:, shift_y:].set(0.0)

            f = f.at[:, :, i, 0].set(fi)

        return f
