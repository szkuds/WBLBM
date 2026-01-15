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
        """
        Args:
            lattice: Lattice object
            bc_config: Boundary condition config dict. If provided, automatically
                       determines periodicity from 'left'/'right' and 'top'/'bottom'.
                       If None, defaults to fully periodic.
        """
        self.c = lattice.c  # Shape: (2, Q)
        self.q = lattice.q

        # Determine periodicity from bc_config
        if bc_config is None:
            self.periodic_x = True
            self.periodic_y = True
        else:
            # X is periodic if both left AND right are 'periodic'
            self.periodic_x = (
                    bc_config.get('left') == 'periodic' and
                    bc_config.get('right') == 'periodic'
            )
            # Y is periodic if both top AND bottom are 'periodic'
            self.periodic_y = (
                    bc_config.get('top') == 'periodic' and
                    bc_config.get('bottom') == 'periodic'
            )

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
