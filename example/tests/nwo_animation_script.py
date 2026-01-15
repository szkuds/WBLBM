#!/usr/bin/env python3
"""
NWO Presentation Animation Script

Standalone script to create composite animations showing:
- Top: Density field visualization
- Bottom: Contact line capillary number and contact angles plot

Usage:
    python nwo_animation_script.py

The script will prompt for:
    1. Simulation directory path (containing data/it*.npz files)
    2. Whether to create video and video parameters

Data requirements:
    - Simulation directory should contain:
        - data/ folder with .npz files (timestep_*.npz or it*.npz)
        - config.json or constants.txt with simulation parameters
    - If Analysis/processed_data.csv exists, it will be used directly
    - Otherwise, data will be generated from npz files

Output:
    - Animation/frames/frame_00000.png, frame_00001.png, ...
    - Animation/animation.mp4 (optional)
"""

# Import the animation module functions
import sys
import os

# Add the parent directory to path if running standalone
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(script_dir))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    from wblbm.utils.animation import (
        main,
        create_animation_frames,
        create_video,
        process_simulation_data
    )
except ImportError:
    # If wblbm is not importable, the functions are defined inline
    print("Warning: Could not import from wblbm.utils.animation")
    print("Running in standalone mode...")

    # Copy the essential functions here for standalone operation
    exec(open(os.path.join(project_root, 'wblbm', 'utils', 'animation.py')).read())


if __name__ == '__main__':
    main()

