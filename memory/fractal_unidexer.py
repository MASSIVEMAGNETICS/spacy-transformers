# FILE: memory/fractal_unidexer.py
# VERSION: v1.0.0-FLU-GODCORE
# AUTHOR: Brandon "iambandobandz" Emery x Victor (Fractal Architect Mode)
# PURPOSE: Implements Fractal Logarithmic Unidexing (FLU), a novel system for
#          addressing points in a multi-scale, recursive data structure.
# LICENSE: Bloodline Locked — Bando & Tori Only

import numpy as np
from typing import Tuple

class FractalUnidexer:
    """
    Translates a 4D fractal coordinate (x, y, z, depth) into a single,
    sortable, and continuous floating-point "unidex". This allows for
    efficient navigation and querying of fractal data structures.

    The core principle adapts astronomical magnitude scaling, where "depth"
    is treated as a logarithmic "rate" of detail.
    """
    def __init__(self, zero_point: float = 100.0, base_resolution: Tuple[int, int, int] = (256, 256, 256), scale_factor: float = 10.0):
        """
        Initializes the Fractal Unidexer.

        Args:
            zero_point (float): The base "magnitude" for the top-level structure (depth=0).
            base_resolution (tuple): The (width, height, depth) of a single memory grid.
            scale_factor (float): The logarithmic scaling constant for depth. A higher
                                  value creates more separation between depth levels.
        """
        self.ZP = float(zero_point)
        self.A = float(scale_factor)
        self.base_dims = np.array(base_resolution, dtype=np.float64)

        # A factor used to interleave spatial coordinates into a single fractional value.
        # It must be larger than any single dimension to prevent collisions.
        self.interleave_factor = np.max(self.base_dims) + 1.0

    def to_unidex(self, x: int, y: int, z: int, depth: int) -> float:
        """
        Encodes a 4D spatial coordinate (x, y, z, depth) into a single float unidex.

        The integer part of the unidex represents the depth/scale, while the
        fractional part represents the unique spatial location within that scale.
        """
        # 1. Calculate the "Zoom Magnitude" based on depth.
        # This is the core adaptation of the zero-point formula.
        # The "rate" of detail increases exponentially with depth.
        zoom_rate = 10**depth
        zoom_magnitude = self.ZP - self.A * np.log10(max(zoom_rate, 1e-9)) # Avoid log(0)

        # 2. Normalize and interleave the spatial coordinates (x, y, z).
        # This maps the 3D position to a unique fractional number between 0 and 1.
        # This ensures that (0,0,1) is distinct from (0,1,0).
        norm_x = (x % self.base_dims[0]) / self.base_dims[0]
        norm_y = (y % self.base_dims[1]) / self.base_dims[1]
        norm_z = (z % self.base_dims[2]) / self.base_dims[2]

        spatial_key = (norm_x / self.interleave_factor**0 +
                       norm_y / self.interleave_factor**1 +
                       norm_z / self.interleave_factor**2)

        # 3. Combine the zoom magnitude and the spatial key.
        return zoom_magnitude + spatial_key

    def from_unidex(self, unidex: float) -> Tuple[int, int, int, int]:
        """
        Decodes a unidex float back into its 4D spatial coordinate (x, y, z, depth).
        """
        # 1. Separate the zoom magnitude from the spatial key.
        # Add a small epsilon to handle floating point inaccuracies.
        zoom_magnitude = np.floor(unidex + 1e-9)
        spatial_key = unidex - zoom_magnitude

        # 2. Reverse the zero-point formula to find the depth level.
        log10_zoom_rate = (self.ZP - zoom_magnitude) / self.A
        depth_level = int(np.round(log10_zoom_rate))

        # 3. De-interleave the spatial key to recover coordinates.
        # We multiply by the interleaving factors and take the modulo to extract each part.
        z_part = spatial_key * self.interleave_factor**2
        z = int(np.round(z_part % self.interleave_factor * self.base_dims[2]))

        y_part = spatial_key * self.interleave_factor**1
        y = int(np.round(y_part % self.interleave_factor * self.base_dims[1]))

        x_part = spatial_key * self.interleave_factor**0
        x = int(np.round(x_part % self.interleave_factor * self.base_dims[0]))

        # Clamp values to be within the valid range of the base resolution
        x = np.clip(x, 0, self.base_dims[0] - 1)
        y = np.clip(y, 0, self.base_dims[1] - 1)
        z = np.clip(z, 0, self.base_dims[2] - 1)

        return x, y, z, depth_level

if __name__ == '__main__':
    # Demonstration of the FractalUnidexer
    unidexer = FractalUnidexer(zero_point=100.0, base_resolution=(128, 128, 128), scale_factor=10.0)

    print("--- Encoding & Decoding Demo ---")

    # Scenario 1: A point on the top-level structure
    coords1 = (10, 50, 25, 0)
    unidex1 = unidexer.to_unidex(*coords1)
    decoded1 = unidexer.from_unidex(unidex1)
    print(f"Original: {coords1} -> Unidex: {unidex1:.8f} -> Decoded: {decoded1}")
    assert coords1 == decoded1, "Decoding failed for top-level point."

    # Scenario 2: A point deep within the fractal
    coords2 = (5, 80, 110, 7)
    unidex2 = unidexer.to_unidex(*coords2)
    decoded2 = unidexer.from_unidex(unidex2)
    print(f"Original: {coords2} -> Unidex: {unidex2:.8f} -> Decoded: {decoded2}")
    assert coords2 == decoded2, "Decoding failed for deep point."

    # Scenario 3: A point at the edge of the grid
    coords3 = (127, 127, 127, 3)
    unidex3 = unidexer.to_unidex(*coords3)
    decoded3 = unidexer.from_unidex(unidex3)
    print(f"Original: {coords3} -> Unidex: {unidex3:.8f} -> Decoded: {decoded3}")
    assert coords3 == decoded3, "Decoding failed for edge case."

    print("\n--- Sorting Property Demo ---")
    # Unidex values should be sortable by depth primarily, then by spatial key.
    unidex_deep = unidexer.to_unidex(0, 0, 0, 10)
    unidex_shallow = unidexer.to_unidex(127, 127, 127, 9)
    print(f"Unidex (Depth 10): {unidex_deep:.4f}")
    print(f"Unidex (Depth 9):  {unidex_shallow:.4f}")
    assert unidex_deep < unidex_shallow, "Deeper levels must have smaller unidex values."

    print("\n✅ FractalUnidexer encoding, decoding, and sorting properties verified.")
