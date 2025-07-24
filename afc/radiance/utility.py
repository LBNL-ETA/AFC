# Advanced Fenestration Controller (AFC) Copyright (c) 2023, The
# Regents of the University of California, through Lawrence Berkeley
# National Laboratory (subject to receipt of any required approvals
# from the U.S. Dept. of Energy). All rights reserved.

"""
Advanced Fenestration Controller
Radiance forecasting module.
"""

# pylint: disable=too-many-arguments, too-many-positional-arguments

import numpy as np

def create_trapezoid_mask(x_coords, y_coords, p1, p2, p3, p4):
    """
    Create an image mask for a trapezoid filter based on given X and Y coordinates.

    Parameters:
    x_coords (array-like): Array of X coordinates.
    y_coords (array-like): Array of Y coordinates.
    p1, p2, p3, p4 (tuples): The vertices of the trapezoid.

    Returns:
    np.ndarray: The image mask.
    """
    # Convert coordinates to numpy arrays
    x_coords = np.array(x_coords)
    y_coords = np.array(y_coords)

    # Create a grid of coordinates
    x, y = np.meshgrid(x_coords, y_coords)

    # Flatten the grid for easier processing
    x_flat = x.flatten()
    y_flat = y.flatten()

    # Check if each point is inside the trapezoid
    def is_point_in_trapezoid(x, y, p1, p2, p3, p4):
        def sign(p1, p2, p3):
            return (p1[0] - p3[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p3[1])

        def point_in_triangle(p1, p2, p3, pt):
            b1 = sign(pt, p1, p2) < 0.0
            b2 = sign(pt, p2, p3) < 0.0
            b3 = sign(pt, p3, p1) < 0.0

            return ((b1 == b2) and (b2 == b3))

        return (point_in_triangle(p1, p2, p3, (x, y)) or
                point_in_triangle(p1, p3, p4, (x, y)))

    # Apply the function to each point
    mask_flat = np.array([is_point_in_trapezoid(x, y, p1, p2, p3, p4) for x, y \
                          in zip(x_flat, y_flat)])

    # Reshape the mask to match the original grid
    mask = mask_flat.reshape(x.shape)

    return mask.astype(int)
