"""A module containing geometry functions.

Example usage:

    import numpy as np
    from geometry import get_rotation_matrix

    # Define the vector to rotate
    vector = np.array([1.0, 0.0, 0.0])

    # Define the unit vector to rotate to
    unit = np.array([0.0, 1.0, 0.0])

    # Get the rotation matrix
    matrix = get_rotation_matrix(vector, unit)
"""

import numpy as np


def get_rotation_matrix(
    vector: np.ndarray | list | tuple,
    unit: np.ndarray | list | tuple = None,
):
    """Get the rotation matrix to rotate a vector to another vector.

    This uses the Rodrigues' rotation formula for the re-projection.

    Adapted from:
    https://stackoverflow.com/questions/43507491/imprecision-with-rotation-
    matrix-to-align-a-vector-to-an-axis

    Args:
        vector (np.ndarray of float):
            The vector to rotate.
        unit (np.ndarray of float, optional):
            The vector to rotate to.

    Returns:
        matrix (np.ndarray of float):
            The rotation matrix.
    """
    # Define the unit vector is needed
    if unit is None:
        unit = [0.0, 0.0, 1.0]

    # Ensure we have arrays
    if not isinstance(vector, np.ndarray):
        vector = np.array(vector)
    if not isinstance(unit, np.ndarray):
        unit = np.array(unit)

    # Normalize vector length
    vector /= np.linalg.norm(vector)

    # Get axis
    uvw = np.cross(vector, unit)

    # Compute trig values - no need to go through arccos and back
    rcos = np.dot(vector, unit)
    rsin = np.linalg.norm(uvw)

    # Normalize and unpack axis
    if not np.isclose(rsin, 0):
        uvw /= rsin
    u, v, w = uvw

    # Compute rotation matrix
    matrix = (
        rcos * np.eye(3)
        + rsin * np.array([[0, -w, v], [w, 0, -u], [-v, u, 0]])
        + (1.0 - rcos) * uvw[:, None] * uvw[None, :]
    )

    return matrix.astype(np.float64)
