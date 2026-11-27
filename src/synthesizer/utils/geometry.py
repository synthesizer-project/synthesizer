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
    # Define the unit vector if needed
    if unit is None:
        unit = [0.0, 0.0, 1.0]

    # Convert to float64 arrays without mutating caller inputs
    vec = np.asarray(vector, dtype=np.float64)
    target = np.asarray(unit, dtype=np.float64)

    # Normalize both vectors
    vec = vec / np.linalg.norm(vec)
    target = target / np.linalg.norm(target)

    # Get axis
    uvw = np.cross(vec, target)

    # Compute trig values - no need to go through arccos and back
    rcos = np.dot(vec, target)
    rsin = np.linalg.norm(uvw)

    # Handle parallel and anti-parallel cases explicitly
    if np.isclose(rsin, 0.0):
        if rcos >= 0.0:
            return np.eye(3, dtype=np.float64)

        # Anti-parallel: construct a proper 180-degree rotation (det = +1)
        # around a unit axis orthogonal to vec.
        ref = (
            np.array([1.0, 0.0, 0.0])
            if abs(vec[0]) < 0.9
            else np.array([0.0, 1.0, 0.0])
        )
        axis = np.cross(vec, ref)
        axis /= np.linalg.norm(axis)
        return (2.0 * np.outer(axis, axis) - np.eye(3)).astype(np.float64)

    # Normalize and unpack axis
    uvw = uvw / rsin
    u, v, w = uvw

    # Compute rotation matrix
    matrix = (
        rcos * np.eye(3)
        + rsin * np.array([[0, -w, v], [w, 0, -u], [-v, u, 0]])
        + (1.0 - rcos) * uvw[:, None] * uvw[None, :]
    )

    return matrix.astype(np.float64)
