import numpy as np
import math


def VectorProjection(a, b):
    """
    Calculates the projection of vector a onto vector b.
    """

    return (np.dot(a, b) / np.dot(b, b)) * b


def SignedAngle(v1, v2, normal):
    """Calculates the signed angle between two vectors.

    Args:
        v1: The first vector.
        v2: The second vector.
        normal: The normal vector defining the positive direction.
                e.g., [0, 0, 1] (z-axis).

    Returns:
        The signed angle in degrees.
    """

    cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    angle = np.arccos(np.clip(cos_angle, -1, 1))  # in radians

    # Determine sign based on cross product
    cross_product = np.cross(v1, v2)
    sign = np.sign(np.dot(cross_product, normal))

    return np.degrees(angle * sign)


def Magnitude(vector):
    return math.sqrt(sum(x**2 for x in vector))
