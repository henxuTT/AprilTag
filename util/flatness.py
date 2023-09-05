import numpy as np

def are_points_coplanar(p1, p2, p3, p4, threshold=1e-6):
    """
    Check if four points are coplanar within a given threshold.

    Parameters:
    - p1, p2, p3, p4: 3x1 numpy arrays representing the points.
    - threshold: The maximum allowed distance of the fourth point from the plane formed by the first three points.

    Returns:
    - True if the points are coplanar within the threshold, False otherwise.
    """
    # Compute the normal vector of the plane defined by p1, p2, and p3
    v1 = p2 - p1
    v2 = p3 - p1
    normal = np.cross(v1, v2)

    # Compute the distance of p4 from the plane
    distance = np.abs(np.dot(normal, p4 - p1)) / np.linalg.norm(normal)
    print(distance)

    return distance <= threshold, distance