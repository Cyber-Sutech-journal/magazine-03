"""
Geometry utilities for bounding box manipulation and crossing logic.

Coordinate System & Sign Convention:
This module uses a standard 2D coordinate system. For distance and side calculations,
a positive cross-product value corresponds to the left side when walking along the
directed line from point A to point B (following the right-hand rule).
"""


def signed_distance(
    point_a: tuple[float, float], point_b: tuple[float, float], point_p: tuple[float, float]
) -> float:
    """
    Calculate the signed cross-product value / oriented side magnitude from point P to the line defined by points A and B.

    If point_a and point_b are exactly the same (a degenerate line), this function returns exactly 0.0 and never raises an exception.

    Args:
        point_a (tuple[float, float]): Coordinates of point A (x1, y1).
        point_b (tuple[float, float]): Coordinates of point B (x2, y2).
        point_p (tuple[float, float]): Coordinates of point P (x0, y0).

        Example:
            1:
            point_a = (0.0, 0.0), point_b = (10.0, 0.0), point_p = (5.0, 5.0)
            after calling this function, it will return 50.0
            its positive number means point_p is above the line defined by point_a and point_b.

            2:
            point_a = (0.0, 0.0), point_b = (10.0, 0.0), point_p = (5.0, -5.0)
            after calling this function, it will return -50.0
            its negative number means point_p is below the line defined by point_a and point_b.

    Returns:
        float: The signed cross-product value.
               Positive if P is on one side of the line, negative if on the other side.
    """
    x1, y1 = point_a
    x2, y2 = point_b
    x0, y0 = point_p

    # Calculate the signed distance using the formula
    distance = ((x2 - x1) * (y0 - y1)) - ((y2 - y1) * (x0 - x1))

    return distance


def get_side(distance: float) -> int:
    """
    Determine which side of a line a point is on based on the raw distance.

    - Positive distance -> Returns 1 (left side of the directed line).
    - Negative distance -> Returns -1 (right side of the directed line).
    - Zero distance (0.0) -> Returns 0. This means the point lies exactly
      on the line. Crossing Logic must treat this as "no decisive side"
      (do not emit a crossing or flip confirmed_side).

    Args:
        distance (float): The raw cross-product from `signed_distance`.

    Returns:
        int: 1 (left), -1 (right), or 0 (exactly on the line).
    """
    if distance > 0:
        return 1
    elif distance < 0:
        return -1
    else:
        return 0


def get_bottom_center(bbox: tuple[float, float, float, float]) -> tuple[float, float]:
    """
    Get the bottom center point of a bounding box.

    This specific point represents the ground-contact point of the object,
    which is why it is chosen as the default reference for crossing evaluation.

    Args:
        bbox (tuple[float, float, float, float]): Bounding box coordinates (x_min, y_min, x_max, y_max).

    Returns:
        tuple[float, float]: Coordinates of the bottom center point.
    """
    x_min, y_min, x_max, y_max = bbox
    x_center = (x_min + x_max) / 2
    y_bottom = y_max
    return (x_center, y_bottom)


def get_bbox_center(bbox: tuple[float, float, float, float]) -> tuple[float, float]:
    """
    Get the center point of a bounding box.

    Args:
        bbox (tuple[float, float, float, float]): Bounding box coordinates (x_min, y_min, x_max, y_max).

            For example:
            our bounding box Coordinates is (10.0, 20.0, 30.0, 60.0)
            in result after calling this function, it will return (20.0, 40.0)
            that means center point of our bounding box Coordinates is (20.0, 40.0)

    Returns:
        tuple[float, float]: Coordinates of the center point (x_center, y_center).
    """
    x_min, y_min, x_max, y_max = bbox
    x_center = (x_min + x_max) / 2
    y_center = (y_min + y_max) / 2
    return (x_center, y_center)
