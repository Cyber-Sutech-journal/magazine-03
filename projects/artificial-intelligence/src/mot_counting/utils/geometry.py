def signed_distance(
    point_a: tuple[float, float], point_b: tuple[float, float], point_p: tuple[float, float]
) -> float:
    """
    Calculate the signed distance from point P to the line defined by points A and B.

    Args:
        point_a (tuple[float, float]): Coordinates of point A (x1, y1).
        point_b (tuple[float, float]): Coordinates of point B (x2, y2).
        point_p (tuple[float, float]): Coordinates of point P (x0, y0).

        Example:

            1:
            point_a = (0.0, 0.0), point_b = (10.0, 0.0), point_p = (5.0, 5.0)
            after calling this function, it will return 50.0
            its positive number its mean point_p is above the line defined by point_a and point_b

            2:
            point_a = (0.0, 0.0), point_b = (10.0, 0.0), point_p = (5.0, -5.0)
            after calling this function, it will return -50.0
            its negative number its mean point_p is below the line defined by point_a and point_b



    Returns:
        float: The signed distance from point P to the line AB.
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
    Determine which side of a line a point is on.

    if our signed distance is positive number, the result its 1
    it means point is above the line defined by point_a and point_b
    ---
    if our signed distance is negative number, the result its -1
    it means point is below the line defined by point_a and point_b
    ---
    if our signed distance is 0.0, the result its 0
    it means point is on the line defined by point_a and point_b

    Args:
        distance (float): The signed distance from the point to the line.

        For example:
        if we have a signed distance is 50.0 after calling this function, it will return 1
        this means point is above the line defined by point_a and point_b



    Returns:
        int: 1 if the point is on one side of the line, -1 if on the other side, 0 if on the line.
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

    Args:
        bbox (tuple[float, float, float, float]): Bounding box coordinates (x_min, y_min, x_max, y_max).

            For example:
            our bounding box Coordinates is (10.0, 20.0, 30.0, 60.0)
            in result after calling this function, it will return (20.0, 60.0)
            that means bottom center point of our bounding box Coordinates is (20.0, 60.0)

    Returns:
        tuple[float, float]: Coordinates of the bottom center point (x_center, y_bottom).
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
