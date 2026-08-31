from mot_counting.utils.geometry import (
    get_bbox_center,
    get_bottom_center,
    get_side,
    signed_distance,
)


def test_get_bbox_center():
    # case 1: test with a bounding box with positive coordinates
    bbox = (10.0, 20.0, 30.0, 60.0)
    expected_center = (20.0, 40.0)
    assert get_bbox_center(bbox) == expected_center

    # case 2: test with a bounding box with negative coordinates
    bbox = (-10.0, -20.0, -30.0, -60.0)
    expected_center = (-20.0, -40.0)
    assert get_bbox_center(bbox) == expected_center

    # case 3: test with a bounding box with mixed coordinates
    bbox = (-10.0, 20.0, 30.0, -60.0)
    expected_center = (10.0, -20.0)
    assert get_bbox_center(bbox) == expected_center

    # case 4: test with a bounding box with floating point coordinates
    bbox = (10.5, 20.2, 30.5, 60.8)
    expected_center = (20.5, 40.5)
    assert get_bbox_center(bbox) == expected_center


def test_get_bottom_center():
    # case 1: test with a bounding box with positive coordinates
    bbox = (10.0, 20.0, 30.0, 60.0)
    expected_bottom_center = (20.0, 60.0)
    assert get_bottom_center(bbox) == expected_bottom_center

    # case 2: test with a bounding box with negative coordinates
    bbox = (-10.0, -20.0, -30.0, -60.0)
    expected_bottom_center = (-20.0, -60.0)
    assert get_bottom_center(bbox) == expected_bottom_center

    # case 3: test with a bounding box with mixed coordinates
    bbox = (-10.0, 20.0, 30.0, -60.0)
    expected_bottom_center = (10.0, -60.0)
    assert get_bottom_center(bbox) == expected_bottom_center

    # case 4: test with a bounding box with floating point coordinates
    bbox = (10.5, 20.2, 30.5, 60.8)
    expected_bottom_center = (20.5, 60.8)
    assert get_bottom_center(bbox) == expected_bottom_center


def test_get_side():
    # case 1: test with a positive distance
    distance_from_line = 20.0
    assert get_side(distance_from_line) == 1

    # case 2: test with a negative distance
    distance_from_line = -20.0
    assert get_side(distance_from_line) == -1

    # case 3: test with a zero distance
    distance_from_line = 0.0
    assert get_side(distance_from_line) == 0


def test_signed_distance():
    # case 1: point above the line
    point_a = (0.0, 0.0)
    point_b = (10.0, 0.0)
    point_p = (5.0, 5.0)
    expected_distance = 50.0
    assert signed_distance(point_a, point_b, point_p) == expected_distance

    # case 2: point below the line
    point_a = (0.0, 0.0)
    point_b = (10.0, 0.0)
    point_p = (5.0, -5.0)
    expected_distance = -50.0
    assert signed_distance(point_a, point_b, point_p) == expected_distance

    # case 3: point on the line
    point_a = (0.0, 0.0)
    point_b = (10.0, 0.0)
    point_p = (5.0, 0.0)
    expected_distance = 0.0
    assert signed_distance(point_a, point_b, point_p) == expected_distance

    # case 4: point on the right side of the line
    point_a = (0.0, 0.0)
    point_b = (0.0, 10.0)
    point_p = (15.0, 5.0)
    expected_distance = -150.0
    assert signed_distance(point_a, point_b, point_p) == expected_distance

    # case 5: point on the left side of the line
    point_a = (0.0, 0.0)
    point_b = (0.0, 10.0)
    point_p = (-5.0, 5.0)
    expected_distance = 50.0
    assert signed_distance(point_a, point_b, point_p) == expected_distance

    # case 6: pint_a and point_b are the same point
    point_a = (0.0, 0.0)
    point_b = (0.0, 0.0)
    point_p = (5.0, 5.0)
    expected_distance = 0.0
    assert signed_distance(point_a, point_b, point_p) == expected_distance


def test_geometry_end_to_end_chain():
    """
    Test the full chain: get_side(signed_distance(...))
    for both horizontal and vertical lines to ensure end-to-end correctness.
    """
    # case 1: Horizontal line: A(0,0) -> B(10,0)
    point_a_horiz = (0.0, 0.0)
    point_b_horiz = (10.0, 0.0)

    # Left side (y > 0)
    assert get_side(signed_distance(point_a_horiz, point_b_horiz, (5.0, 5.0))) == 1
    # Right side (y < 0)
    assert get_side(signed_distance(point_a_horiz, point_b_horiz, (5.0, -5.0))) == -1
    # Exactly on line (y = 0)
    assert get_side(signed_distance(point_a_horiz, point_b_horiz, (5.0, 0.0))) == 0

    # Case 2: Vertical line: A(0,0) -> B(0,10)
    point_a_vert = (0.0, 0.0)
    point_b_vert = (0.0, 10.0)

    # Left side (x < 0)
    assert get_side(signed_distance(point_a_vert, point_b_vert, (-5.0, 5.0))) == 1
    # Right side (x > 0)
    assert get_side(signed_distance(point_a_vert, point_b_vert, (5.0, 5.0))) == -1
    # Exactly on line (x = 0)
    assert get_side(signed_distance(point_a_vert, point_b_vert, (0.0, 5.0))) == 0
