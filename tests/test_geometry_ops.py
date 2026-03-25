"""geometry_ops 纯函数单测。"""

from app.utils.geometry_ops import (
    find_room_index_by_id,
    next_room_id,
    next_room_name,
    flatten_geometry,
    find_split_points,
)


def test_room_helpers():
    rooms = [
        {"id": "ROOM_001", "name": "A"},
        {"id": "ROOM_007", "name": "C"},
    ]
    assert find_room_index_by_id(rooms, "ROOM_007") == 1
    assert find_room_index_by_id(rooms, "ROOM_999") == -1
    assert next_room_id(rooms) == "ROOM_008"
    assert next_room_name(rooms) == "B"


def test_find_split_points():
    # 矩形 world geometry（闭合）
    geom = [0.0, 0.0, 4.0, 0.0, 4.0, 2.0, 0.0, 2.0, 0.0, 0.0]
    ok, result = find_split_points((2.0, -1.0), (2.0, 3.0), geom)
    assert ok
    poly_a, poly_b, intersections = result
    assert len(intersections) == 2
    assert len(poly_a) >= 3
    assert len(poly_b) >= 3
    flat = flatten_geometry(poly_a)
    assert len(flat) >= 8

