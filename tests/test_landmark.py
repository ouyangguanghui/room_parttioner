"""LandmarkManager 单测。"""

import math
import pytest

from app.utils.landmark import LandmarkPoint, LandmarkManager


# ==================== LandmarkPoint ====================

class TestLandmarkPoint:
    def test_distance_to(self):
        p1 = LandmarkPoint(0, 0)
        p2 = LandmarkPoint(3, 4)
        assert p1.distance_to(p2) == pytest.approx(5.0)

    def test_distance_to_self(self):
        p = LandmarkPoint(1, 1)
        assert p.distance_to(p) == 0.0

    def test_distance_to_line_perpendicular(self):
        """点到线段的垂直距离"""
        p = LandmarkPoint(1, 1)
        l1 = LandmarkPoint(0, 0)
        l2 = LandmarkPoint(2, 0)
        assert p.distance_to_line(l1, l2) == pytest.approx(1.0)

    def test_distance_to_line_endpoint(self):
        """点到线段端点的距离（投影在线段外）"""
        p = LandmarkPoint(5, 0)
        l1 = LandmarkPoint(0, 0)
        l2 = LandmarkPoint(2, 0)
        assert p.distance_to_line(l1, l2) == pytest.approx(3.0)

    def test_distance_to_line_degenerate(self):
        """线段退化为点"""
        p = LandmarkPoint(3, 4)
        l1 = LandmarkPoint(0, 0)
        assert p.distance_to_line(l1, l1) == pytest.approx(5.0)

    def test_is_inside_polygon_square(self):
        """正方形内部"""
        poly = [
            LandmarkPoint(0, 0), LandmarkPoint(10, 0),
            LandmarkPoint(10, 10), LandmarkPoint(0, 10),
        ]
        assert LandmarkPoint(5, 5).is_inside_polygon(poly)
        assert not LandmarkPoint(15, 5).is_inside_polygon(poly)

    def test_is_inside_polygon_boundary(self):
        """边界上的点（射线法行为与实现相关，不做严格断言）"""
        poly = [
            LandmarkPoint(0, 0), LandmarkPoint(10, 0),
            LandmarkPoint(10, 10), LandmarkPoint(0, 10),
        ]
        # 边界点不保证结果，只要不崩溃即可
        LandmarkPoint(0, 5).is_inside_polygon(poly)

    def test_distance_to_polygon(self):
        """点到多边形边界的最小距离"""
        poly = [
            LandmarkPoint(0, 0), LandmarkPoint(10, 0),
            LandmarkPoint(10, 10), LandmarkPoint(0, 10),
        ]
        # 中心到最近边距离 = 5
        assert LandmarkPoint(5, 5).distance_to_polygon(poly) == pytest.approx(5.0)
        # 距离底边 2
        assert LandmarkPoint(5, 2).distance_to_polygon(poly) == pytest.approx(2.0)


# ==================== LandmarkManager ====================

class TestLandmarkManager:
    def test_generate_single_room(self):
        """单个房间应生成一个 landmark"""
        mgr = LandmarkManager()
        geom = [0, 0, 10, 0, 10, 10, 0, 10, 0, 0]
        result = mgr.generate_landmarks([geom], ["A"], ["ROOM_001"])
        assert len(result) == 1
        assert result[0]["id"] == "PLATFORM_LANDMARK_001"
        assert result[0]["roomId"] == "ROOM_001"
        assert result[0]["name"] == "A"
        assert result[0]["type"] == "pose"
        assert len(result[0]["geometry"]) == 3

    def test_generate_multiple_rooms(self):
        """多个房间应各生成一个 landmark"""
        mgr = LandmarkManager()
        g1 = [0, 0, 10, 0, 10, 10, 0, 10, 0, 0]
        g2 = [20, 0, 30, 0, 30, 10, 20, 10, 20, 0]
        result = mgr.generate_landmarks([g1, g2], ["A", "B"], ["ROOM_001", "ROOM_002"])
        assert len(result) == 2
        assert result[0]["id"] == "PLATFORM_LANDMARK_001"
        assert result[1]["id"] == "PLATFORM_LANDMARK_002"

    def test_landmark_inside_room(self):
        """landmark 应在房间内部"""
        mgr = LandmarkManager({"landmark_min_distance": 0.1})
        geom = [0, 0, 100, 0, 100, 100, 0, 100, 0, 0]
        result = mgr.generate_landmarks([geom], ["A"], ["ROOM_001"])
        x, y = result[0]["geometry"][0], result[0]["geometry"][1]
        assert 0 < x < 100
        assert 0 < y < 100

    def test_avoids_marker_polygon(self):
        """landmark 应避开家具标记区域"""
        mgr = LandmarkManager({"landmark_min_distance": 0.5})
        # 大房间
        geom = [0, 0, 100, 0, 100, 100, 0, 100, 0, 0]
        # 家具在中心
        marker = [45, 45, 55, 45, 55, 55, 45, 55, 45, 45]
        result = mgr.generate_landmarks(
            [geom], ["A"], ["ROOM_001"], marker_polygons=[marker]
        )
        x, y = result[0]["geometry"][0], result[0]["geometry"][1]
        # 不应在家具区域内
        assert not (45 <= x <= 55 and 45 <= y <= 55)

    def test_empty_geometry(self):
        """空 geometry 应返回 (0, 0)"""
        mgr = LandmarkManager()
        result = mgr.generate_landmarks([[]], ["A"], ["ROOM_001"])
        assert len(result) == 1
        assert result[0]["geometry"] == [0.0, 0.0, 0]


class TestToPointList:
    def test_basic(self):
        pts = LandmarkManager._to_point_list([0, 0, 10, 0, 10, 10])
        assert len(pts) == 3

    def test_removes_closing_point(self):
        """闭合的重复末端点应被移除"""
        pts = LandmarkManager._to_point_list([0, 0, 10, 0, 10, 10, 0, 0])
        assert len(pts) == 3

    def test_empty(self):
        pts = LandmarkManager._to_point_list([])
        assert pts == []
