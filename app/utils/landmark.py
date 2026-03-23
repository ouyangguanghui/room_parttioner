"""平台标记点管理 —— 在每个房间内生成清扫起始点 (k20 机型)"""

import math
from typing import Dict, Any, List, Tuple, Optional


class LandmarkPoint:
    """二维点, 支持距离计算和多边形判定"""

    __slots__ = ('x', 'y')

    def __init__(self, x: float, y: float):
        self.x = x
        self.y = y

    def distance_to(self, other: 'LandmarkPoint') -> float:
        return math.sqrt((self.x - other.x) ** 2 + (self.y - other.y) ** 2)

    def distance_to_line(self, p1: 'LandmarkPoint',
                         p2: 'LandmarkPoint') -> float:
        """点到线段的距离"""
        ab_sq = (p2.x - p1.x) ** 2 + (p2.y - p1.y) ** 2
        if ab_sq < 1e-10:
            return self.distance_to(p1)
        t = max(0, min(1, (
            (self.x - p1.x) * (p2.x - p1.x) +
            (self.y - p1.y) * (p2.y - p1.y)
        ) / ab_sq))
        proj = LandmarkPoint(p1.x + t * (p2.x - p1.x),
                             p1.y + t * (p2.y - p1.y))
        return self.distance_to(proj)

    def distance_to_polygon(self, polygon: List['LandmarkPoint']) -> float:
        """点到多边形边界的最小距离"""
        n = len(polygon)
        return min(
            self.distance_to_line(polygon[i], polygon[(i + 1) % n])
            for i in range(n)
        )

    def is_inside_polygon(self, polygon: List['LandmarkPoint']) -> bool:
        """射线法判断点是否在多边形内部"""
        n = len(polygon)
        inside = False
        for i in range(n):
            p1, p2 = polygon[i], polygon[(i + 1) % n]
            if (p1.y > self.y) != (p2.y > self.y):
                if p1.x == p2.x:
                    if self.x < p1.x:
                        inside = not inside
                else:
                    ix = (p2.x - p1.x) * (self.y - p1.y) / (p2.y - p1.y) + p1.x
                    if self.x < ix:
                        inside = not inside
        return inside


class LandmarkManager:
    """
    平台标记点管理器 (k20 机型专用)

    为每个房间生成一个清扫起始点:
    - 在房间多边形内部
    - 远离边界
    - 避开家具标记区域
    """

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.min_distance = self.config.get("landmark_min_distance", 0.1)
        self.grid_size = 50

    def generate_landmarks(
        self,
        rooms_geometry: List[List[float]],
        room_names: List[str],
        room_ids: List[str],
        marker_polygons: List[List[float]] = None,
    ) -> List[Dict[str, Any]]:
        """
        为所有房间生成平台标记点

        Args:
            rooms_geometry: 各房间的 geometry [x0,y0,x1,y1,...]
            room_names: 房间名称
            room_ids: 房间 ID
            marker_polygons: 家具标记多边形 (世界坐标)

        Returns:
            [{"geometry": [x,y,0], "id": "PLATFORM_LANDMARK_001",
              "roomId": "ROOM_001", "name": "A", "type": "pose"}, ...]
        """
        markers = []
        if marker_polygons:
            markers = [self._to_point_list(g) for g in marker_polygons]

        landmarks = []
        for i, (geom, name, rid) in enumerate(
            zip(rooms_geometry, room_names, room_ids)
        ):
            center = self._find_center(geom, markers)
            landmarks.append({
                "geometry": [center[0], center[1], 0],
                "id": f"PLATFORM_LANDMARK_{len(landmarks) + 1:03d}",
                "roomId": rid,
                "name": name,
                "type": "pose",
            })
        return landmarks

    def _find_center(self, geometry: List[float],
                     markers: List[List[LandmarkPoint]]) -> Tuple[float, float]:
        """在多边形内找合适的中心点"""
        poly = self._to_point_list(geometry)
        if not poly:
            return (0.0, 0.0)

        # 计算边界框
        xs = [p.x for p in poly]
        ys = [p.y for p in poly]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)

        # 先尝试质心
        cx = (min_x + max_x) / 2
        cy = (min_y + max_y) / 2
        centroid = LandmarkPoint(cx, cy)
        if self._is_valid(centroid, poly, markers):
            return (cx, cy)

        # 网格搜索
        best = None
        best_dist = -1
        x_coords = [min_x + (max_x - min_x) * i / self.grid_size
                     for i in range(self.grid_size + 1)]
        y_coords = [min_y + (max_y - min_y) * i / self.grid_size
                     for i in range(self.grid_size + 1)]

        for x in x_coords:
            for y in y_coords:
                p = LandmarkPoint(x, y)
                if not self._is_valid(p, poly, markers):
                    continue
                d = p.distance_to_polygon(poly)
                if d > best_dist:
                    best_dist = d
                    best = (x, y)

        return best if best else (cx, cy)

    def _is_valid(self, point: LandmarkPoint,
                  polygon: List[LandmarkPoint],
                  markers: List[List[LandmarkPoint]]) -> bool:
        """检查点是否满足条件"""
        if not point.is_inside_polygon(polygon):
            return False
        if point.distance_to_polygon(polygon) <= self.min_distance:
            return False
        for m in markers:
            if not m:
                continue
            if point.is_inside_polygon(m):
                return False
            if point.distance_to_polygon(m) <= self.min_distance:
                return False
        return True

    @staticmethod
    def _to_point_list(geometry: List[float]) -> List[LandmarkPoint]:
        """geometry flat list → LandmarkPoint 列表"""
        points = []
        for i in range(0, len(geometry) - 1, 2):
            points.append(LandmarkPoint(geometry[i], geometry[i + 1]))
        # 去除闭合的重复点
        if len(points) > 1 and (
            abs(points[0].x - points[-1].x) < 1e-6 and
            abs(points[0].y - points[-1].y) < 1e-6
        ):
            points.pop()
        return points
