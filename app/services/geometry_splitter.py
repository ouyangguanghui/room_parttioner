"""几何级多边形分割服务 —— 在世界坐标 geometry 上执行画线分割"""

import math
import logging
from typing import Dict, Any, List, Tuple

import numpy as np
import cv2

from app.utils.coordinate import CoordinateTransformer
from app.core.errors import (
    InsufficientIntersectionsError,
    RoomIndexOutOfRangeError,
    RoomTooSmallError,
    SplitNotConnectedError,
)

logger = logging.getLogger(__name__)


class GeometrySplitter:
    """
    世界坐标级的房间分割

    用于 Lambda 场景：用户画线分割一个房间，操作直接在 labels_json 的
    geometry 多边形上进行（而非像素级 label_map）。
    """

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}

    def split(
        self,
        rooms_data: List[Dict[str, Any]],
        division_croods_dict: Dict,
        transformer: CoordinateTransformer,
        map_img: np.ndarray,
        resolution: float,
    ) -> Tuple[List[Dict[str, Any]], int]:
        """
        执行分割操作

        Args:
            rooms_data: labels_json 中的 ROOM 数据列表
            division_croods_dict: {"id": "ROOM_001", "A": [wx, wy], "B": [wx, wy]}
            transformer: 坐标转换器
            map_img: 原始地图 (BGR)
            resolution: 地图分辨率

        Returns:
            (更新后的 rooms_data, 新增房间的索引)

        Raises:
            RoomIndexOutOfRangeError, InsufficientIntersectionsError,
            RoomTooSmallError, SplitNotConnectedError
        """
        h, w = map_img.shape[:2]

        # 解析参数
        room_id = division_croods_dict['id']
        seg_idx = int(room_id.split('_')[-1]) - 1

        if seg_idx >= len(rooms_data):
            raise RoomIndexOutOfRangeError()

        A = division_croods_dict['A']
        B = division_croods_dict['B']

        # 在 geometry 上找交点, 分割多边形
        geometry = rooms_data[seg_idx]['geometry']
        ok, result = self._find_split_points(A, B, geometry)
        if not ok:
            raise InsufficientIntersectionsError()

        poly_a, poly_b, intersections = result

        # 构造新 geometry
        geom_a = self._flatten_geometry(poly_a)
        geom_b = self._flatten_geometry(poly_b)

        # 面积检查 (像素)
        cnt_a = transformer.world_to_contour(geom_a)
        cnt_b = transformer.world_to_contour(geom_b)
        area_min = 0.25 / (resolution ** 2)
        if cv2.contourArea(cnt_a) < area_min or cv2.contourArea(cnt_b) < area_min:
            raise RoomTooSmallError()

        # 连通性检查
        for cnt in [cnt_a, cnt_b]:
            mask = np.zeros((h, w), dtype=np.uint8)
            cv2.drawContours(mask, [cnt], -1, 255, -1)
            kernel = np.ones((5, 5), np.uint8)
            eroded = cv2.erode(mask, kernel, iterations=1)
            n_obj, _, stats, _ = cv2.connectedComponentsWithStats(eroded)
            if n_obj != 2 and all(
                stats[k, cv2.CC_STAT_AREA] >= 0.10 / (resolution ** 2)
                for k in range(1, n_obj)
            ):
                raise SplitNotConnectedError()

        # 更新 labels
        original_name = rooms_data[seg_idx]['name']
        ground = rooms_data[seg_idx].get('groundMaterial')

        rooms_data[seg_idx]['geometry'] = geom_a
        new_id = f"ROOM_{len(rooms_data) + 1:03d}"
        new_name = self._next_room_name(rooms_data)

        rooms_data.append({
            "name": new_name,
            "id": new_id,
            "type": "polygon",
            "geometry": geom_b,
            "colorType": None,
            "graph": None,
            "groundMaterial": ground,
        })

        # 大面积保留原名
        if cv2.contourArea(cnt_a) >= cv2.contourArea(cnt_b):
            rooms_data[seg_idx]['name'] = original_name
            rooms_data[-1]['name'] = new_name
            select_idx = len(rooms_data) - 1
        else:
            rooms_data[seg_idx]['name'] = new_name
            rooms_data[-1]['name'] = original_name
            select_idx = seg_idx

        return rooms_data, select_idx

    # ==================== 内部工具 ====================

    @staticmethod
    def _find_split_points(A, B, geometry):
        """找分割线与多边形的交点, 拆分多边形"""
        contour_pts = [(geometry[i], geometry[i + 1])
                       for i in range(0, len(geometry), 2)]

        intersections = []
        intersection_indices = []

        for i in range(len(contour_pts)):
            p1 = contour_pts[i]
            p2 = contour_pts[(i + 1) % len(contour_pts)]
            ip = GeometrySplitter._line_intersection(A, B, p1, p2)
            if ip:
                intersections.append(ip)
                intersection_indices.append(i + 1)

        if len(intersections) < 2:
            return False, "交点不足两个"

        if len(intersections) > 2:
            # 选离 A/B 最近的两个
            da = [math.dist(A, p) for p in intersections]
            db = [math.dist(B, p) for p in intersections]
            ia = da.index(min(da))
            ib = db.index(min(db))
            i1, i2 = min(ia, ib), max(ia, ib)
            intersections = [intersections[i1], intersections[i2]]
            intersection_indices = [intersection_indices[i1], intersection_indices[i2]]

        idx1, idx2 = intersection_indices
        poly_a = contour_pts[idx1:idx2] + [intersections[1], intersections[0]]
        poly_b = contour_pts[idx2:] + contour_pts[:idx1] + [intersections[0], intersections[1]]

        return True, (poly_a, poly_b, intersections)

    @staticmethod
    def _line_intersection(A, B, p1, p2):
        """计算线段 AB 与线段 p1p2 的交点"""
        if A == B:
            return None

        ax, ay = A
        bx, by = B
        p1x, p1y = p1
        p2x, p2y = p2

        if ax == bx:  # AB 垂直
            x = ax
            if p1x == p2x:
                return None
            if min(p1x, p2x) <= x <= max(p1x, p2x):
                k = (p2y - p1y) / (p2x - p1x)
                y = k * (x - p1x) + p1y
                return (x, y)
        else:
            k1 = (by - ay) / (bx - ax)
            b1 = ay - k1 * ax

            if p1x == p2x:
                x = p1x
                y = k1 * x + b1
                if min(p1y, p2y) <= y <= max(p1y, p2y):
                    return (x, y)
            else:
                k2 = (p2y - p1y) / (p2x - p1x)
                b2 = p1y - k2 * p1x
                cross_a = k1 * p1x + b1 - p1y
                cross_b = k1 * p2x + b1 - p2y
                if cross_a * cross_b <= 0:
                    if k1 == k2:
                        return None
                    x = (b2 - b1) / (k1 - k2)
                    y = k1 * x + b1
                    return (x, y)
        return None

    @staticmethod
    def _flatten_geometry(poly_pts):
        """多边形点列表 → flat geometry [x0,y0,x1,y1,...,x0,y0]"""
        geom = []
        for pt in poly_pts:
            geom.extend([pt[0], pt[1]])
        geom.extend([poly_pts[0][0], poly_pts[0][1]])
        return geom

    @staticmethod
    def _next_room_name(rooms_data):
        """分配下一个可用房间名 (A~Z)"""
        used = {r.get('name') for r in rooms_data}
        name = chr(ord('A'))
        while name in used:
            name = chr(ord(name) + 1)
        return name
