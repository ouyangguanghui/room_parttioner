"""坐标转换模块 —— 像素坐标 ↔ 世界坐标"""

import math
from typing import Dict, Any, List, Tuple

import numpy as np


class CoordinateTransformer:
    """
    栅格地图像素坐标与世界坐标之间的转换

    世界坐标系: 机器人 SLAM 坐标 (米)
    像素坐标系: 图片左上角为原点, 向右x增, 向下y增

    转换公式:
        world_x = pixel_x * resolution + origin_x + offset
        world_y = (height - pixel_y - 1) * resolution + origin_y - offset
    """

    def __init__(self, resolution: float, origin: List[float], height: int,
                 offset: float = 0.025):
        """
        Args:
            resolution: 地图分辨率 (m/pixel)
            origin: 地图原点 [x, y] (世界坐标, 对应像素左下角)
            height: 地图像素高度
            offset: 坐标偏移量 (半个像素的世界距离)
        """
        self.resolution = resolution
        self.origin = origin
        self.height = height
        self.offset = offset
    
    def set_params(self, resolution: float, origin: List[float], height: int):
        """设置坐标转换器参数。"""
        self.resolution = resolution
        self.origin = origin
        self.height = height

    def pixel_to_world(self, px: int, py: int) -> Tuple[float, float]:
        """单个像素坐标 → 世界坐标"""
        wx = round(px * self.resolution + self.origin[0] + self.offset, 3)
        wy = round((self.height - py - 1) * self.resolution + self.origin[1] - self.offset, 3)
        return wx, wy

    @staticmethod
    def _round_half_away_from_zero(value: float) -> int:
        """稳定整型量化：0.5 始终远离 0，避免 Python round 的银行家舍入。"""
        if value >= 0:
            return int(math.floor(value + 0.5))
        return int(math.ceil(value - 0.5))

    def world_to_pixel(self, wx: float, wy: float) -> Tuple[int, int]:
        """单个世界坐标 → 像素坐标"""
        # 与 pixel_to_world 对称：减去/加回 offset，并使用稳定舍入避免 int 截断漂移。
        px_f = (wx - self.origin[0] - self.offset) / self.resolution
        py_f = self.height - (wy - self.origin[1] + self.offset) / self.resolution - 1
        px = self._round_half_away_from_zero(px_f)
        py = self._round_half_away_from_zero(py_f)

        # pix_x = (geometry[i] - self.origin[0]) / self.resolution
        # pix_y = h - (geometry[i+1] - self.origin[1]) / self.resolution - 1
        return px, py

    def contour_to_world(self, contour: np.ndarray) -> List[Tuple[float, float]]:
        """
        OpenCV 轮廓 (N,1,2) → 世界坐标列表 (去重, 保序)

        Args:
            contour: shape (N, 1, 2) int32

        Returns:
            [(wx, wy), ...] 去重后的世界坐标
        """
        coords = [self.pixel_to_world(int(pt[0][0]), int(pt[0][1])) for pt in contour]
        # 去重保序
        seen = set()
        unique = []
        for c in coords:
            if c not in seen:
                seen.add(c)
                unique.append(c)
        return unique

    def world_to_contour(self, geometry: List[float]) -> np.ndarray:
        """
        世界坐标 geometry [x0,y0,x1,y1,...] → OpenCV 轮廓 (N,1,2) int32
        """
        points = []
        for i in range(0, len(geometry) - 1, 2):  # 跳过最后的闭合点
            px, py = self.world_to_pixel(geometry[i], geometry[i + 1])
            points.append([px, py])
        # 去重
        seen = set()
        unique = []
        for p in points:
            key = (p[0], p[1])
            if key not in seen:
                seen.add(key)
                unique.append(p)
        return np.array(unique, dtype=np.int32).reshape(-1, 1, 2)
    
    def rooms_data_to_contours(self, rooms_data: List[Dict[str, Any]]) -> List[np.ndarray]:
        """
        房间数据 → 轮廓列表
        """
        return [self.world_to_contour(room['geometry']) for room in rooms_data]

    def contour_to_geometry(self, contour: np.ndarray, clockwise: bool = True) -> List[float]:
        """
        OpenCV 轮廓 → labels.json 的 geometry 格式 (闭合的 flat list)

        Args:
            contour: (N,1,2) int32
            clockwise: 是否确保顺时针方向

        Returns:
            [x0, y0, x1, y1, ..., x0, y0]
        """
        world_coords = self.contour_to_world(contour)
        if clockwise and not self._is_clockwise(world_coords):
            world_coords.reverse()
        geometry = []
        for wx, wy in world_coords:
            geometry.extend([wx, wy])
        # 闭合
        geometry.extend(geometry[:2])
        return geometry

    @staticmethod
    def _is_clockwise(points: List[Tuple[float, float]]) -> bool:
        """判断多边形顶点是否顺时针排列"""
        signed_area = sum(
            x1 * y2 - x2 * y1
            for (x1, y1), (x2, y2) in zip(points, points[1:] + [points[0]])
        )
        return signed_area < 0
