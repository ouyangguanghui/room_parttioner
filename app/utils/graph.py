"""房间邻接图构建 + 图着色"""

import random
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
import cv2


class RoomGraph:
    """
    房间拓扑关系管理

    功能:
    - 构建房间邻接图 (哪些房间共享边界)
    - 图着色 (5 色, 确保相邻房间颜色不同)
    - DFS 房间排序 (从指定起点遍历)
    """

    NUM_COLORS = 5  # 0-4 共 5 种颜色

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.resolution = self.config.get("resolution", 0.05)
        self._graph: Dict[int, List[int]] = {}

    # ==================== 邻接图构建 ====================

    def build_graph(self, contours: List[np.ndarray],
                    map_img: np.ndarray) -> Dict[int, List[int]]:
        """
        构建房间邻接图

        两个房间"相邻" = 它们的填充区域合并后是一个连通域，
        或者中间的间隙面积小于阈值 (0.10 m²)

        Args:
            contours: 各房间轮廓 [(N,1,2), ...]
            map_img: 灰度地图 (用于判断墙壁)

        Returns:
            {room_index: [neighbor_indices]}
        """
        h, w = map_img.shape[:2]
        n = len(contours)
        graph: Dict[int, List[int]] = {i: [] for i in range(n)}
        min_gap_pixels = 0.10 / (self.resolution ** 2)

        # 灰度图: 墙壁标为255方便后续掩码运算
        if map_img.ndim == 3:
            gray = cv2.cvtColor(map_img, cv2.COLOR_BGR2GRAY)
        else:
            gray = map_img.copy()
        wall_mask = np.zeros((h, w), dtype=np.uint8)
        wall_mask[gray == 0] = 255
        wall_mask[gray == 127] = 127

        for i in range(n):
            for j in range(i + 1, n):
                if self._are_adjacent(contours[i], contours[j],
                                      wall_mask, h, w, min_gap_pixels):
                    graph[i].append(j)
                    graph[j].append(i)

        self._graph = graph
        return graph

    def _are_adjacent(self, cnt1: np.ndarray, cnt2: np.ndarray,
                      wall_mask: np.ndarray, h: int, w: int,
                      min_gap_pixels: float) -> bool:
        """判断两个轮廓是否相邻"""
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.drawContours(mask, [cnt1], -1, 255, -1)
        cv2.drawContours(mask, [cnt2], -1, 255, -1)

        # 墙壁区域也算连通
        mask[(wall_mask == 255) & (mask == 255)] = 255
        mask[(wall_mask == 127) & (mask == 255)] = 127

        num_objects, _, stats, _ = cv2.connectedComponentsWithStats(mask)
        # 合并后是单个连通域 → 相邻
        if num_objects == 2:
            return True
        # 间隙面积都小于阈值 → 也算相邻
        all_small = all(
            stats[k, cv2.CC_STAT_AREA] < min_gap_pixels
            for k in range(1, num_objects)
        )
        return not all_small if num_objects > 2 else False

    def check_connectivity(self, cnt1: np.ndarray, cnt2: np.ndarray,
                           map_img: np.ndarray) -> bool:
        """检查两个轮廓是否连通 (用于合并/分割后重建图)"""
        h, w = map_img.shape[:2]
        if map_img.ndim == 3:
            gray = cv2.cvtColor(map_img, cv2.COLOR_BGR2GRAY)
        else:
            gray = map_img.copy()
        wall_mask = np.zeros((h, w), dtype=np.uint8)
        wall_mask[gray == 0] = 255
        wall_mask[gray == 127] = 127
        min_gap = 0.10 / (self.resolution ** 2)
        return self._are_adjacent(cnt1, cnt2, wall_mask, h, w, min_gap)

    # ==================== 图着色 ====================

    def assign_colors(self, graph: Dict[int, List[int]] = None) -> Dict[int, int]:
        """
        贪心图着色 (5色)

        按邻居数量从少到多排序，依次分配最小可用颜色。

        Returns:
            {room_index: color_id (0-4)}
        """
        g = graph or self._graph
        colors: Dict[int, Optional[int]] = {k: None for k in g}

        # 按邻居数量升序
        sorted_rooms = sorted(g.keys(), key=lambda k: len(g[k]))

        for room in sorted_rooms:
            neighbor_colors = {
                colors[nb] for nb in g[room]
                if colors[nb] is not None
            }
            # 分配最小可用颜色
            for c in range(self.NUM_COLORS):
                if c not in neighbor_colors:
                    colors[room] = c
                    break
            else:
                # 5 色都被占了，随机选一个
                colors[room] = random.randint(0, self.NUM_COLORS - 1)

        return colors

    def assign_color_for_room(self, room_idx: int,
                              graph: Dict[int, List[int]],
                              current_colors: Dict[int, int]) -> int:
        """为单个房间分配颜色 (分割/合并后用)"""
        
        neighbor_colors = {
            current_colors[nb] for nb in graph.get(room_idx, [])
            if nb in current_colors and current_colors[nb] is not None
        }
        for c in range(self.NUM_COLORS):
            if c not in neighbor_colors:
                return c
        return random.randint(0, self.NUM_COLORS - 1)

    # ==================== DFS 房间排序 ====================

    def dfs_sort(self, graph: Dict[int, List[int]] = None,
                 start: int = None) -> List[int]:
        """
        DFS 遍历房间，返回遍历顺序

        如果图不连通，会继续遍历剩余连通分量。

        Args:
            graph: 邻接图
            start: 起始房间索引 (None=自动选邻居最多的)

        Returns:
            [room_index, ...] 按 DFS 顺序
        """
        g = graph or self._graph
        if not g:
            return []

        if start is None:
            start = max(g.keys(), key=lambda k: len(g[k]))

        visited = set()
        order = []

        def _dfs(node):
            visited.add(node)
            order.append(node)
            # 按邻居数量降序访问
            neighbors = sorted(g.get(node, []),
                               key=lambda x: len(g.get(x, [])), reverse=True)
            for nb in neighbors:
                if nb not in visited:
                    _dfs(nb)

        _dfs(start)

        # 处理不连通的分量
        remaining = set(g.keys()) - visited
        while remaining:
            next_start = max(remaining, key=lambda k: len(g.get(k, [])))
            _dfs(next_start)
            remaining = set(g.keys()) - visited

        return order

    def find_start_room(self, contours: List[np.ndarray],
                        charge_point_pixel: Tuple[int, int],
                        max_area_start: bool = False) -> int:
        """
        从充电桩位置找到起始房间

        Args:
            contours: 各房间轮廓
            charge_point_pixel: 充电桩像素坐标 (x, y)

        Returns:
            起始房间索引
        """
       # 如果charge_point_pixel为None，面积最大房间为起始房间
        if max_area_start:
            max_area = 0
            start_room_idx = 0
            for i, cnt in enumerate(contours):
                area = cv2.contourArea(cnt)
                if area > max_area:
                    max_area = area
                    start_room_idx = i
            return start_room_idx

        pt = tuple(charge_point_pixel)

        # 先找包含充电桩的房间
        for i, cnt in enumerate(contours):
            if cv2.pointPolygonTest(cnt, pt, False) >= 0:
                return i

        # 没有包含的，找最近的
        min_dist = float('inf')
        closest = 0
        for i, cnt in enumerate(contours):
            dist = abs(cv2.pointPolygonTest(cnt, pt, True))
            if dist < min_dist:
                min_dist = dist
                closest = i
        return closest

    @property
    def graph(self) -> Dict[int, List[int]]:
        return self._graph
