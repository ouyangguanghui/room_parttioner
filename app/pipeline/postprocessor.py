"""
模型后处理模块

数据流:
    模型输出 OBB 列表 → OBB 逆转回线段 → 线段沿方向延伸至墙壁 → threshold 掩膜
    → 切割自由空间 → 连通域标记 → 形态学优化 → 小区域过滤 → room_map

OBB 与 line 的关系 (参见 dataset/line_to_obb.py):
    line_to_obb: 线段 (p1,p2) + width → 沿法向膨胀 → OBB 四顶点
    顶点顺序:
        v0 = p1 + normal * w/2
        v1 = p2 + normal * w/2
        v2 = p2 - normal * w/2
        v3 = p1 - normal * w/2
    因此:
        短边 (v1→v2, v3→v0) 中点 = p2, p1  ← 原始线段端点
        长边 (v0→v1, v2→v3) 方向 = p2-p1   ← 原始线段方向
"""

import logging
from typing import Dict, Any, Tuple, List

import numpy as np
import cv2

logger = logging.getLogger(__name__)


class Postprocessor:
    """模型输出后处理：将原始推理结果转为房间标签图"""

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.min_room_area = self.config.get("min_room_area", 1.0)  # m²
        self.resolution = self.config.get("resolution", 0.05)  # m/pixel
        self.morph_kernel_size = self.config.get("morph_kernel_size", 5)
        self.max_extend = self.config.get("max_extend", 3.0)  # 最大延伸像素距离
        self.thickness = self.config.get("thickness", 2)

    def process(self, raw_output: List[List[int]], meta: Dict[str, Any]) -> np.ndarray:
        """
        后处理流水线

        Args:
            raw_output: 模型原始输出 OBB 列表，每个元素为 4 个顶点 [[x,y], ...]
            meta:
                map_data: 前处理后的地图 (补墙平滑后)
                input_data: 前处理前的原始输入

        Returns:
            room_map: 房间标签图 (H, W) int32, 0=背景
        """
        map_data, input_data = meta["map_data"], meta["input_data"]

        # step 1: OBB → threshold 线掩膜
        threshold_result = self._build_threshold_mask(raw_output, map_data)
        threshold_mask = threshold_result["threshold_mask"]

        # step 2: 用 threshold 线切割自由空间，连通域标记为房间
        room_map = self._split_by_threshold(threshold_mask, map_data, input_data)

        # step 3: 形态学优化 
        room_map = self._morphology_refine(room_map)
        # step 4: 小区域过滤
        room_map = self._filter_small_rooms(room_map)
        return room_map

    # ==================== Step 1: OBB → threshold 掩膜 ====================

    def _build_threshold_mask(
        self,
        obb_list: List[List[int]],
        map_data: np.ndarray,
    ) -> Dict[str, Any]:
        """
        将 OBB 列表转换为 threshold 线掩膜。

        流程 (每个 OBB):
            1. obb_to_line: OBB 四顶点 → 还原原始线段 (p1, p2) + 方向
            2. extend_line: 从 p1, p2 沿线段方向向外延伸至非自由像素
            3. 画线到掩膜

        Args:
            obb_list: OBB 列表, 每个 OBB 为 4 个顶点 [[x,y], ...]
            map_data: 地图数据 (灰度图, >=200 为自由空间)

        Returns:
            {
                "threshold_mask": (H, W) uint8, 255=threshold 线, 0=其他,
                "threshold_list": 延伸后的分割线端点列表 [[(x1,y1), (x2,y2)], ...],
                "line_list": OBB 还原的原始线段列表 [[(x1,y1), (x2,y2)], ...],
            }
        """
        h, w = map_data.shape[:2]
        free_mask = map_data >= 200

        threshold_mask = np.zeros((h, w), dtype=np.uint8)
        threshold_list = []
        line_list = []

        for obb in obb_list or []:
            # 1) OBB → 原始线段
            line_result = self._obb_to_line(obb)
            if line_result is None:
                continue

            p1, p2, direction = line_result
            line_list.append([p1, p2])

            # 2) 从线段两端沿方向向外延伸至墙壁
            #    direction: p1 → p2 的单位向量
            #    p2 沿 +direction 延伸, p1 沿 -direction 延伸
            ext_p1 = self._extend_to_wall(p1, -direction, free_mask, self.max_extend)
            ext_p2 = self._extend_to_wall(p2, direction, free_mask, self.max_extend)

            # 3) 画到掩膜
            pt1 = (int(round(ext_p1[0])), int(round(ext_p1[1])))
            pt2 = (int(round(ext_p2[0])), int(round(ext_p2[1])))
            cv2.line(threshold_mask, pt1, pt2, color=255, thickness=self.thickness)
            threshold_list.append([ext_p1, ext_p2])

        logger.info("threshold 线: %d 个 OBB → %d 条 threshold",
                     len(obb_list or []), len(threshold_list))

        return {
            "threshold_mask": threshold_mask,
            "threshold_list": threshold_list,
            "line_list": line_list,
        }

    # ==================== Step 2: threshold 切割 → 房间标签 ====================

    @staticmethod
    def _split_by_threshold(
        threshold_mask: np.ndarray,
        map_data: np.ndarray,
        input_data: np.ndarray,
    ) -> np.ndarray:
        """
        用 threshold 线掩膜切割自由空间，连通域标记为房间。

        Args:
            threshold_mask: (H, W) uint8, 非零=threshold 线
            map_data: 前处理后的地图 (补墙平滑后)
            input_data: 前处理前的原始输入

        Returns:
            room_map: (H, W) int32, 0=背景, >0=房间标签
        """
        free_mask = map_data >= 200
        if not free_mask.any():
            free_mask = input_data >= 200
        if not free_mask.any():
            free_mask = map_data > 0

        split_mask = (free_mask & (threshold_mask == 0)).astype(np.uint8)
        _, labels = cv2.connectedComponents(split_mask, connectivity=8)
        return labels.astype(np.int32)

    # ==================== OBB → line 逆运算 ====================

    @staticmethod
    def _obb_to_line(
        obb: List[List[int]],
    ) -> Tuple[Tuple[float, float], Tuple[float, float], np.ndarray]:
        """
        OBB 四顶点 → 还原原始线段 (line_to_obb 的逆运算)

        line_to_obb 的顶点顺序:
            v0 = p1 + normal * w/2
            v1 = p2 + normal * w/2
            v2 = p2 - normal * w/2
            v3 = p1 - normal * w/2

        因此短边 (宽度方向) 的两个中点就是原始线段的 p1, p2。

        Args:
            obb: 4 个顶点 [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]

        Returns:
            (p1, p2, direction) 或 None
            - p1, p2: 原始线段端点 (float, float)
            - direction: p1→p2 单位方向向量
        """
        pts = np.asarray(obb, dtype=np.float32).reshape(-1, 2)
        if pts.shape[0] != 4:
            return None

        # 4 条边及其长度
        edges = np.roll(pts, -1, axis=0) - pts
        lengths = np.linalg.norm(edges, axis=1)
        if np.all(lengths < 1e-6):
            return None

        # 最短边 = OBB 宽度方向 (即原始 line_to_obb 的 width)
        short_idx = int(np.argmin(lengths))
        opp_idx = (short_idx + 2) % 4

        # 两条短边中点 = 原始线段端点
        # 短边 short_idx: pts[short_idx] → pts[(short_idx+1)%4]
        # 短边 opp_idx:   pts[opp_idx]   → pts[(opp_idx+1)%4]
        p1 = (pts[opp_idx] + pts[(opp_idx + 1) % 4]) / 2
        p2 = (pts[short_idx] + pts[(short_idx + 1) % 4]) / 2

        # 线段方向: p1 → p2
        direction = p2 - p1
        norm = float(np.linalg.norm(direction))
        if norm < 1e-6:
            return None
        direction = direction / norm

        p1_tuple = (float(p1[0]), float(p1[1]))
        p2_tuple = (float(p2[0]), float(p2[1]))
        return p1_tuple, p2_tuple, direction

    @staticmethod
    def _extend_to_wall(
        start: Tuple[float, float],
        direction: np.ndarray,
        free_mask: np.ndarray,
        max_extend: float = 50.0,
    ) -> Tuple[float, float]:
        """
        从起点沿方向逐像素延伸，直到遇到非自由像素、越界或超过最大延伸距离。

        Args:
            start: 起始坐标 (x, y)
            direction: 单位方向向量
            free_mask: 自由空间掩膜 (True=自由)
            max_extend: 最大延伸像素距离

        Returns:
            最后一个自由像素的坐标 (x, y)
        """
        h, w = free_mask.shape[:2]
        x, y = float(start[0]), float(start[1])
        last_valid = (x, y)

        max_steps = int(max_extend)
        for _ in range(max_steps):
            x += float(direction[0])
            y += float(direction[1])
            xi, yi = int(round(x)), int(round(y))

            if xi < 0 or xi >= w or yi < 0 or yi >= h:
                break
            if not free_mask[yi, xi]:
                break
            last_valid = (x, y)

        return last_valid

    # ==================== Step 3: 形态学 + 过滤 ====================

    def _morphology_refine(self, label_map: np.ndarray) -> np.ndarray:
        """形态学优化：对每个标签做开运算去噪"""
        refined = np.zeros_like(label_map)
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (self.morph_kernel_size, self.morph_kernel_size)
        )
        for lid in range(1, label_map.max() + 1):
            mask = (label_map == lid).astype(np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            refined[mask > 0] = lid
        return refined

    def _filter_small_rooms(self, label_map: np.ndarray) -> np.ndarray:
        """过滤面积过小的区域"""
        filtered = label_map.copy()
        pixel_area = self.resolution ** 2
        for lid in range(1, label_map.max() + 1):
            mask = label_map == lid
            area = mask.sum() * pixel_area
            if area < self.min_room_area:
                filtered[mask] = 0
        return filtered
