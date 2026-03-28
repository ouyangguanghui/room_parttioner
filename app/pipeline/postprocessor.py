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

from cgitb import small
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

    def process(self, raw_output: List[List[int]], meta: Dict[str, Any]) -> Dict[str, Any]:
        f"""
        后处理流水线

        Args:
            raw_output: 模型原始输出 OBB 列表，每个元素为 4 个顶点 [[x,y], ...]
            meta:
                map_data:      去噪但未补墙的地图（自由空间判断备用
                input_data:    补墙平滑后的地图
                tensor_scale:  (可选) letterbox 缩放比，用于 OBB 坐标逆映射
                tensor_pad:    (可选) (pad_top, pad_left) letterbox 填充，用于逆映射

        Returns:
            Dict[str, Any]:
                "room_map": 房间标签图 (H, W) int32, 0=背景
                "threshold_list": 延伸后的分割线端点列表 [[(x1,y1), (x2,y2)], ...],
                "thickness_size" : 分割线粗细
        """
        map_data  = meta["map_data"]

        # step 1: OBB 坐标逆映射 + 画 threshold 线掩膜
        threshold_result = self._build_threshold_mask(raw_output, map_data, meta)
        threshold_mask = threshold_result["threshold_mask"]

        # step 2: threshold 线切割自由空间 → 连通域标记
        room_map = self._split_by_threshold(threshold_mask, map_data)

        # step 3: 按面积阈值拆分 → 正常区域图 + 碎片图
        normal_map, small_map = self._split_by_area(room_map, map_data)

        # step 4: 碎片合并
        room_map = self._merge_fragments(normal_map, small_map)

        return {
            "room_map": room_map,
            "threshold_list": threshold_result["threshold_list"],
            "thickness_size": self.thickness,
        }

    # ==================== Step 1: OBB → threshold 掩膜 ====================

    def _build_threshold_mask(
        self,
        obb_list: List[List[int]],
        map_data: np.ndarray,
        meta: Dict[str, Any] = None,
    ) -> Dict[str, Any]:
        """
        将 OBB 列表转换为 threshold 线掩膜。

        若 meta 中含有 tensor_scale / tensor_pad，则先将 OBB 顶点坐标从
        Triton 输入空间（letterboxed）逆映射回原始地图像素坐标，再画线。

        流程 (每个 OBB):
            1. 坐标逆映射: tensor 空间 → 原始地图空间
            2. obb_to_line: OBB 四顶点 → 还原原始线段 (p1, p2) + 方向
            3. extend_line: 从 p1, p2 沿线段方向向外延伸至非自由像素
            4. 画线到掩膜

        Args:
            obb_list: OBB 列表, 每个 OBB 为 4 个顶点 [[x,y], ...]
                      坐标可能处于 tensor 空间，需通过 meta 逆映射
            map_data: 地图数据 (灰度图, >=200 为自由空间)
            meta:     (可选) 含 tensor_scale / tensor_pad 的字典

        Returns:
            {
                "threshold_mask": (H, W) uint8, 255=threshold 线, 0=其他,
                "threshold_list": 延伸后的分割线端点列表 [[(x1,y1), (x2,y2)], ...],
                "line_list": OBB 还原的原始线段列表 [[(x1,y1), (x2,y2)], ...],
            }
        """
        h, w = map_data.shape[:2]
        free_mask = map_data >= 200

        # 解析坐标逆映射参数
        scale    = (meta or {}).get("tensor_scale", 1.0)
        pad_top, pad_left = (meta or {}).get("tensor_pad", (0, 0))
        pre_pad_top, pre_pad_left = (meta or {}).get("pre_pad", (0, 0))
        need_remap = (scale != 1.0 or pad_top != 0 or pad_left != 0 or pre_pad_top != 0 or pre_pad_left != 0)

        threshold_mask = np.zeros((h, w), dtype=np.uint8)
        threshold_list = []
        line_list = []

        for obb in obb_list or []:
            # 1) 坐标逆映射：tensor 空间 → letterbox undo → pre-pad undo → 原始地图像素空间
            if need_remap:
                obb = [
                    [(pt[0] - pad_left) / scale - pre_pad_left,
                     (pt[1] - pad_top) / scale - pre_pad_top]
                    for pt in obb
                ]

            # 2) OBB → 原始线段
            line_result = self._obb_to_line(obb)
            if line_result is None:
                continue

            p1, p2, direction = line_result
            line_list.append([p1, p2])

            # 3) 从线段两端沿方向向外延伸至墙壁
            ext_p1 = self._extend_to_wall(p1, -direction, free_mask, self.max_extend)
            ext_p2 = self._extend_to_wall(p2,  direction, free_mask, self.max_extend)

            # 4) 画到掩膜
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
        map_data: np.ndarray
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
        # 生成自由空间掩膜。理想情况下，自由空间像素值应大于等于200（通常是255），
        # 但有时 map_data 全部不是自由空间（比如补墙后只剩墙壁/未知），就用 input_data 再尝试一次。
        # 如果仍然检测不到自由空间区域，则采取最后的保险方案：将 map_data 中所有非零像素都视为自由空间。
        free_mask = map_data >= 200
        if not free_mask.any():
            free_mask = map_data > 0

        # 在自由空间（free_mask）且未被 threshold 线覆盖（threshold_mask==0）的区域作为可分割空间 split_mask。
        # 用 connectedComponents 寻找这些连通的自由空间块，每个块会被赋予一个独立的标签，即房间编号。
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


    # ==================== Step 3: 面积拆分 ====================

    def _split_by_area(
        self, label_map: np.ndarray, map_data: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        按面积阈值将 label_map 拆分为两张独立的图。

        Args:
            label_map: (H, W) int32，0=背景，>0=房间标签
            map_data: 前处理后的地图 (补墙平滑前)
        Returns:
            normal_map: 面积 >= min_room_area 的区域保留原标签，其余置 0
            small_map:  面积 <  min_room_area 的区域保留原标签，其余置 0
        """
        pixel_area = self.resolution ** 2
        min_pixels = max(1, int(self.min_room_area / pixel_area))

        normal_map = np.zeros_like(label_map)
        small_map  = np.zeros_like(label_map)

        for lid in range(1, label_map.max() + 1):
            mask = label_map == lid
            if not mask.any():
                continue
            if mask.sum() >= min_pixels:
                normal_map[mask] = lid
            else:
                small_map[mask] = lid

        return normal_map, small_map

    # ==================== Step 4: 碎片合并 ====================

    def _merge_fragments(
        self, normal_map: np.ndarray, small_map: np.ndarray
    ) -> np.ndarray:
        """
        将碎片图中的每个小区域全部合并进正常区域图，不丢弃任何碎片。

        策略：
            1. 按面积从大到小排序处理——大碎片先合并后可成为后续小碎片的"锚点"。
            2. 对每个碎片，先尝试 1 像素膨胀快速找邻居；
               若未找到，渐进式扩大搜索掩膜（仅扩展搜索前沿，不改变碎片本身），
               直到搜索前沿触碰到 result 中已标记的区域为止。
            3. 搜索成功后，选取与搜索前沿接触最多的邻室，
               将碎片像素全部归入该邻室。
            4. 极端情况（地图中完全无正常区域）：将碎片像素置 0（背景），
               并记录警告日志。

        Args:
            normal_map: step 3 输出的正常区域图
            small_map:  step 3 输出的碎片图

        Returns:
            合并后的结果图，所有碎片均已归入某个正常区域
        """
        result = normal_map.copy()
        kernel = np.ones((3, 3), np.uint8)

        # 渐进搜索最大半径：取地图对角线像素数，确保必然能覆盖全图
        h, w   = result.shape[:2]
        max_search = int((h ** 2 + w ** 2) ** 0.5) + 1

        # 按面积从大到小排序，优先处理较大碎片
        frag_ids = [int(lid) for lid in np.unique(small_map) if lid > 0]
        frag_ids.sort(key=lambda lid: -int((small_map == lid).sum()))

        merged_count  = 0
        fallback_count = 0

        for lid in frag_ids:
            frag_mask  = small_map == lid
            frag_u8    = frag_mask.astype(np.uint8)

            # ── 阶段一：1 像素膨胀快速路径 ──────────────────────────
            dilated     = cv2.dilate(frag_u8, kernel, iterations=1)
            neighbor_px = result[(dilated > 0) & ~frag_mask & (result > 0)]

            if neighbor_px.size > 0:
                best = int(np.bincount(neighbor_px).argmax())
                result[frag_mask] = best
                merged_count += 1
                continue

            # ── 阶段二：渐进式扩展搜索前沿 ──────────────────────────
            # search_front 仅保存"搜索边界"，不累积，避免覆盖碎片自身
            search_front = frag_u8.copy()
            best = 0

            for _ in range(max_search):
                search_front = cv2.dilate(search_front, kernel, iterations=1)
                # 只看搜索前沿扩展到的新区域（排除碎片本身）
                frontier = (search_front > 0) & ~frag_mask
                neighbor_px = result[frontier & (result > 0)]

                if neighbor_px.size > 0:
                    best = int(np.bincount(neighbor_px).argmax())
                    break

            if best > 0:
                result[frag_mask] = best
                merged_count += 1
            else:
                # 地图中完全无正常区域（极端情况）
                result[frag_mask] = 0
                fallback_count += 1
                logger.warning("碎片 %d 未找到任何正常区域，置为背景", lid)

        logger.info("碎片合并: 全部 %d 个碎片处理完毕 (合并 %d 个, 极端置背景 %d 个)",
                    len(frag_ids), merged_count, fallback_count)
        return result
