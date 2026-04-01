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
from typing import Dict, Any, Tuple, List, Set

import numpy as np
import cv2

from app.utils.labels_ops import expand_one

logger = logging.getLogger(f"{__name__} [Postprocessor]")


class Postprocessor:
    """模型输出后处理：将原始推理结果转为房间标签图"""

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.min_room_area = self.config.get("min_room_area", 1.0)  # m²
        self.resolution = self.config.get("resolution", 0.05)  # m/pixel
        self.morph_kernel_size = self.config.get("morph_kernel_size", 5)
        self.max_extend = self.config.get("max_extend", 3.0)  # 最大延伸像素距离
        self.thickness = self.config.get("thickness", 2)

    def process(self, raw_output: List[List[int]], map_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        后处理流水线

        Args:
            raw_output: 模型原始输出 OBB 列表，每个元素为 4 个顶点 [[x,y], ...]
            map_data: 地图数据字典
                input_img:     补墙平滑后的灰度地图 (H, W) uint8
                cleaned_img:   去噪但未补墙的地图 (H, W) uint8
                cleaned_img2:  补墙平滑后的地图 (H, W) uint8
                tensor_scale:  (可选) letterbox 缩放比，用于 OBB 坐标逆映射
                tensor_pad:    (可选) (pad_top, pad_left) letterbox 填充，用于逆映射
                pre_pad:       (可选) (pad_top, pad_left) 小图预填充偏移

        Returns:
            Dict[str, Any]:
                "room_map": 房间标签图 (H, W) int32, 0=背景
                "threshold_list": 延伸后的分割线端点列表 [[(x1,y1), (x2,y2)], ...],
                "thickness_size" : 分割线粗细
        """
        input_img = map_data["cleaned_img"]

        # step 1: OBB 坐标逆映射 + 画 threshold 线掩膜
        threshold_result = self._build_threshold_mask(raw_output, input_img, map_data)
        threshold_mask = threshold_result["threshold_mask"]

        # step 2: threshold 线切割自由空间 → 连通域标记
        room_map = self._split_by_threshold(threshold_mask, input_img)

        # step 3: 按面积阈值拆分 → 正常区域图 + 碎片图
        normal_map, small_map = self._split_by_area(room_map, input_img)

        # step 4: 碎片合并
        room_map = self._merge_fragments(normal_map, small_map)

        # step 5: 转换成房间多边形列表
        room_polygons = self._convert_to_polygons(input_img, 
                                                  room_map,
                                                  threshold_result["threshold_list"])

        return room_polygons

    # ==================== Step 1: OBB → threshold 掩膜 ====================

    def _build_threshold_mask(
        self,
        obb_list: List[List[int]],
        input_img: np.ndarray,
        map_data: Dict[str, Any] = None,
    ) -> Dict[str, Any]:
        """
        将 OBB 列表转换为 threshold 线掩膜。

        若 map_data 中含有 tensor_scale / tensor_pad，则先将 OBB 顶点坐标从
        Triton 输入空间（letterboxed）逆映射回原始地图像素坐标，再画线。

        Args:
            obb_list: OBB 列表, 每个 OBB 为 4 个顶点 [[x,y], ...]
            input_img: 地图数据 (灰度图, >=200 为自由空间)
            map_data: (可选) 含 tensor_scale / tensor_pad / pre_pad 的字典
        """
        h, w = input_img.shape[:2]
        free_mask = input_img >= 200

        # 解析坐标逆映射参数
        scale    = (map_data or {}).get("tensor_scale", 1.0)
        pad_top, pad_left = (map_data or {}).get("tensor_pad", (0, 0))
        pre_pad_top, pre_pad_left = (map_data or {}).get("pre_pad", (0, 0))
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

    @staticmethod
    def _find_best_contour(cnts: List[np.ndarray]) -> np.ndarray:
        """
        合并多个不连通轮廓为覆盖所有非0像素的单一轮廓。

        当某个房间的掩膜存在多个不连通区域时，通过形态学闭运算
        逐步桥接间隙，直到所有区域合为一体。若形态学则返回最大面积的轮廓。

        Args:
            cnts: cv2.findContours 返回的轮廓列表

        Returns:
            覆盖所有输入轮廓区域的单一轮廓 (N, 1, 2) int32
        """
        if not cnts:
            return np.zeros((0, 1, 2), dtype=np.int32)
        if len(cnts) == 1:
            return cnts[0]

        all_pts = np.vstack(cnts)
        x, y, bw, bh = cv2.boundingRect(all_pts)

        mask = np.zeros((y + bh + 2, x + bw + 2), dtype=np.uint8)
        cv2.drawContours(mask, cnts, -1, 255, thickness=cv2.FILLED)

        for ksize in (3, 5, 7, 11, 15, 21, 31):
            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (ksize, ksize))
            closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            merged, _ = cv2.findContours(
                closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if len(merged) == 1:
                return merged[0]

        return max(cnts, key=cv2.contourArea)

    def _extract_polygons_from_label_map(self, label_map: np.ndarray) -> Dict[int, np.ndarray]:
        """从标签图中提取每个标签对应的外轮廓多边形。"""
        polygons: Dict[int, np.ndarray] = {}
        max_label = int(label_map.max())
        for lid in range(1, max_label + 1):
            mask = (label_map == lid).astype(np.uint8)
            if not mask.any():
                continue

            cnts, _ = cv2.findContours(
                mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not cnts:
                continue

            cnt = self._find_best_contour(cnts)
            polygons[lid] = cnt.reshape(-1, 2).astype(np.float64)
        return polygons

    @staticmethod
    def _draw_polygons_debug(
        input_img: np.ndarray,
        polygons: Dict[int, np.ndarray],
        palette: Dict[int, Tuple[int, int, int]],
        output_path: str,
    ) -> None:
        """绘制房间多边形及标签文本到 debug 图像。"""
        canvas = (cv2.cvtColor(input_img, cv2.COLOR_GRAY2BGR)
                  if input_img.ndim == 2 else input_img.copy())

        for lid, poly in polygons.items():
            pts = np.asarray(poly, dtype=np.float64).reshape(-1, 2).astype(np.int32)
            if pts.shape[0] == 0:
                continue
            color = palette.get(lid, (0, 255, 0))
            cv2.polylines(canvas, [pts], True, color, 1)
            cx, cy = pts.mean(axis=0).astype(int)
            cv2.putText(canvas, str(lid), (cx, cy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        cv2.imwrite(output_path, canvas)

    @staticmethod
    def _merge_fragments_to_polygons(
        normal_map: np.ndarray,
        frag_labels: np.ndarray,
    ) -> Tuple[np.ndarray, int, int]:
        """
        将碎片标签并入 normal_map 对应房间。

        策略：
            1) 先判断是否与 normal_map 直接连通；
            2) 若不连通，再渐进膨胀搜索最近正常区域；
            3) 候选多个时，按接触像素众数选择目标房间。
        """
        result = normal_map.copy()
        h, w = normal_map.shape[:2]
        kernel = np.ones((3, 3), np.uint8)
        max_search = int((h ** 2 + w ** 2) ** 0.5) + 1

        frag_ids = [int(fid) for fid in np.unique(frag_labels) if fid > 0]
        frag_ids.sort(key=lambda fid: -int((frag_labels == fid).sum()))

        merged_count = 0
        fallback_count = 0

        for fid in frag_ids:
            frag_mask = frag_labels == fid
            frag_u8 = frag_mask.astype(np.uint8)

            # 第一阶段：直接连通（1 像素邻接）检查
            dilated = cv2.dilate(frag_u8, kernel, iterations=1)
            neighbor_px = normal_map[(dilated > 0) & ~frag_mask & (normal_map > 0)]
            if neighbor_px.size > 0:
                best = int(np.bincount(neighbor_px).argmax())
                result[frag_mask] = best
                merged_count += 1
                continue

            # 第二阶段：无直接连通时，渐进搜索最近 normal_map 房间
            search_front = frag_u8.copy()
            best = 0
            for _ in range(max_search):
                search_front = cv2.dilate(search_front, kernel, iterations=1)
                frontier = (search_front > 0) & ~frag_mask
                neighbor_px = normal_map[frontier & (normal_map > 0)]
                if neighbor_px.size > 0:
                    best = int(np.bincount(neighbor_px).argmax())
                    break

            if best > 0:
                result[frag_mask] = best
                merged_count += 1
            else:
                result[frag_mask] = 0
                fallback_count += 1
                logger.warning("多边形碎片 %d 未找到邻居，置为背景", fid)

        return result, merged_count, fallback_count

    @staticmethod
    def _edge_key(p1: Tuple[int, int], p2: Tuple[int, int]) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        """无向边标准化 key。"""
        return (p1, p2) if p1 <= p2 else (p2, p1)

    @staticmethod
    def _trace_polygon_from_edges(edges: Set[Tuple[Tuple[int, int], Tuple[int, int]]]) -> np.ndarray:
        """
        从网格边集合中追踪主轮廓环（最大面积环）。
        返回 (N,2) float64；失败时返回空数组。
        """
        if not edges:
            return np.zeros((0, 2), dtype=np.float64)

        adjacency: Dict[Tuple[int, int], List[Tuple[int, int]]] = {}
        for a, b in edges:
            adjacency.setdefault(a, []).append(b)
            adjacency.setdefault(b, []).append(a)

        unused = set(edges)
        loops: List[np.ndarray] = []

        while unused:
            a, b = next(iter(unused))
            loop = [a, b]
            unused.remove((a, b))
            prev, curr = a, b

            while True:
                if curr == loop[0]:
                    break

                neighbors = adjacency.get(curr, [])
                next_pt = None
                for cand in neighbors:
                    ekey = Postprocessor._edge_key(curr, cand)
                    if ekey in unused and cand != prev:
                        next_pt = cand
                        break
                if next_pt is None:
                    for cand in neighbors:
                        ekey = Postprocessor._edge_key(curr, cand)
                        if ekey in unused:
                            next_pt = cand
                            break
                if next_pt is None:
                    break

                unused.remove(Postprocessor._edge_key(curr, next_pt))
                loop.append(next_pt)
                prev, curr = curr, next_pt

            if len(loop) >= 4 and loop[0] == loop[-1]:
                arr = np.asarray(loop[:-1], dtype=np.float64)
                loops.append(arr)

        if not loops:
            return np.zeros((0, 2), dtype=np.float64)

        def polygon_area(poly: np.ndarray) -> float:
            x = poly[:, 0]
            y = poly[:, 1]
            return 0.5 * float(np.abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))

        return max(loops, key=polygon_area)

    def _extract_shared_edge_polygons_from_label_map(
        self,
        label_map: np.ndarray,
    ) -> Dict[int, np.ndarray]:
        """
        基于统一网格边界重建多边形：
        相邻房间会共享同一组边段，从而实现严格共边界。
        """
        h, w = label_map.shape[:2]
        labels = [int(lid) for lid in np.unique(label_map) if lid > 0]
        edges_by_label: Dict[int, Set[Tuple[Tuple[int, int], Tuple[int, int]]]] = {
            lid: set() for lid in labels
        }

        padded = np.pad(label_map, ((1, 1), (1, 1)), mode="constant", constant_values=0)
        for y in range(1, h + 1):
            for x in range(1, w + 1):
                lid = int(padded[y, x])
                if lid <= 0:
                    continue

                # 当前像素对应角点坐标范围: [x-1, x] x [y-1, y]
                if int(padded[y, x - 1]) != lid:
                    p1, p2 = (x - 1, y - 1), (x - 1, y)
                    edges_by_label[lid].add(self._edge_key(p1, p2))
                if int(padded[y, x + 1]) != lid:
                    p1, p2 = (x, y - 1), (x, y)
                    edges_by_label[lid].add(self._edge_key(p1, p2))
                if int(padded[y - 1, x]) != lid:
                    p1, p2 = (x - 1, y - 1), (x, y - 1)
                    edges_by_label[lid].add(self._edge_key(p1, p2))
                if int(padded[y + 1, x]) != lid:
                    p1, p2 = (x - 1, y), (x, y)
                    edges_by_label[lid].add(self._edge_key(p1, p2))

        polygons: Dict[int, np.ndarray] = {}
        for lid, edges in edges_by_label.items():
            poly = self._trace_polygon_from_edges(edges)
            if poly.size > 0:
                polygons[lid] = poly
        return polygons

    def _align_polygons_shared_boundary(
        self,
        label_map: np.ndarray,
        polygons: Dict[int, np.ndarray],
    ) -> Dict[int, np.ndarray]:
        """Step5: 以统一边界网格重建，确保相邻房间严格共边界。"""
        if not polygons:
            return polygons
        return self._extract_shared_edge_polygons_from_label_map(label_map)

    def _convert_to_polygons(
        self,
        input_img: np.ndarray,
        room_map: np.ndarray,
        threshold_list: List[List[float]] = None,
    ) -> List[List[List[float]]]:
        """
        将房间标签图转换成房间多边形列表，根据分割线列表判断该分割线位于哪两个
        房间之间，并把分割线加入到对应的房间多边形列表中，使其共用一条分割线。
        最后返回处理后的房间多边形列表，详细绘制处理前后房间多边形列表的图像进行 debug。

        Args:
            input_img: 原始地图 (H, W) uint8
            room_map:  房间标签图 (H, W) int32, 0=背景, >0=房间
            threshold_list: 分割线端点列表 [[(x1,y1), (x2,y2)], ...]

        Returns:
            按标签升序排列的房间多边形列表 [[[x, y], ...], ...]
        """
        _ = threshold_list

        h, w = room_map.shape[:2]
        max_label = int(room_map.max())

        # ---- debug: 保存输入图 ----
        # cv2.imwrite("./dataset/debug/1_input_img.png", input_img)
        # if max_label > 0:
        #     label_vis = (room_map * 255 // max_label).astype(np.uint8)
        # else:
        #     label_vis = np.zeros_like(room_map, dtype=np.uint8)
        # cv2.imwrite("./dataset/debug/2_room_map.png", label_vis)

        if max_label == 0:
            logger.warning("room_map 无有效房间标签, 返回空列表")
            return []

        # ---- step 1: 提取每个房间的轮廓多边形 ----
        logger.info(f"start extract room polygons")
        logger.info(f"max_label: {max_label}")
        room_polygons = self._extract_polygons_from_label_map(room_map)

        # ---- debug: 绘制修改前的多边形 ----
        # palette = self._make_palette(max_label)
        # self._draw_polygons_debug(
        #     input_img,
        #     room_polygons,
        #     palette,
        #     "./dataset/debug/3_polygons_before.png",
        # )

        # ---- step 2: 构建 normal_map（多边形覆盖）和碎片标签 ----
        normal_map = np.zeros((h, w), dtype=np.int32)
        for lid, polygon in room_polygons.items():
            pts = polygon.astype(np.int32).reshape(-1, 1, 2)
            cv2.drawContours(normal_map, [pts], -1, int(lid),
                             thickness=cv2.FILLED)

        fragment_mask = (input_img > 127) & (normal_map == 0)
        if fragment_mask.any():
            _, frag_labels = cv2.connectedComponents(
                fragment_mask.astype(np.uint8), connectivity=8)
            frag_labels = frag_labels.astype(np.int32)
        else:
            frag_labels = np.zeros((h, w), dtype=np.int32)


        # ---- step 3: 先直接连通，后最近邻搜索，合并碎片 ----
        result, merged_count, fallback_count = self._merge_fragments_to_polygons(
            normal_map, frag_labels)

        frag_count = int(np.unique(frag_labels).size - (1 if (frag_labels == 0).any() else 0))
        logger.info("多边形碎片合并: %d 个碎片 (合并 %d, 置背景 %d)",
                    frag_count, merged_count, fallback_count)

        # ---- step 4: 由合并后的标签图重新提取最终多边形 ----
        final_polygons = self._extract_polygons_from_label_map(result)
        # ---- debug: 绘制修改后的多边形 ----
        # self._draw_polygons_debug(
        #     input_img,
        #     final_polygons,
        #     palette,
        #     "./dataset/debug/4_polygons_after.png",
        # )
        
        
        # ---- step5: 相邻房间共边界对齐 ----
        final_polygons = self._align_polygons_shared_boundary(result, final_polygons)

        # ---- debug: 绘制修改后的多边形 ----
        # self._draw_polygons_debug(
        #     input_img,
        #     final_polygons,
        #     palette,
        #     "./dataset/debug/5_polygons_after.png",
        # )

        # ---- step6 : 外扩房间多边形 ----
        for lid, polygon in final_polygons.items():
            expanded = expand_one(polygon, input_img)
            final_polygons[lid] = np.asarray(expanded, dtype=np.float64).reshape(-1, 2)
        

        # ---- debug: 绘制修改后的多边形 ----
        # self._draw_polygons_debug(
        #     input_img,
        #     final_polygons,
        #     palette,
        #     "./dataset/debug/6_polygons_after.png",
        # )

        logger.info("多边形转换完成: %d 个房间", len(final_polygons))
        return [final_polygons[lid].tolist()
                for lid in sorted(final_polygons.keys())]

        


    @staticmethod
    def _make_palette(n: int) -> Dict[int, Tuple[int, int, int]]:
        """生成 n 种 HSV 等距分布的可视化颜色"""
        palette: Dict[int, Tuple[int, int, int]] = {}
        for i in range(1, n + 1):
            hue = int(180 * (i - 1) / max(n, 1))
            bgr = cv2.cvtColor(
                np.array([[[hue, 200, 230]]], dtype=np.uint8),
                cv2.COLOR_HSV2BGR,
            )
            palette[i] = tuple(int(c) for c in bgr[0, 0])
        return palette


