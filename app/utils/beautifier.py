"""轮廓美化模块 —— 将不规则轮廓简化为轴对齐多边形 + 门槛线检测"""

import collections
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
import cv2


class ContourBeautifier:
    """
    地图美化 (s10 机型)

    功能:
    - 将不规则房间轮廓简化为轴对齐 (水平/垂直) 的多边形
    - 检测房间边界上的可通行区域 (门槛线)
    - 处理相邻美化框的重叠
    """

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}

    # ==================== 主流程 ====================

    def beautify(self, contours: List[np.ndarray],
                 map_img: np.ndarray
                 ) -> Tuple[List[List], List[List]]:
        """
        完整美化流程

        Args:
            contours: 房间轮廓列表 [(N,1,2), ...]
            map_img: 原始地图 (BGR)

        Returns:
            (all_bbox_list, all_threshold_list)
            - all_bbox_list: 每个房间的美化框顶点 [[[x,y], ...], ...]
            - all_threshold_list: 每个房间的门槛线 [[[[x1,y1],[x2,y2]], ...], ...]
        """
        if map_img.ndim == 3:
            gray = cv2.cvtColor(map_img.copy(), cv2.COLOR_BGR2GRAY)
        else:
            gray = map_img.copy()
        # 墙壁标为127方便后续判断
        gray[gray == 0] = 127

        n = len(contours)
        all_bbox = [None] * n

        for i, cnt in enumerate(contours):
            # 在灰度图上标记轮廓边界
            cv2.drawContours(gray, [cnt], -1, 127, 1)
            origin_pts = cnt.reshape(-1, 2).tolist()

            # 优化 → 轴对齐 → 合并
            opt1 = self._optimize_contour(origin_pts, gray)
            opt2 = self._optimize_contour(opt1, gray)
            opt2 = self._remove_collinear(opt2)
            aligned = self._adjust_to_axis_aligned(opt2, gray)
            merged = self._merge_lines(aligned, gray)
            all_bbox[i] = merged if merged else opt2

        # 解决相邻框重叠
        self._resolve_overlaps(all_bbox, gray)

        # 去冗余点
        for i in range(len(all_bbox)):
            if all_bbox[i]:
                all_bbox[i] = self._remove_collinear(all_bbox[i])

        # 门槛线检测
        all_bbox, all_threshold = self._detect_thresholds(all_bbox, map_img)

        return all_bbox, all_threshold

    # ==================== 轮廓优化 ====================

    def _optimize_contour(self, pts: List[List[int]],
                          gray: np.ndarray) -> List[List[int]]:
        """去除穿过墙壁的边，只保留不穿墙的顶点"""
        mask = np.zeros_like(gray, dtype=np.uint8)
        result = [pts[0]]
        for i in range(1, len(pts) + 1):
            p1 = result[-1]
            p2 = pts[i % len(pts)]
            p_prev = pts[(i - 1) % len(pts)]
            mask.fill(0)
            cv2.line(mask, tuple(p1), tuple(p2), 255, 1)
            if np.all(gray[mask == 255] != 255):
                continue
            result.append(p_prev)
        return result

    def _adjust_to_axis_aligned(self, pts: List[List[int]],
                                gray: np.ndarray) -> List[List[int]]:
        """将斜线段调整为水平/垂直组合"""
        if len(pts) < 2:
            return pts

        result = [pts[0]]
        for i in range(1, len(pts) + 1):
            p1 = result[-1]
            p2 = pts[i % len(pts)]

            if p1[0] == p2[0] or p1[1] == p2[1]:
                result.append(p2)
                continue

            x1, y1 = p1
            x2, y2 = p2

            # 尝试对角路径: 先水平再垂直 或 先垂直再水平
            mid1 = [x2, y1]
            mid2 = [x1, y2]

            if self._path_clear(gray, p1, mid1, p2):
                result.extend([mid1, p2])
            elif self._path_clear(gray, p1, mid2, p2):
                result.extend([mid2, p2])
            else:
                # BFS 找轴对齐路径
                path = self._bfs_axis_path(gray, p1, p2)
                if path:
                    result.extend(path)
                else:
                    result.append(p2)

        return result

    def _merge_lines(self, pts: List[List[int]],
                     gray: np.ndarray) -> Optional[List[List[int]]]:
        """合并优化：用填充+轮廓提取去除内缩点"""
        mask = np.zeros(gray.shape, dtype=np.uint8)
        arr = np.array(pts, dtype=np.int32).reshape(-1, 1, 2)
        cv2.drawContours(mask, [arr], -1, 255, -1)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        largest = max(contours, key=cv2.contourArea)
        result = self._adjust_to_axis_aligned(
            largest.reshape(-1, 2).tolist(), gray
        )
        result = self._remove_collinear(result)

        # 迭代优化
        for _ in range(3):
            result = self._optimize_contour(
                self._optimize_contour(result, gray), gray
            )
            result = self._remove_collinear(result)
            result = self._adjust_to_axis_aligned(result, gray)
            result = self._remove_collinear(result)

        return result

    # ==================== 重叠处理 ====================

    def _resolve_overlaps(self, all_bbox: List, gray: np.ndarray):
        """解决相邻美化框的重叠问题"""
        h, w = gray.shape
        for i in range(len(all_bbox)):
            for j in range(len(all_bbox)):
                if i == j or not all_bbox[i] or not all_bbox[j]:
                    continue

                arr_i = np.array(all_bbox[i], dtype=np.int32).reshape(-1, 1, 2)
                arr_j = np.array(all_bbox[j], dtype=np.int32).reshape(-1, 1, 2)

                # 检查是否重叠
                mask = np.zeros((h, w), dtype=np.uint8)
                cv2.drawContours(mask, [arr_i, arr_j], -1, 255, -1)
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                               cv2.CHAIN_APPROX_SIMPLE)
                if len(contours) != 1:
                    continue

                # 从 j 中减去 i 的区域
                mask_i = np.zeros((h, w), dtype=np.uint8)
                mask_j = np.zeros((h, w), dtype=np.uint8)
                cv2.drawContours(mask_i, [arr_i], -1, 255, 1)
                cv2.drawContours(mask_j, [arr_j], -1, 255, -1)

                overlap = np.zeros((h, w), dtype=np.uint8)
                overlap[(mask_i == 255) & (mask_j == 255)] = 255
                mask_j2 = mask_j.copy()
                cv2.drawContours(mask_j2, [arr_i], -1, 0, -1)
                mask_j2[overlap == 255] = 255

                contours_j, _ = cv2.findContours(mask_j2, cv2.RETR_EXTERNAL,
                                                 cv2.CHAIN_APPROX_SIMPLE)
                if contours_j:
                    largest = max(contours_j, key=cv2.contourArea)
                    result = self._adjust_to_axis_aligned(
                        largest.reshape(-1, 2).tolist(), gray
                    )
                    all_bbox[j] = self._remove_collinear(result)

    # ==================== 门槛线检测 ====================

    def _detect_thresholds(self, all_bbox: List,
                           map_img: np.ndarray
                           ) -> Tuple[List, List]:
        """
        检测每个美化框边界上的可通行区域 (门槛线)

        门槛线 = 美化框边界上像素值为 255 (空闲空间) 的连续线段
        """
        if map_img.ndim == 3:
            gray = cv2.cvtColor(map_img, cv2.COLOR_BGR2GRAY)
        else:
            gray = map_img.copy()
        gray[gray == 0] = 127

        h, w = gray.shape
        mask = np.zeros((h, w), dtype=np.uint8)
        all_threshold = [[] for _ in range(len(all_bbox))]

        # 去除单像素门槛噪点
        for i, bbox in enumerate(all_bbox):
            if not bbox:
                continue
            mask.fill(0)
            arr = np.array(bbox, dtype=np.int32).reshape(-1, 1, 2)
            cv2.drawContours(mask, [arr], -1, 255, 1)
            mask[(mask == 255) & (gray == 255)] = 0
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                unique_pts = np.unique(cnt, axis=0)
                if len(unique_pts) == 1:
                    cv2.drawContours(gray, [cnt], -1, 255, 1)

        # 提取门槛线段
        for i, bbox in enumerate(all_bbox):
            if not bbox:
                continue
            mask.fill(0)
            arr = np.array(bbox, dtype=np.int32).reshape(-1, 1, 2)
            cv2.drawContours(mask, [arr], -1, 255, 1)

            # 找空闲空间交叉点
            valid_y, valid_x = np.where((mask == 255) & (gray == 255))
            mask.fill(0)
            mask[valid_y, valid_x] = 255

            threshold_list = []
            contours, _ = cv2.findContours(mask, cv2.RETR_LIST,
                                           cv2.CHAIN_APPROX_NONE)
            for cnt in contours:
                pts = cnt.reshape(-1, 2)
                # 简化为线段序列
                segments = self._extract_segments(pts)
                threshold_list.extend(segments)

            # 合并相近的门槛线段
            threshold_list = self._merge_nearby_thresholds(threshold_list)
            all_threshold[i] = threshold_list

        return all_bbox, all_threshold

    def _extract_segments(self, pts: np.ndarray) -> List[List]:
        """从连续点序列中提取轴对齐线段"""
        if len(pts) < 2:
            return []

        # 去重保序
        unique_pts = list(dict.fromkeys(map(tuple, pts)))
        cleaned = self._remove_collinear([list(p) for p in unique_pts])

        segments = []
        for i in range(len(cleaned) - 1):
            segments.append([cleaned[i], cleaned[i + 1]])
        return segments

    def _merge_nearby_thresholds(self, thresholds: List[List],
                                 gap: int = 3) -> List[List]:
        """合并共线且间距小于 gap 的门槛线段"""
        if len(thresholds) < 2:
            return thresholds

        merged = list(thresholds)
        changed = True
        while changed:
            changed = False
            for i in range(len(merged)):
                for j in range(i + 1, len(merged)):
                    if not merged[i] or not merged[j]:
                        continue
                    seg_a, seg_b = merged[i], merged[j]
                    result = self._try_merge_segments(seg_a, seg_b, gap)
                    if result:
                        merged[i] = result
                        merged[j] = None
                        changed = True

        return [s for s in merged if s]

    def _try_merge_segments(self, seg_a, seg_b, gap):
        """尝试合并两个共线且接近的线段"""
        (x1, y1), (x2, y2) = seg_a
        (x3, y3), (x4, y4) = seg_b

        # 水平共线
        if y1 == y2 == y3 == y4:
            lo_a, hi_a = min(x1, x2), max(x1, x2)
            lo_b, hi_b = min(x3, x4), max(x3, x4)
            if abs(hi_a - lo_b) <= gap or abs(hi_b - lo_a) <= gap:
                new_lo = min(lo_a, lo_b)
                new_hi = max(hi_a, hi_b)
                return [[new_lo, y1], [new_hi, y1]]

        # 垂直共线
        if x1 == x2 == x3 == x4:
            lo_a, hi_a = min(y1, y2), max(y1, y2)
            lo_b, hi_b = min(y3, y4), max(y3, y4)
            if abs(hi_a - lo_b) <= gap or abs(hi_b - lo_a) <= gap:
                new_lo = min(lo_a, lo_b)
                new_hi = max(hi_a, hi_b)
                return [[x1, new_lo], [x1, new_hi]]

        return None

    # ==================== 工具方法 ====================

    def _path_clear(self, gray: np.ndarray,
                    p1, mid, p2) -> bool:
        """检查 p1→mid→p2 路径上是否都没有墙壁 (255)"""
        mask = np.zeros(gray.shape, dtype=np.uint8)
        cv2.line(mask, tuple(p1), tuple(mid), 255, 1)
        cv2.line(mask, tuple(mid), tuple(p2), 255, 1)
        if gray[tuple(mid)[::-1]] == 255:
            return False
        return np.all(gray[mask == 255] != 255)

    def _bfs_axis_path(self, gray: np.ndarray,
                       p1: List[int], p2: List[int]) -> Optional[List]:
        """BFS 寻找轴对齐路径 (只走水平/垂直)"""
        import heapq

        x1, y1 = p1
        x2, y2 = p2
        min_x, max_x = min(x1, x2), max(x1, x2)
        min_y, max_y = min(y1, y2), max(y1, y2)
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        dir_names = ["l", "r", "u", "d"]

        pq = [(0, x1, y1, [], None)]
        visited = set()

        while pq:
            turns, cx, cy, path, cur_dir = heapq.heappop(pq)
            if (cx, cy) in visited:
                continue
            visited.add((cx, cy))

            if (cx, cy) == (x2, y2):
                return path + [[x2, y2]]

            for i, (dx, dy) in enumerate(directions):
                nx, ny = cx + dx, cy + dy
                if min_x <= nx <= max_x and min_y <= ny <= max_y:
                    if (nx, ny) not in visited and gray[ny, nx] != 255:
                        new_turns = turns + (1 if cur_dir and dir_names[i] != cur_dir else 0)
                        heapq.heappush(pq, (new_turns, nx, ny,
                                            path + [[cx, cy]], dir_names[i]))
        return None

    @staticmethod
    def _remove_collinear(pts: List[List[int]]) -> List[List[int]]:
        """去除共线的中间点"""
        if len(pts) < 3:
            return pts

        # 先去重
        unique = []
        for p in pts:
            if not unique or p != unique[-1]:
                unique.append(p)

        result = [unique[0]]
        for i in range(1, len(unique) - 1):
            p0, p1, p2 = unique[i - 1], unique[i], unique[i + 1]
            cross = (p1[0] - p0[0]) * (p2[1] - p0[1]) - \
                    (p1[1] - p0[1]) * (p2[0] - p0[0])
            if cross != 0:
                result.append(p1)
        result.append(unique[-1])
        return result
