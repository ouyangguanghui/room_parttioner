"""扩展分区模块

在已有 labels 的基础上进行扩展分区：
  1. 检测新增自由区域（地图更新后出现的未标记空闲像素）
  2. 判断新区域归属：合并入已有房间 or 创建新房间
  3. 门口检测拆分 + 区域生长填充
  4. 重新序列化，保留已有房间元数据

像素级底层操作：
  - split_by_doorway: 通过门口检测拆分大区域
  - grow_unassigned: 区域生长填充未分配像素
  - extend_pixel: 完整像素级流程 (门口拆分 + 区域生长)

process() 入口签名与 AutoPartitioner / ManualPartitioner 保持一致:
  process(map_data, transformer, graph_builder, landmark_builder) -> labels_json
"""

import logging
from typing import Dict, Any, List, Tuple

import numpy as np
import cv2

from app.services.base_partitioner import BasePartitioner
from app.utils.coordinate import CoordinateTransformer
from app.utils.graph import RoomGraph
from app.utils.landmark import LandmarkManager
from app.utils.labels_ops import ContourExpander
from app.utils.geometry_ops import (
    split_labels_data,
    next_room_id,
    next_room_name,
)

logger = logging.getLogger(f"{__name__} [ExtendedPartitioner]")



class ExtendedPartitioner(BasePartitioner):
    """
    扩展分区：在已有 labels 基础上检测新增自由区域并分配归属。

    功能：
    - 检测新增自由区域 (detect_new_regions)
    - 判断新区域归属策略：合并 or 新房间 (classify_region)
    - 门口检测拆分大区域 (split_by_doorway)
    - 区域生长扩展未分配像素 (grow_unassigned)
    - 完整 process() 流程，保留已有房间元数据
    """

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.door_width = self.config.get("door_width", 20)
        self.grow_iterations = self.config.get("grow_iterations", 10)
        self.wall_threshold = self.config.get("wall_threshold", 128)
        self.resolution = self.config.get("resolution", 0.05)
        self.min_room_area = self.config.get("min_room_area", 1.0)  # m²
        self.min_new_region_area = self.config.get("min_new_region_area", 50)  # pixels
        self.merge_area_threshold = self.config.get("merge_area_threshold", 0)  # m², 0=use min_room_area
        self.merge_ratio_threshold = self.config.get("merge_ratio_threshold", 0.6)

    # ==================== 像素级底层操作 ====================
    def _region_grow(self, mask: np.ndarray, seeds: np.ndarray,
                     num_labels: int) -> np.ndarray:
        """从种子标签向外生长，填满 mask 区域"""
        result = seeds.copy()
        for _ in range(self.grow_iterations):
            claimed = np.zeros_like(result, dtype=np.int32)
            for lid in range(1, num_labels):
                seed_mask = (result == lid).astype(np.uint8)
                dilated = cv2.dilate(seed_mask, np.ones((3, 3), np.uint8))
                fill = (dilated > 0) & (result == 0) & (mask > 0)
                if not fill.any():
                    continue
                take = fill & ((claimed == 0) | (lid < claimed))
                claimed[take] = lid
            if not claimed.any():
                break
            result[claimed > 0] = claimed[claimed > 0]
        return result

    # ==================== 新区域检测 ====================

    def detect_new_regions(self, label_map: np.ndarray,
                           grid_map: np.ndarray) -> np.ndarray:
        """
        检测新增自由区域

        对比已有 label_map 与当前 grid_map，找出未被标记但属于自由空间的区域。
        用连通域分析分离独立区域块，过滤面积过小的碎片。

        Args:
            label_map: 已有房间标签 (H, W) int32, 0=未分配
            grid_map: 当前地图 (H, W) uint8

        Returns:
            new_regions_map: (H, W) int32, 每个新区域一个独立 label (从 1 开始), 0=非新区域
        """

        free_space = grid_map >= self.wall_threshold
        unassigned = (label_map == 0) & free_space  
        unassigned_u8 = unassigned.astype(np.uint8)

        if not unassigned.any():
            return np.zeros_like(label_map)

        num_labels, regions = cv2.connectedComponents(unassigned_u8, connectivity=8)

        result = np.zeros_like(label_map)
        new_id = 0
        for rid in range(1, num_labels):
            region_mask = regions == rid
            new_id += 1
            result[region_mask] = new_id

        return result

    def classify_region(self, region_mask: np.ndarray,
                        label_map: np.ndarray) -> Tuple[str, int]:
        """
        对单个新区域判断归属策略

        计算新区域与每个已有房间的接触边长度（共享边界像素数），
        结合面积和接触比例决定是合并还是创建新房间。

        Args:
            region_mask: (H, W) bool, 新区域的像素 mask
            label_map: (H, W) int32, 已有房间标签

        Returns:
            ("merge", target_room_label) 或 ("new", 0)
        """
        # 计算新区域面积 (m²)
        pixel_count = int(region_mask.sum())
        area_m2 = pixel_count * (self.resolution ** 2)

        # 膨胀新区域 1 像素，找与已有房间的接触边
        region_u8 = region_mask.astype(np.uint8)
        dilated = cv2.dilate(region_u8, np.ones((3, 3), np.uint8))
        border = (dilated > 0) & (~region_mask)

        # 统计接触边上各房间的像素数
        contact_counts: Dict[int, int] = {}
        border_labels = label_map[border]
        for lid in np.unique(border_labels):
            if lid == 0:
                continue
            contact_counts[int(lid)] = int((border_labels == lid).sum())

        if not contact_counts:
            # 无相邻房间 → 新房间
            return ("new", 0)

        # 最大接触房间
        best_room = max(contact_counts, key=contact_counts.get)  # type: ignore
        best_contact = contact_counts[best_room]

        # 计算新区域周长（边界像素数）
        contours, _ = cv2.findContours(region_u8, cv2.RETR_EXTERNAL,
                                        cv2.CHAIN_APPROX_NONE)
        perimeter = sum(len(c) for c in contours) if contours else 1

        # 决策逻辑
        area_threshold = self.merge_area_threshold if self.merge_area_threshold > 0 else self.min_room_area

        # 小区域 → 合并
        if area_m2 < area_threshold:
            return ("merge", best_room)

        # 接触比例高 → 合并
        contact_ratio = best_contact / max(perimeter, 1)
        if contact_ratio > self.merge_ratio_threshold:
            return ("merge", best_room)

        # 否则 → 新房间
        return ("new", 0)

    # ==================== threshold 线分界 ====================

    def _filter_thresholds(
        self,
        threshold_list: List,
        old_label_map: np.ndarray,
        gray_map: np.ndarray,
        threshold_thickness: int = 10,
    ) -> np.ndarray:
        """
        过滤 threshold 线：只保留至少一个端点不在旧房间内的线。

        两端都在旧房间内的线会切割旧房间，应跳过。

        Args:
            threshold_list: [[(x1,y1), (x2,y2)], ...]
            old_label_map: 旧房间标签图 (H, W) int32

        Returns:
            filtered_mask: (H, W) uint8, 255=threshold 线像素
        """
        h, w = old_label_map.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)

        for p1, p2 in threshold_list:
            x1 = int(np.clip(round(p1[0]), 0, w - 1))
            y1 = int(np.clip(round(p1[1]), 0, h - 1))
            x2 = int(np.clip(round(p2[0]), 0, w - 1))
            y2 = int(np.clip(round(p2[1]), 0, h - 1))

            # 先沿线段方向两端各延伸 2px，再以该中心线绘制线段矩形（非齐次矩形）
            dx, dy = x2 - x1, y2 - y1
            length = np.hypot(dx, dy)
            if length == 0:
                # 退化为点
                continue

            dir_x = dx / length
            dir_y = dy / length
            ext_x1 = x1 - dir_x * 2
            ext_y1 = y1 - dir_y * 2
            ext_x2 = x2 + dir_x * 2
            ext_y2 = y2 + dir_y * 2

            wx, wy = -dy / length, dx / length  # 单位法向（左侧为正）
            # 矩形4点
            p1a = (int(round(ext_x1 + wx * threshold_thickness / 2)), int(round(ext_y1 + wy * threshold_thickness / 2)))
            p1b = (int(round(ext_x1 - wx * threshold_thickness / 2)), int(round(ext_y1 - wy * threshold_thickness / 2)))
            p2a = (int(round(ext_x2 + wx * threshold_thickness / 2)), int(round(ext_y2 + wy * threshold_thickness / 2)))
            p2b = (int(round(ext_x2 - wx * threshold_thickness / 2)), int(round(ext_y2 - wy * threshold_thickness / 2)))
            poly = np.array([p1a, p2a, p2b, p1b], dtype=np.int32)
            cv2.fillPoly(mask, [poly], 255)
    
        filtered_mask = np.zeros((h, w), dtype=np.uint8)
        filtered_mask[(mask == 255) & (old_label_map == 0)] = 255
        filtered_mask[(filtered_mask == 255) & (gray_map != 255)] = 0

        return filtered_mask

    def _classify_new_regions(
        self,
        old_label_map: np.ndarray,
        old_contours: List[np.ndarray],
        grid_map: np.ndarray,
        filtered_mask: np.ndarray,
    ) -> Tuple[List[np.ndarray], List[int]]:
        """
        用 threshold 线切割新区域，决定合并或新房间。
        """
        # 1. 找出所有新区域（connected components）
        free_space = grid_map >= self.wall_threshold
        # 新区域 = 自由空间 & 不在旧房间 & 不在 threshold 线上
        split_mask = free_space & (old_label_map == 0)
        split_mask[filtered_mask == 255] = 0

        # 2. 找出所有新区域的外轮廓
        new_contours, _ = cv2.findContours(split_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        if len(new_contours) == 0:
            return old_contours, []

        # 2. 连通域分块，逐个新区域做归属判定
        result = old_contours.copy()
        room_id_list = []
        h, w = grid_map.shape[:2]
        new_region_mask = np.zeros((h, w), dtype=np.uint8)
        old_region_mask = np.zeros((h, w), dtype=np.uint8)
        new_id_dict = {}
        kernel = np.ones((3, 3), np.uint8)
        add_new_room_idx = set()
        merged_to_old_new_idx = set()
        for new_idx, new_contour in enumerate(new_contours):
            merge_old_idx_dict = {}
            # 只和旧房间索引范围做归属判断，但轮廓使用 result 中的最新形状
            for old_idx in range(len(old_contours)):
                old_contour = result[old_idx]
                old_region_mask.fill(0)
                new_region_mask.fill(0)
                cv2.drawContours(new_region_mask, [old_contour], -1, 255, -1)
                new_region_mask[~free_space] = 0
                cv2.drawContours(new_region_mask, [new_contour], -1, 255, -1)
                merge_cnts, _ = cv2.findContours(new_region_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                if len(merge_cnts) == 1:
                    cv2.drawContours(new_region_mask, [old_contour], -1, 255, -1)
                    merge_cnts, _ = cv2.findContours(new_region_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                    new_region_mask.fill(0)
                    old_region_mask.fill(0)
                    cv2.drawContours(new_region_mask, [new_contour], -1, 255, -1)
                    cv2.drawContours(old_region_mask, [old_contour], -1, 255, -1)
                    dilated_new = cv2.dilate(new_region_mask, kernel)
                    overlap = (dilated_new > 0) & (old_region_mask > 0)
                    contact_count = int(overlap.sum())
                    merge_old_idx_dict[old_idx] = (contact_count, merge_cnts[0])

            # 判断是否符合新房间条件
            if merge_old_idx_dict:
               # 取出contact_count最大的old_idx
               best_old_idx = max(merge_old_idx_dict.items(), key=lambda item: item[1][0])[0]
               result[best_old_idx] = merge_old_idx_dict[best_old_idx][1]
               if best_old_idx not in room_id_list:
                   room_id_list.append(best_old_idx)
               merged_to_old_new_idx.add(new_idx)
               logger.info(f"新区域 {new_idx} 合并到旧房间 {best_old_idx}")
            else:
                # 符合新房间条件， 现在像素坐标cv2.contourArea需要resolution转换为面积
                if cv2.contourArea(new_contour) * self.resolution ** 2 > self.min_room_area: 
                    result.append(new_contour)
                    new_id_dict[new_idx] = len(result) - 1 
                    if (len(result) - 1) not in room_id_list:
                        room_id_list.append(len(result) - 1)
                    add_new_room_idx.add(new_idx)
                    logger.info(f"新区域 {new_idx} 创建新房间")
                else:
                    logger.info(f"新区域 {new_idx} 面积太小，{cv2.contourArea(new_contour) * self.resolution ** 2} < {self.min_room_area}，不创建新房间")

        
        # 把门槛像素合并到新房间
        threshold_contours, _ = cv2.findContours(filtered_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        add_threshold_idx = set()
        for new_idx, idx in new_id_dict.items():
            if new_idx not in add_new_room_idx:
                continue
            new_contour = result[idx]
            merge_threshold_dict = {}
            for threshold_idx, threshold_contour in enumerate(threshold_contours):
                new_region_mask.fill(0)
                cv2.drawContours(new_region_mask, [new_contour], -1, 255, -1)
                cv2.drawContours(new_region_mask, [threshold_contour], -1, 255, -1)
                merge_cnts, _ = cv2.findContours(new_region_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                if len(merge_cnts) == 1:
                    merge_threshold_dict[threshold_idx] = merge_cnts[0]
                    new_region_mask.fill(0)
                    old_region_mask.fill(0)
                    cv2.drawContours(new_region_mask, [new_contour], -1, 255, -1)
                    cv2.drawContours(old_region_mask, [threshold_contour], -1, 255, -1)
                    dilated_new = cv2.dilate(new_region_mask, kernel)
                    overlap = (dilated_new > 0) & (old_region_mask > 0)
                    contact_count = int(overlap.sum())
                    merge_threshold_dict[threshold_idx] = [contact_count, merge_cnts[0]]
            
            if merge_threshold_dict:
                best_threshold_idx = max(merge_threshold_dict.items(), key=lambda item: item[1][0])[0]
                result[idx] = merge_threshold_dict[best_threshold_idx][1]
                add_threshold_idx.add(best_threshold_idx)
        
        # 收集仍未分配的区域：未并入旧房间/未创建新房间的新区域 + 未被并入新房间的 threshold 区域
        new_region_mask.fill(0)
        for new_idx, new_contour in enumerate(new_contours):
            if new_idx in merged_to_old_new_idx or new_idx in add_new_room_idx:
                continue
            cv2.drawContours(new_region_mask, [new_contour], -1, 255, -1)

        for threshold_idx, threshold_contour in enumerate(threshold_contours):
            if threshold_idx in add_threshold_idx:
                continue
            cv2.drawContours(new_region_mask, [threshold_contour], -1, 255, -1)

        small_contours, _ = cv2.findContours(new_region_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        logger.info(f"samll contours 数量: {len(small_contours)}")

        def _nearest_distance_with_points(
            contour_a: np.ndarray, contour_b: np.ndarray
        ) -> Tuple[float, np.ndarray, np.ndarray]:
            pts_a = contour_a.reshape(-1, 2).astype(np.float32)
            pts_b = contour_b.reshape(-1, 2).astype(np.float32)
            if len(pts_a) == 0 or len(pts_b) == 0:
                # 异常兜底：返回极大距离，点取原轮廓首点
                pa = contour_a.reshape(-1, 2)[0]
                pb = contour_b.reshape(-1, 2)[0]
                return float("inf"), pa, pb

            diff = pts_a[:, None, :] - pts_b[None, :, :]
            dist2 = np.sum(diff * diff, axis=2)
            flat_min_idx = int(np.argmin(dist2))
            idx_a, idx_b = np.unravel_index(flat_min_idx, dist2.shape)
            pa = pts_a[idx_a].astype(np.int32)
            pb = pts_b[idx_b].astype(np.int32)
            return float(np.sqrt(dist2[idx_a, idx_b])), pa, pb

        for small_idx, small_contour in enumerate(small_contours):
            small_area = cv2.contourArea(small_contour) * self.resolution ** 2
            if small_area > self.min_room_area:
                result.append(small_contour)
                if (len(result) - 1) not in room_id_list:
                    room_id_list.append(len(result) - 1)
                logger.info(f"samll contours 面积: {small_area}")
                logger.info(f"samll contours 新区域 {small_idx} 创建新房间")
            else:
                if not result:
                    result.append(small_contour)
                    logger.info(f"新区域 {small_idx} 太小但无可合并区域，创建兜底新房间")
                    continue

                # 优先合并到连通区域：若多个连通，取相邻区域中面积最大的
                connected_candidates = []
                new_region_mask.fill(0)
                cv2.drawContours(new_region_mask, [small_contour], -1, 255, -1)
                dilated_small = cv2.dilate(new_region_mask, kernel)
                for candidate_idx, candidate_contour in enumerate(result):
                    old_region_mask.fill(0)
                    cv2.drawContours(old_region_mask, [candidate_contour], -1, 255, -1)
                    if ((dilated_small > 0) & (old_region_mask > 0)).any():
                        connected_candidates.append(
                            (candidate_idx, cv2.contourArea(candidate_contour))
                        )

                if connected_candidates:
                    target_idx = max(connected_candidates, key=lambda item: item[1])[0]
                    logger.info(f"新区域 {small_idx} 面积太小，合并到相邻最大区域 {target_idx}")
                    bridge_p1 = bridge_p2 = None
                else:
                    # 若没有连通区域，合并到最近区域
                    best_distance = float("inf")
                    target_idx = 0
                    bridge_p1 = bridge_p2 = None
                    for candidate_idx, candidate_contour in enumerate(result):
                        distance, p1, p2 = _nearest_distance_with_points(
                            small_contour, candidate_contour
                        )
                        if distance < best_distance:
                            best_distance = distance
                            target_idx = candidate_idx
                            bridge_p1 = p1
                            bridge_p2 = p2
                    logger.info(
                        f"新区域 {small_idx} 面积太小且无连通区域，合并到最近区域 {target_idx}, distance={best_distance:.2f}"
                    )

                new_region_mask.fill(0)
                cv2.drawContours(new_region_mask, [result[target_idx]], -1, 255, -1)
                cv2.drawContours(new_region_mask, [small_contour], -1, 255, -1)
                if bridge_p1 is not None and bridge_p2 is not None:
                    # 非连通场景补一条连接线，确保并集是单连通区域
                    cv2.line(
                        new_region_mask,
                        (int(bridge_p1[0]), int(bridge_p1[1])),
                        (int(bridge_p2[0]), int(bridge_p2[1])),
                        255,
                        1,
                    )

                merged_cnts, _ = cv2.findContours(
                    new_region_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
                )
                if merged_cnts:
                    result[target_idx] = max(merged_cnts, key=cv2.contourArea)
                    if target_idx not in room_id_list:
                        room_id_list.append(target_idx)
                else:
                    logger.warning(f"新区域 {small_idx} 合并失败，保留原区域不变")

        # debug
        # draw_img = cv2.cvtColor(grid_map, cv2.COLOR_GRAY2BGR) if grid_map.ndim == 2 else grid_map.copy()
        # for idx, contour in enumerate(result):
        #     # 使用确定性颜色，避免循环内随机与重复 import
        #     color = ((37 * idx) % 256, (97 * idx) % 256, (157 * idx) % 256)
        #     cv2.drawContours(draw_img, [contour], -1, color, -1)
        # cv2.imwrite("./dataset/debug/5_gray.png", draw_img)

        return result, room_id_list

    def _get_shared_edge_polygons(
        self,
        contours: List[np.ndarray],
        shape: Tuple[int, int],
        targte_idx : int
    ) -> List[np.ndarray]:
        """
        仅对新房间执行“共享边界”重建，旧房间轮廓保持不变。

        做法：
        1) 先把当前 contours rasterize 成 label_map；
        2) 复用 postprocessor 的统一网格边界重建；
        3) 只替换 new room（不在 old_room_mapping 的 label）。
        """
        if not contours:
            return contours

        label_map = self._contours_to_label_map(contours, np.zeros(shape, dtype=np.uint8), shape)
        shared_polygons = self.postprocessor._extract_shared_edge_polygons_from_label_map(
            label_map
        )

        if not shared_polygons:
            return contours

        merged = contours.copy()
        for idx in range(len(contours)):
            if idx <= targte_idx:
                continue
            label_id = idx + 1
            poly = shared_polygons.get(label_id)
            if poly is None or poly.size == 0:
                continue
            merged[idx] = poly.reshape(-1, 1, 2).astype(np.int32)
        return merged
            
    # ==================== 序列化 (保留已有元数据) ====================

    def _build_old_room_mapping(
        self,
        old_label_map: np.ndarray,
        new_label_map: np.ndarray,
        old_rooms_data: List[Dict[str, Any]],
    ) -> Dict[int, int]:
        """
        建立新 label_map 中的 label → 旧 rooms_data 索引的映射

        通过像素重叠判断：新 label 中哪个与旧的某个 room 重叠最多，
        就认为该新 label 继承该旧房间的元数据。

        Args:
            old_label_map: 扩展前的 label_map (来自旧 labels_json)
            new_label_map: 扩展后的 label_map
            old_rooms_data: 旧的 rooms_data 列表

        Returns:
            {new_label_id: old_room_index} 映射，未匹配的新房间不在其中
        """
        mapping: Dict[int, int] = {}
        new_ids = sorted(set(new_label_map.flat) - {0})

        for new_id in new_ids:
            new_mask = new_label_map == new_id
            # 找与旧 label_map 重叠最多的 old_label
            overlap_labels = old_label_map[new_mask]
            overlap_labels = overlap_labels[overlap_labels > 0]

            if len(overlap_labels) == 0:
                continue

            unique, counts = np.unique(overlap_labels, return_counts=True)
            best_old_label = int(unique[np.argmax(counts)])

            # old_label 是 1-based，对应 old_rooms_data 索引 = label - 1
            old_idx = best_old_label - 1
            if 0 <= old_idx < len(old_rooms_data):
                mapping[new_id] = old_idx

        return mapping

    def serialize_contours(
        self,
        contours: List[np.ndarray],
        graph: Dict[int, List[int]],
        colors: Dict[int, int],
        order: List[int],
        transformer: CoordinateTransformer,
        old_rooms_data: List[Dict[str, Any]],
        charge_room_id_list: List[str],
    ) -> List[Dict[str, Any]]:
        """
        将轮廓序列化为 labels_json 的 data 列表 (ROOM 部分)

        三类房间：
        - 未变化旧房间：id/name/geometry/colorType/groundMaterial 全用旧值，仅更新 graph
        - 合并过旧房间：保留 id/name/groundMaterial，更新 geometry/graph/colorType
        - 新房间：分配新 id/name，全新 geometry/graph/colorType

        Args:
            contours: 排序后的轮廓列表
            graph: 邻接图 (排序前索引)
            colors: 颜色映射 (排序前索引)
            order: 排序映射 (new_idx → old_idx_in_contours)
            transformer: 坐标变换器
            old_rooms_data: 旧的 rooms_data 列表
            old_room_mapping: {label_id → old_room_index}
            merged_old_labels: 发生过合并的旧 label 集合 (None=全部视为合并过)
        """
        rooms_data: List[Dict[str, Any]] = []
        old_to_new = {old: new for new, old in enumerate(order)}
        logger.info(f"order: {order}")
        logger.info(f"graph: {graph}")
        logger.info(f"colors: {colors}")
        logger.info(f"old_to_new: {old_to_new}")
        logger.info(f"old number: {len(old_rooms_data)}")
        logger.info(f"contours number: {len(contours)}")
        
        for old_idx, new_idx in old_to_new.items():
            if old_idx < len(old_rooms_data):
                if old_idx in charge_room_id_list:
                    geometry = transformer.contour_to_geometry(contours[new_idx])
                    old_rooms_data[old_idx]["geometry"] = geometry
                rooms_data.append({
                    "name": old_rooms_data[old_idx]["name"],
                    "id": old_rooms_data[old_idx]["id"],
                    "type": "polygon",
                    "geometry": old_rooms_data[old_idx]["geometry"],
                    "colorType": colors.get(old_idx, 0),
                    "graph": graph.get(old_idx, []),
                    "groundMaterial": None,
                })
            else:
                geometry = transformer.contour_to_geometry(contours[new_idx])
                rooms_data.append({
                    "name": next_room_name(rooms_data),
                    "id": next_room_id(rooms_data),
                    "type": "polygon",
                    "geometry": geometry,
                    "colorType": colors.get(old_idx, 0),
                    "graph": graph.get(old_idx, []),
                    "groundMaterial": None,
                })
        
            # logger.info(f"rooms_data: {rooms_data}")
        return rooms_data

    # ==================== 完整流程入口 ====================

    def process(
        self,
        map_data: Dict[str, Any],
        transformer: CoordinateTransformer,
        graph_builder: RoomGraph,
        landmark_builder: LandmarkManager,
    ) -> Dict[str, Any]:
        """
        完整扩展分区流程 (已有 labels 场景)

        两条分支：
        - 无新增区域：最小改动，仅更新 graph/colors，保留旧数据
        - 有新增区域：合并/新房间处理，DFS 排序，完整流程
        """
        map_img = map_data["cleaned_img"]
        grid_map = map_data["cleaned_img2"]
        robot_model = map_data.get("robot_model", "s10")
        labels_json = map_data.get("labels_json", {})

        # ---- 公共前置 ----
        # step1: 分离房间和标记点
        old_rooms_data, old_landmarks_data = split_labels_data(labels_json)

        # step2: 从已有 labels 恢复轮廓 → 构建旧 label_map
        old_contours = transformer.rooms_data_to_contours(old_rooms_data)
        h, w = map_img.shape[:2]
        old_label_map = self._contours_to_label_map(
            old_contours, map_img, (h, w)
        )

        # step3: 检测新增自由区域
        # 用完整轮廓 label_map（非自由空间过滤版）避免边界像素被误判为新增区域
        new_regions_map = self.detect_new_regions(old_label_map, grid_map)
        new_region_count = int(new_regions_map.max())

        gray = self._to_gray(map_img)

        if new_region_count == 0:
            # ============ 分支 A：无新增区域 ============
            return self._process_no_new_regions(
                old_rooms_data, old_landmarks_data, old_contours,
                gray, graph_builder, map_data,
            )
        else:
            # ============ 分支 B：有新增区域 ============
            return self._process_with_new_regions(
                old_rooms_data, old_contours,old_label_map, new_regions_map,
                map_img, grid_map, gray, robot_model,
                transformer, graph_builder, landmark_builder, map_data,
            )

    def _process_no_new_regions(
        self,
        old_rooms_data: List[Dict[str, Any]],
        old_landmarks_data: List[Dict[str, Any]],
        old_contours: List[np.ndarray],
        gray: np.ndarray,
        graph_builder: RoomGraph,
        map_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        无新增区域：最小改动路径

        仅更新 graph/colors，id/name/geometry/groundMaterial 全部保留旧值。
        """
        logger.info("未检测到新增区域，保持原有分区")

        # 用旧轮廓在当前地图上重建邻接图
        graph = graph_builder.build_graph(old_contours, gray)

        # 保留旧颜色，仅修复冲突（相邻房间颜色相同）
        old_colors = {
            idx: room.get("colorType", 0) for idx, room in enumerate(old_rooms_data)
        }
        colors = self._fix_color_conflicts(graph, old_colors, graph_builder)

        # 就地更新 graph 和 colorType，其余字段不变
        for idx, room in enumerate(old_rooms_data):
            room["graph"] = sorted(graph.get(idx, []))
            room["colorType"] = colors.get(idx, room.get("colorType", 0))

        # 标记点保留旧数据，不重排序
        labels = {
            "version": self.config.get(
                "labels_version",
                f"v{self.config.get('service_version', '4.0.2')}",
            ),
            "uuid": map_data.get("uuid", ""),
            "data": old_rooms_data + old_landmarks_data,
        }

        logger.info(
            f"扩展分区完成(无新增): {len(old_rooms_data)} 个房间, "
            f"{len(old_landmarks_data)} 个标记点"
        )
        return labels

    def _process_with_new_regions(
        self,
        old_rooms_data: List[Dict[str, Any]],
        old_contours: List[np.ndarray],
        old_label_map: np.ndarray,
        new_regions_map: np.ndarray,
        map_img: np.ndarray,
        grid_map: np.ndarray,
        gray: np.ndarray,
        robot_model: str,
        transformer: CoordinateTransformer,
        graph_builder: RoomGraph,
        landmark_builder: LandmarkManager,
        map_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        有新增区域：threshold 线分界方案

        1. Triton 推理获取 threshold_list（兼容无 Triton）
        2. 过滤 threshold（跳过旧区域内的线）
        3. 用 threshold 切割新区域 → 合并/新房间
        4. 区域生长 + 重编号
        5. 三类房间分别处理（未变化/合并过/新房间）
        6. Graph/Colors 最小改动 + DFS 排序 + 序列化
        """
        new_region_count = int(new_regions_map.max())
        logger.info(f"检测到 {new_region_count} 个新增区域")
        h, w = old_label_map.shape[:2]


        # step1: Triton 推理获取 threshold_list（兼容无 Triton）
        threshold_list = []
        if self.inferencer and self.inferencer.is_ready():
            tensor = self._prepare_tensor(map_data["input_img"], map_data)
            raw_output = self.inferencer.run(tensor)
            self.postprocessor.max_extend = 5.0
            threshold_result = self.postprocessor._build_threshold_mask(
                raw_output, map_data["cleaned_img"], map_data
            )
            threshold_list = threshold_result.get("threshold_list", [])
            logger.info(f"Triton 推理: {len(threshold_list)} 条 threshold 线")


        # step2: 过滤 threshold（跳过两端都在旧区域内的线）
        if threshold_list:
            filtered_mask = self._filter_thresholds(threshold_list, old_label_map, gray)
        else:
            filtered_mask = np.zeros((h, w), dtype=np.uint8)

        # step3: 用 threshold 切割新区域 → 合并/新房间
        new_contours, charge_room_id_list = self._classify_new_regions(
            old_label_map, old_contours, grid_map, filtered_mask
        )
        target_idx = len(old_contours)-1
        new_contours = self._get_shared_edge_polygons(new_contours, (h, w), target_idx)

        # # debug 
        # draw_img = cv2.cvtColor(grid_map, cv2.COLOR_GRAY2BGR) if grid_map.ndim == 2 else grid_map.copy()
        # for idx, contour in enumerate(new_contours):
        #     # 使用确定性颜色，避免循环内随机与重复 import
        #     color = ((37 * idx) % 256, (97 * idx) % 256, (157 * idx) % 256)
        #     cv2.drawContours(draw_img, [contour], -1, color, 1)
        # cv2.imwrite("./dataset/debug/2_new_contours.png", draw_img)

        # step4: 基于 charge_room_id_list 外扩
        expander = ContourExpander(self.config)
        contours = new_contours.copy()
        for charge_room_id in charge_room_id_list:
            contours[charge_room_id] = expander.contour_expand(new_contours[charge_room_id], map_img)

        # debug 
        # draw_img = cv2.cvtColor(map_img, cv2.COLOR_BGR2RGB) if map_img.ndim == 2 else map_img.copy()
        # for idx, contour in enumerate(contours):
        #     # 使用确定性颜色，避免循环内随机与重复 import
        #     color = ((37 * idx) % 256, (97 * idx) % 256, (157 * idx) % 256)
        #     cv2.drawContours(draw_img, [contour], -1, color, 1)
        # cv2.imwrite("./dataset/debug/3_contours.png", draw_img)
        
        # step5: 邻接图 + 着色（最小改动）
        graph = graph_builder.build_graph(contours, gray)
        # 旧房间保留旧颜色，新房间分配新颜色，最后修复冲突
        old_colors: Dict[int, int] = {}
        for idx in range(len(contours)):
            if idx <= target_idx:
                color = old_rooms_data[idx].get("colorType", None)
                if color is None:
                    color = graph_builder.assign_color_for_room(
                        idx, graph, old_colors
                    )
                old_colors[idx] = color
            else:
                old_colors[idx] = graph_builder.assign_color_for_room(
                    idx, graph, old_colors
                )
        colors = self._fix_color_conflicts(graph, old_colors, graph_builder)

        # step6: DFS 排序
        charge_pixel = self._get_charge_pixel(map_data, transformer)
        start = graph_builder.find_start_room(
            contours, charge_pixel if charge_pixel else (0, 0),
            max_area_start=(charge_pixel is None or robot_model == "S-K20PRO"),
        )
        order = graph_builder.dfs_sort(graph, start)

        if not order:
            order = list(range(len(contours)))
        else:
            missing = [i for i in range(len(contours)) if i not in order]
            order.extend(missing)

        sorted_contours = [contours[i] for i in order]

        # debug 
        # draw_img = cv2.cvtColor(map_img, cv2.COLOR_BGR2RGB) if map_img.ndim == 2 else map_img.copy()
        # for idx, contour in enumerate(sorted_contours):
        #     # 使用确定性颜色，避免循环内随机与重复 import
        #     color = ((37 * idx) % 256, (97 * idx) % 256, (157 * idx) % 256)
        #     cv2.drawContours(draw_img, [contour], -1, color, 1)
        # cv2.imwrite("./dataset/debug/4_sorted_contours.png", draw_img)



        # step7: 序列化（三类房间分别处理）
        rooms_data = self.serialize_contours(
            sorted_contours, graph, colors, order,
            transformer, old_rooms_data, charge_room_id_list
        )

        # step8: 标记点 (K20)
        landmarks_data: List[Dict[str, Any]] = []
        if robot_model == "S-K20PRO":
            marker_polygons = self._get_marker_polygons(map_data)
            rooms_geometry = [d["geometry"] for d in rooms_data]
            room_names = [d["name"] for d in rooms_data]
            room_ids = [d["id"] for d in rooms_data]
            landmarks_data = landmark_builder.generate_landmarks(
                rooms_geometry, room_names, room_ids,
                marker_polygons=marker_polygons,
            )

        # step9: 组装输出
        labels = {
            "version": self.config.get(
                "labels_version",
                f"v{self.config.get('service_version', '4.0.2')}",
            ),
            "uuid": map_data.get("uuid", ""),
            "data": rooms_data + landmarks_data,
        }

        logger.info(
            f"扩展分区完成: {len(rooms_data)} 个房间 "
            f"(新增 {len(rooms_data) - len(old_rooms_data)} 个, "
            f"{len(landmarks_data)} 个标记点"
        )
        return labels

    @staticmethod
    def _fix_color_conflicts(
        graph: Dict[int, List[int]],
        colors: Dict[int, int],
        graph_builder: RoomGraph,
    ) -> Dict[int, int]:
        """
        保留旧颜色，仅修复冲突：相邻房间颜色相同时重新分配。
        """
        result = dict(colors)
        changed = True
        while changed:
            changed = False
            for room_idx in sorted(result.keys()):
                neighbors = graph.get(room_idx, [])
                neighbor_colors = {result[nb] for nb in neighbors if nb in result}
                if result[room_idx] in neighbor_colors:
                    new_color = graph_builder.assign_color_for_room(
                        room_idx, graph, result
                    )
                    if new_color != result[room_idx]:
                        result[room_idx] = new_color
                        changed = True
        return result

