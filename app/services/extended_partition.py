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
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
import cv2

from app.utils.coordinate import CoordinateTransformer
from app.utils.graph import RoomGraph
from app.utils.landmark import LandmarkManager
from app.utils.contour_expander import ContourExpander
from app.utils.beautifier import ContourBeautifier
from app.utils.geometry_ops import (
    split_labels_data,
    next_room_id,
    next_room_name,
)

logger = logging.getLogger(__name__)



class ExtendedPartitioner:
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
        self.config = config or {}
        self.door_width = self.config.get("door_width", 20)
        self.grow_iterations = self.config.get("grow_iterations", 10)
        self.wall_threshold = self.config.get("wall_threshold", 128)
        self.resolution = self.config.get("resolution", 0.05)
        self.min_room_area = self.config.get("min_room_area", 1.0)  # m²
        self.min_new_region_area = self.config.get("min_new_region_area", 50)  # pixels
        self.merge_area_threshold = self.config.get("merge_area_threshold", 0)  # m², 0=use min_room_area
        self.merge_ratio_threshold = self.config.get("merge_ratio_threshold", 0.6)

        self.beautifier_status = False

    def set_beautifier_status(self, status: bool):
        self.beautifier_status = status

    # ==================== 像素级底层操作 ====================

    def split_by_doorway(self, label_map: np.ndarray, _grid_map: np.ndarray) -> np.ndarray:
        """
        通过门口检测拆分大区域

        在墙壁上寻找窄通道（门口），切断后重新标记连通域
        """
        result = label_map.copy()
        next_label = label_map.max() + 1

        for lid in range(1, label_map.max() + 1):
            mask = (label_map == lid).astype(np.uint8)
            if mask.sum() == 0:
                continue

            # 腐蚀找窄通道
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                               (self.door_width, self.door_width))
            eroded = cv2.erode(mask, kernel)

            # 检查腐蚀后是否分裂成多个连通域
            num_labels, sub_labels = cv2.connectedComponents(eroded)
            if num_labels <= 2:  # 1(背景) + 1(区域) = 没有分裂
                continue

            # 用区域生长把原始像素分配给最近的子区域
            grown = self._region_grow(mask, sub_labels, num_labels)
            for sub_id in range(2, num_labels):
                sub_mask = grown == sub_id
                result[sub_mask] = next_label
                next_label += 1

        return result

    def grow_unassigned(self, label_map: np.ndarray, grid_map: np.ndarray) -> np.ndarray:
        """
        区域生长：将未分配的空闲像素分配给最近的房间

        适用于自动分区后边界附近的未标记像素
        """
        free_space = grid_map >= self.wall_threshold
        unassigned = (label_map == 0) & free_space

        if not unassigned.any():
            return label_map

        result = label_map.copy()
        for _ in range(self.grow_iterations):
            if not (result == 0).any():
                break
            # 对每个已标记区域膨胀一步
            for lid in range(1, result.max() + 1):
                mask = (result == lid).astype(np.uint8)
                dilated = cv2.dilate(mask, np.ones((3, 3), np.uint8))
                # 只填充未分配的空闲区域
                fill = (dilated > 0) & (result == 0) & free_space
                result[fill] = lid

        return result

    def extend_pixel(self, label_map: np.ndarray, grid_map: np.ndarray) -> np.ndarray:
        """
        像素级完整扩展分区流程

        1. 门口检测拆分
        2. 未分配区域生长
        """
        result = self.split_by_doorway(label_map, grid_map)
        result = self.grow_unassigned(result, grid_map)
        return result

    def extend(self, label_map: np.ndarray, grid_map: np.ndarray) -> np.ndarray:
        """兼容旧接口：等价于 extend_pixel。"""
        return self.extend_pixel(label_map, grid_map)

    def _region_grow(self, mask: np.ndarray, seeds: np.ndarray,
                     num_labels: int) -> np.ndarray:
        """从种子标签向外生长，填满 mask 区域"""
        result = seeds.copy()
        for _ in range(self.grow_iterations):
            changed = False
            for lid in range(1, num_labels):
                seed_mask = (result == lid).astype(np.uint8)
                dilated = cv2.dilate(seed_mask, np.ones((3, 3), np.uint8))
                fill = (dilated > 0) & (result == 0) & (mask > 0)
                if fill.any():
                    result[fill] = lid
                    changed = True
            if not changed:
                break
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

        # 过滤面积过小的碎片
        result = np.zeros_like(label_map)
        new_id = 0
        for rid in range(1, num_labels):
            region_mask = regions == rid
            pixel_count = int(region_mask.sum())
            if pixel_count < self.min_new_region_area:
                continue
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

    def assign_new_regions(self, label_map: np.ndarray,
                           new_regions_map: np.ndarray) -> np.ndarray:
        """
        将检测到的新区域分配到已有房间或创建新房间

        Args:
            label_map: 已有房间标签 (H, W) int32
            new_regions_map: detect_new_regions 返回的新区域图

        Returns:
            更新后的 label_map
        """
        if new_regions_map.max() == 0:
            return label_map

        result = label_map.copy()
        next_label = int(label_map.max()) + 1

        for rid in range(1, new_regions_map.max() + 1):
            region_mask = new_regions_map == rid
            if not region_mask.any():
                continue

            action, target = self.classify_region(region_mask, result)

            if action == "merge" and target > 0:
                result[region_mask] = target
            else:
                result[region_mask] = next_label
                next_label += 1

        return result

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
        old_room_mapping: Dict[int, int],
    ) -> List[Dict[str, Any]]:
        """
        将轮廓序列化为 labels_json 的 data 列表 (ROOM 部分)

        与 AutoPartitioner.serialize_contours 的关键区别：
        保留已有房间的 name/id/colorType/groundMaterial 元数据，
        仅对新增房间分配新的 name/id。

        Args:
            contours: 排序后的轮廓列表
            graph: 邻接图 (排序前索引)
            colors: 颜色映射 (排序前索引)
            order: 排序映射 (new_idx → old_idx_in_contours)
            transformer: 坐标变换器
            old_rooms_data: 旧的 rooms_data 列表
            old_room_mapping: {label_id → old_room_index}，label_id 是 relabel 后的 1-based

        Returns:
            rooms_data 列表
        """
        rooms_data: List[Dict[str, Any]] = []
        old_to_new = {old: new for new, old in enumerate(order)}

        for new_idx, cnt in enumerate(contours):
            old_idx = order[new_idx]
            geometry = transformer.contour_to_geometry(cnt, clockwise=True)

            # 重映射 graph 邻居索引
            old_neighbors = graph.get(old_idx, [])
            new_neighbors = sorted(
                old_to_new[nb] for nb in old_neighbors if nb in old_to_new
            )

            # label_id = old_idx + 1 (因为 contours 是按 relabel 后的顺序提取的)
            label_id = old_idx + 1
            old_room_idx = old_room_mapping.get(label_id)

            if old_room_idx is not None:
                # 继承旧房间元数据
                old_room = old_rooms_data[old_room_idx]
                rooms_data.append({
                    "name": old_room.get("name", chr(ord("A") + new_idx % 26)),
                    "id": old_room.get("id", f"ROOM_{new_idx + 1:03d}"),
                    "type": old_room.get("type", "polygon"),
                    "geometry": geometry,
                    "colorType": colors.get(old_idx, old_room.get("colorType", 0)),
                    "graph": new_neighbors,
                    "groundMaterial": old_room.get("groundMaterial"),
                })
            else:
                # 新房间：分配新 name/id
                rooms_data.append({
                    "name": next_room_name(rooms_data + old_rooms_data),
                    "id": next_room_id(rooms_data + old_rooms_data),
                    "type": "polygon",
                    "geometry": geometry,
                    "colorType": colors.get(old_idx, 0),
                    "graph": new_neighbors,
                    "groundMaterial": None,
                })

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

        Args:
            map_data: 地图数据
                "map_img": 地图图像 (H, W) 或 (H, W, 3)
                "labels_json": 已有标注 (必须有值)
                "robot_model": 机器人型号 str
                "uuid": UUID str
                "markers_json": 标记信息 json
                "world_charge_pose": 充电桩世界坐标 [x, y, z]
                "input_img": 补墙平滑后的灰度地图 (H, W) uint8
            transformer: 坐标变换器
            graph_builder: 邻接图构建器
            landmark_builder: 标记点管理器

        Returns:
            Dict[str, Any]: labels_json 格式
        """
        map_img = map_data["map_img"]
        grid_map = map_data["input_img"]
        robot_model = map_data.get("robot_model", "s10")
        labels_json = map_data.get("labels_json", {})

        # step1: 分离房间和标记点
        old_rooms_data, _old_landmarks_data = split_labels_data(labels_json)

        # step2: 从已有 labels 恢复轮廓 → 构建旧 label_map
        old_contours = transformer.rooms_data_to_contours(old_rooms_data)
        h, w = map_img.shape[:2]
        old_label_map = self._contours_to_label_map(old_contours, (h, w))

        # step3: 检测新增自由区域
        new_regions_map = self.detect_new_regions(old_label_map, grid_map)
        new_region_count = int(new_regions_map.max())

        if new_region_count > 0:
            logger.info(f"检测到 {new_region_count} 个新增区域")
            # step4: 分配新区域归属
            label_map = self.assign_new_regions(old_label_map, new_regions_map)
        else:
            logger.info("未检测到新增区域，仅执行扩展优化")
            label_map = old_label_map.copy()

        # step5: 门口检测拆分 + 区域生长
        label_map = self.extend_pixel(label_map, grid_map)

        # step6: 重编号
        label_map = self._relabel(label_map)

        # step7: 建立新旧房间映射
        old_room_mapping = self._build_old_room_mapping(
            old_label_map, label_map, old_rooms_data
        )

        # step8: 提取轮廓
        contours = self._extract_contours(label_map)

        # step9: 轮廓外扩
        expander = ContourExpander(self.config)
        contours = expander.expand(contours, map_img)

        # step10: 邻接图 + 着色
        gray = self._to_gray(map_img)
        graph = graph_builder.build_graph(contours, gray)
        colors = graph_builder.assign_colors(graph)

        # step11: DFS 排序
        charge_pixel = self._get_charge_pixel(map_data, transformer)
        start = graph_builder.find_start_room(
            contours, charge_pixel or (0, 0),
            max_area_start=(charge_pixel is None or robot_model == "S-K20PRO"),
        )
        order = graph_builder.dfs_sort(graph, start)

        if not order:
            order = list(range(len(contours)))
        else:
            missing = [i for i in range(len(contours)) if i not in order]
            order.extend(missing)

        sorted_contours = [contours[i] for i in order]

        # step12: 序列化 (保留旧元数据)
        rooms_data = self.serialize_contours(
            sorted_contours, graph, colors, order,
            transformer, old_rooms_data, old_room_mapping,
        )

        # step13: 美化框 (s10)
        if robot_model and "s10" in robot_model.lower():
            self.set_beautifier_status(True)
        if self.beautifier_status:
            beautifier = ContourBeautifier(self.config)
            beautifier.beautify(sorted_contours, map_img)

        # step14: 标记点 (K20)
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

        # step15: 组装输出
        labels = {
            "version": self.config.get("labels_version", f"v{self.config.get('service_version', '4.0.2')}"),
            "uuid": map_data.get("uuid", ""),
            "data": rooms_data + landmarks_data,
        }

        logger.info(
            f"扩展分区完成: {len(rooms_data)} 个房间 "
            f"(新增 {len(rooms_data) - len(old_rooms_data)} 个), "
            f"{len(landmarks_data)} 个标记点"
        )
        return labels

    # ==================== 内部工具 ====================

    @staticmethod
    def _extract_contours(label_map: np.ndarray) -> List[np.ndarray]:
        """从 label_map 提取各房间轮廓"""
        contours = []
        if label_map is None or label_map.max() == 0:
            return contours
        for lid in range(1, label_map.max() + 1):
            mask = (label_map == lid).astype(np.uint8)
            if mask.sum() == 0:
                continue
            cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
            if cnts:
                contours.append(max(cnts, key=cv2.contourArea))
        return contours

    @staticmethod
    def _relabel(label_map: np.ndarray) -> np.ndarray:
        """重编号使 label ID 从 1 开始连续"""
        unique_ids = sorted(set(label_map.flat) - {0})
        new_map = np.zeros_like(label_map)
        for new_id, old_id in enumerate(unique_ids, start=1):
            new_map[label_map == old_id] = new_id
        return new_map

    @staticmethod
    def _contours_to_label_map(contours: List[np.ndarray],
                               shape: Tuple[int, int]) -> np.ndarray:
        """轮廓列表 → label_map"""
        h, w = shape[:2]
        label_map = np.zeros((h, w), dtype=np.int32)
        for i, cnt in enumerate(contours):
            cv2.drawContours(label_map, [cnt], -1, i + 1, -1)
        return label_map

    @staticmethod
    def _to_gray(img: np.ndarray) -> np.ndarray:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img.copy()

    @staticmethod
    def _get_charge_pixel(
        map_data: Dict[str, Any],
        transformer: CoordinateTransformer,
    ) -> Optional[Tuple[int, int]]:
        """充电桩世界坐标 → 像素坐标"""
        pose = map_data.get("world_charge_pose", [0, 0, 0])
        if pose == [0, 0, 0]:
            return None
        return transformer.world_to_pixel(pose[0], pose[1])

    @staticmethod
    def _get_marker_polygons(map_data: Dict[str, Any]) -> Optional[List]:
        """提取家具标记多边形"""
        markers_json = map_data.get("markers_json")
        if not markers_json:
            return None
        polys = []
        for item in markers_json.get("data", []):
            if item.get("name") == "家具" and item.get("type") == "polygon":
                polys.append(item["geometry"])
        return polys if polys else None
