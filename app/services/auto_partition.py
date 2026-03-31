"""自动分区模块

继承 BasePartitioner 获得推理能力和通用工具方法，
在此基础上实现完整的自动分区流程：
  推理/fallback → 轮廓外扩 → 邻接图着色 → DFS排序 → 序列化 → 美化 → 标记点

process() 入口签名与 ManualPartitioner / ManualMerger 保持一致:
  process(map_data, transformer, graph_builder, landmark_builder) -> labels_json
"""

import logging
from typing import Dict, Any, List, Tuple, Optional

import numpy as np

from app.services.base_partitioner import BasePartitioner
from app.utils.coordinate import CoordinateTransformer
from app.utils.graph import RoomGraph
from app.utils.landmark import LandmarkManager
from app.utils.labels_ops import ContourExpander
from app.utils.beautifier import ContourBeautifier

logger = logging.getLogger(f"{__name__} [AutoPartitioner]")


class AutoPartitioner(BasePartitioner):
    """
    自动分区：推理 → 后处理 → 轮廓提取 → 图着色 → 序列化

    当 Triton 不可用时，fallback 到传统连通域方法。
    """

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)

    # ==================== 轮廓扩展 ====================

    def expand_contours(self, contours: List[np.ndarray],
                        map_img: np.ndarray) -> List[np.ndarray]:
        """将房间轮廓向外扩展一圈, 覆盖边界空闲像素"""
        expander = ContourExpander(self.config)
        return expander.expand(contours, map_img)

    # ==================== 图着色 ====================

    def build_graph_and_colors(
        self,
        contours: List[np.ndarray],
        map_img: np.ndarray,
        graph_builder: RoomGraph,
    ) -> Tuple[Dict[int, List[int]], Dict[int, int]]:
        """
        构建邻接图 + 全量着色

        Returns:
            (graph, colors)
        """
        graph = graph_builder.build_graph(contours, map_img)
        colors = graph_builder.assign_colors(graph)
        return graph, colors

    # ==================== 房间排序 ====================

    def sort_contours(
        self,
        contours: List[np.ndarray],
        graph: Dict[int, List[int]],
        graph_builder: RoomGraph,
        charge_pixel: Tuple[int, int] = None,
        max_area_start: bool = False,
    ) -> Tuple[List[np.ndarray], List[int]]:
        """
        按 DFS 顺序排列轮廓

        Returns:
            (排序后的 contours, 排序顺序 order)
        """
        if charge_pixel:
            start = graph_builder.find_start_room(contours, charge_pixel,
                                                   max_area_start=max_area_start)
        else:
            start = graph_builder.find_start_room(contours, (0, 0),
                                                   max_area_start=True)

        order = graph_builder.dfs_sort(graph, start)

        # 兜底: 确保每个房间都在 order 中
        if not order:
            order = list(range(len(contours)))
        else:
            missing = [i for i in range(len(contours)) if i not in order]
            order.extend(missing)

        sorted_contours = [contours[i] for i in order]
        return sorted_contours, order

    # ==================== 序列化 ====================

    def serialize_contours(
        self,
        contours: List[np.ndarray],
        graph: Dict[int, List[int]],
        colors: Dict[int, int],
        order: List[int],
        transformer: CoordinateTransformer,
    ) -> List[Dict[str, Any]]:
        """
        将轮廓序列化为 labels_json 的 data 列表 (ROOM 部分)
        """
        rooms_data = []
        old_to_new = {old: new for new, old in enumerate(order)}

        for new_idx, cnt in enumerate(contours):
            old_idx = order[new_idx]
            geometry = transformer.contour_to_geometry(cnt, clockwise=True)
            old_neighbors = graph.get(old_idx, [])
            new_neighbors = sorted(
                old_to_new[nb] for nb in old_neighbors if nb in old_to_new
            )
            name_idx = new_idx % 26
            name_suffix = new_idx // 26
            name = chr(ord("A") + name_idx) + (str(name_suffix) if name_suffix else "")

            rooms_data.append({
                "name": name,
                "id": f"ROOM_{new_idx + 1:03d}",
                "type": "polygon",
                "geometry": geometry,
                "colorType": colors.get(old_idx, 0),
                "graph": new_neighbors,
                "groundMaterial": None,
            })

        return rooms_data

    # ==================== 标记点 ====================

    def build_landmarks(
        self,
        rooms_data: List[Dict[str, Any]],
        landmark_builder: LandmarkManager,
        marker_polygons: List[List[float]] = None,
    ) -> List[Dict[str, Any]]:
        """为所有房间生成平台标记点"""
        rooms_geometry = [d["geometry"] for d in rooms_data]
        room_names = [d["name"] for d in rooms_data]
        room_ids = [d["id"] for d in rooms_data]

        return landmark_builder.generate_landmarks(
            rooms_geometry, room_names, room_ids,
            marker_polygons=marker_polygons,
        )

    # ==================== 美化框 ====================

    def beautify_contours(
        self,
        contours: List[np.ndarray],
        map_img: np.ndarray,
    ) -> Tuple[Optional[List], Optional[List]]:
        """s10 美化框 + 门槛线"""
        if not self.beautifier_status:
            return None, None
        beautifier = ContourBeautifier(self.config)
        return beautifier.beautify(contours, map_img)

    # ==================== 完整流程入口 ====================

    def process(
        self,
        map_data: Dict[str, Any],
        transformer: CoordinateTransformer,
        graph_builder: RoomGraph,
        landmark_builder: LandmarkManager,
        extend: bool = True,
        repartition: bool = False,
    ) -> Dict[str, Any]:
        """
        完整自动分区流程

        Args:
            map_data: 地图数据字典
            transformer: 坐标变换器
            graph_builder: 邻接图构建器
            landmark_builder: 标记点管理器
            extend: 是否扩展分区
            repartition: 是否重新分区

        Returns:
            Dict[str, Any]: labels_json 格式
        """
        map_img = map_data["map_img"]
        robot_model = map_data.get("robot_model", "s10")

        # step1: 分区
        logger.info("重新分区" if repartition else "首次分区")
        room_polygons = self.partition(map_data, extend=extend)

        contours = [np.asarray(polygon, dtype=np.int32).reshape(-1, 1, 2)
                    for polygon in room_polygons]

        # step2: 邻接图 + 着色
        gray = self._to_gray(map_img)
        graph, colors = self.build_graph_and_colors(contours, gray, graph_builder)

        # step3: 排序
        charge_pixel = self._get_charge_pixel(map_data, transformer)
        contours, order = self.sort_contours(
            contours, graph, graph_builder,
            charge_pixel=charge_pixel,
            max_area_start=(robot_model == "S-K20PRO"),
        )

        # step4: 序列化房间
        rooms_data = self.serialize_contours(
            contours, graph, colors, order, transformer
        )

        # step5: 美化框 (s10)
        if robot_model and "s10" in robot_model.lower():
            self.set_beautifier_status(True)
        _bbox_list, _threshold_list = self.beautify_contours(contours, map_img)

        # step6: 标记点 (K20)
        landmarks_data = []
        if robot_model == "S-K20PRO":
            marker_polygons = self._get_marker_polygons(map_data)
            landmarks_data = self.build_landmarks(
                rooms_data, landmark_builder,
                marker_polygons=marker_polygons,
            )

        # step7: 组装输出
        labels = {
            "version": self.config.get("labels_version", f"v{self.config.get('service_version', '4.0.2')}"),
            "uuid": map_data.get("uuid", ""),
            "data": rooms_data + landmarks_data,
        }

        logger.info(f"分区完成: {len(rooms_data)} 个房间, {len(landmarks_data)} 个标记点")
        return labels

    # ==================== 内部工具 ====================

    def _restore_contours(
        self,
        labels_json: Dict[str, Any],
        transformer: CoordinateTransformer,
        _map_img: np.ndarray,
    ) -> List[np.ndarray]:
        """从已有 labels_json 恢复轮廓"""
        rooms_data = [d for d in labels_json.get("data", [])
                      if "ROOM" in d.get("id", "")]
        return transformer.rooms_data_to_contours(rooms_data)
