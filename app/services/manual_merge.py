"""手动合并模块

支持两种坐标模式：
  - 像素级 (pixel)：操作 label_map (numpy int32)，用于内存中的实时分区流程
  - 世界坐标级 (world)：操作 labels_json geometry，用于 Lambda 无状态场景
"""

import logging
from typing import Dict, Any, List, Tuple

import numpy as np
import cv2
from shapely.geometry import Polygon
from shapely.ops import unary_union

from app.utils.coordinate import CoordinateTransformer
from app.utils.graph import RoomGraph
from app.utils.landmark import LandmarkManager
from app.core.errors import (
    InvalidParameterError,
    RoomIndexOutOfRangeError,
    RoomsNotConnectedError,
)
from app.utils.geometry_ops import (
    split_labels_data,
    find_room_index_by_id,
    flatten_geometry,
)

logger = logging.getLogger(__name__)


class ManualMerger:
    """
    手动合并：用户指定房间ROOM_ID列表，将多个房间区域合并为一个房间。
    """

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.beautifier_status = False

    def set_beautifier_status(self, status: bool):
        self.beautifier_status = status

    # ==================== 参数校验 ====================

    def _extract_merge_params(self,
                              rooms_data: List[Dict[str, Any]],
                              room_merge_list: List[str]
                              ) -> List[int]:
        """
        校验参数，提取合并列表ROOM_ID对应的索引列表
        """
        if not isinstance(room_merge_list, list):
            raise InvalidParameterError("room_merge_list 必须是列表")
        if not all(isinstance(room_id, str) for room_id in room_merge_list):
            raise InvalidParameterError("room_merge_list 中的每个元素必须是字符串")
        if len(room_merge_list) < 2:
            raise InvalidParameterError("room_merge_list 中的房间数量不足 2 个")
        if not all(room_id in [room["id"] for room in rooms_data] for room_id in room_merge_list):
            raise RoomIndexOutOfRangeError()

        return [find_room_index_by_id(rooms_data, room_id) for room_id in room_merge_list]

    # ==================== 像素级合并 ====================

    @staticmethod
    def merge_rooms_pixel(label_map: np.ndarray,
                          room_ids: List[int]) -> np.ndarray:
        """
        像素级合并：将多个 label ID 合并为一个

        Args:
            label_map: (H, W) int32, 房间标签图
            room_ids: 待合并的 label ID 列表 (1-based)

        Returns:
            更新后的 label_map
        """
        if len(room_ids) < 2:
            raise InvalidParameterError("至少需要 2 个房间 ID")

        label_map = label_map.copy()
        keep_id = room_ids[0]
        for rid in room_ids[1:]:
            label_map[label_map == rid] = keep_id

        return label_map

    @staticmethod
    def merge_by_point_pixel(label_map: np.ndarray,
                             pt1: Tuple[int, int],
                             pt2: Tuple[int, int]) -> np.ndarray:
        """
        像素级点选合并：找到两个点所在的房间并合并

        Args:
            label_map: (H, W) int32
            pt1: 第一个点 (x, y)
            pt2: 第二个点 (x, y)

        Returns:
            更新后的 label_map
        """
        id1 = int(label_map[pt1[1], pt1[0]])
        id2 = int(label_map[pt2[1], pt2[0]])

        if id1 == 0 or id2 == 0:
            raise InvalidParameterError("所选点不在任何房间内")
        if id1 == id2:
            raise InvalidParameterError("两个点在同一房间内")

        return ManualMerger.merge_rooms_pixel(label_map, [id1, id2])

    @staticmethod
    def relabel(label_map: np.ndarray) -> np.ndarray:
        """
        重新编号 label_map，使 label ID 从 1 开始连续

        Args:
            label_map: (H, W) int32

        Returns:
            重编号后的 label_map
        """
        label_map = label_map.copy()
        unique_ids = sorted(set(label_map.flat) - {0})
        new_map = np.zeros_like(label_map)
        for new_id, old_id in enumerate(unique_ids, start=1):
            new_map[label_map == old_id] = new_id
        return new_map

    # ==================== 世界坐标级合并 ====================

    def _check_connectivity(self,
                            rooms_data: List[Dict[str, Any]],
                            merge_indices: List[int],
                            transformer: CoordinateTransformer,
                            map_img: np.ndarray):
        """
        检查待合并房间是否两两连通（至少所有房间在同一连通分量中）

        通过逐对检查邻接关系，构建合并子图，验证连通性。
        """
        graph_builder = RoomGraph(self.config)
        contours = [transformer.world_to_contour(rooms_data[i]['geometry'])
                     for i in merge_indices]

        # 构建子图：检查每对轮廓是否相邻
        n = len(merge_indices)
        adj = {i: [] for i in range(n)}
        for i in range(n):
            for j in range(i + 1, n):
                if graph_builder.check_connectivity(contours[i], contours[j], map_img):
                    adj[i].append(j)
                    adj[j].append(i)

        # BFS 验证连通性
        visited = set()
        queue = [0]
        visited.add(0)
        while queue:
            node = queue.pop(0)
            for nb in adj[node]:
                if nb not in visited:
                    visited.add(nb)
                    queue.append(nb)

        if len(visited) < n:
            raise RoomsNotConnectedError()

    def merge_rooms(self,
                    rooms_data: List[Dict[str, Any]],
                    roomid_index_list: List[int]
                    ) -> Tuple[List[Dict[str, Any]], int]:
        """
        世界坐标级合并：使用 Shapely 对多个房间的 geometry 做 union

        保留第一个房间的元数据（name, id, type 等），删除其余房间。

        Args:
            rooms_data: 房间数据列表
            roomid_index_list: 待合并房间在 rooms_data 中的索引列表

        Returns:
            (更新后的 rooms_data, 合并后房间在新列表中的索引)
        """
        keep_idx = roomid_index_list[0]
        remove_indices = set(roomid_index_list[1:])

        # 收集所有待合并房间的 Polygon
        polygons = []
        for idx in roomid_index_list:
            geom = rooms_data[idx]['geometry']
            pts = [(geom[i], geom[i + 1]) for i in range(0, len(geom) - 1, 2)]
            # 去掉闭合尾点
            if len(pts) >= 2 and pts[0] == pts[-1]:
                pts = pts[:-1]
            polygons.append(Polygon(pts))

        # Shapely union
        merged = unary_union(polygons)

        # 处理 union 结果
        if merged.geom_type == 'MultiPolygon':
            # 取面积最大的
            merged = max(merged.geoms, key=lambda g: g.area)

        # 提取外轮廓坐标（去掉闭合尾点）
        merged_coords = list(merged.exterior.coords)[:-1]
        merged_geom = flatten_geometry(merged_coords)

        # 更新保留房间的 geometry
        rooms_data[keep_idx]['geometry'] = merged_geom

        # 删除其余房间（倒序删除避免索引偏移）
        for idx in sorted(remove_indices, reverse=True):
            rooms_data.pop(idx)

        # 计算合并后房间在新列表中的索引
        new_idx = keep_idx - sum(1 for ri in remove_indices if ri < keep_idx)

        # 重置合并后房间的 colorType 和 graph，交由后续重建
        rooms_data[new_idx]['colorType'] = None
        rooms_data[new_idx]['graph'] = None

        return rooms_data, new_idx

    def merge_world(self,
                    rooms_data: List[Dict[str, Any]],
                    room_merge_list: List[str],
                    transformer: CoordinateTransformer,
                    map_img: np.ndarray,
                    ) -> Tuple[List[Dict[str, Any]], int]:
        """
        世界坐标级合并入口（editor.py 调用）

        Args:
            rooms_data: 房间数据列表
            room_merge_list: 待合并房间 ID 列表 ["ROOM_001", "ROOM_002"]
            transformer: 坐标变换器
            map_img: 地图图像

        Returns:
            (更新后的 rooms_data, 合并后房间索引)
        """
        # 校验参数
        merge_indices = self._extract_merge_params(rooms_data, room_merge_list)

        # 检查连通性
        self._check_connectivity(rooms_data, merge_indices, transformer, map_img)

        # 执行合并
        rooms_data, new_idx = self.merge_rooms(rooms_data, merge_indices)

        return rooms_data, new_idx

    # ==================== 完整处理流程（含图着色和标记点） ====================

    def build_graph_and_colors(self,
                               rooms_data: List[Dict[str, Any]],
                               transformer: CoordinateTransformer,
                               graph_builder: RoomGraph,
                               map_img: np.ndarray,
                               target_room_idx: int,
                               ) -> Tuple[List[Dict[str, Any]], List[np.ndarray], Dict[int, List[int]]]:
        """
        构建邻接图和五色地图
        """
        contours_list = transformer.rooms_data_to_contours(rooms_data)
        graph = graph_builder.build_graph(contours_list, map_img)

        for room_idx in range(len(rooms_data)):
            logger.info(f">>>> room_idx: {room_idx}, "
                        f"original graph: {rooms_data[room_idx]['graph']}, "
                        f"new graph: {graph[room_idx]}")
            rooms_data[room_idx]["graph"] = graph[room_idx]

        # 着色：保留已有颜色，仅为合并后的房间重新分配
        current_colors = {i: rooms_data[i]["colorType"] for i in range(len(rooms_data))}
        if rooms_data[target_room_idx]["colorType"] is None:
            rooms_data[target_room_idx]["colorType"] = graph_builder.assign_color_for_room(
                target_room_idx, graph, current_colors
            )

        return rooms_data, contours_list, graph

    def build_landmarks(
        self,
        landmarks_data: List[Dict[str, Any]],
        new_rooms_data: List[Dict[str, Any]],
        merged_room_idx: int,
        graph: Dict[int, List[int]],
        contours_list: List[np.ndarray],
        world_charge_pose: List[float],
        transformer: CoordinateTransformer,
        graph_builder: RoomGraph,
        landmark_builder: LandmarkManager,
    ) -> List[Dict[str, Any]]:
        """
        重建标记点：
        - 非合并房间保留原标记点
        - 合并后的房间重新计算中心点
        """
        world_charge_pixel = transformer.world_to_pixel(
            world_charge_pose[0], world_charge_pose[1]
        )
        start_room_idx = graph_builder.find_start_room(
            contours_list, world_charge_pixel, max_area_start=True
        )
        room_order = graph_builder.dfs_sort(graph, start_room_idx)

        if not room_order:
            room_order = list(range(len(new_rooms_data)))
        else:
            missing = [idx for idx in range(len(new_rooms_data)) if idx not in room_order]
            room_order.extend(missing)

        # 旧标记点按 roomId 分组
        old_landmarks_by_room = {}
        for landmark in landmarks_data:
            room_id = landmark.get("roomId")
            if room_id:
                old_landmarks_by_room.setdefault(room_id, []).append(landmark)

        new_landmarks_data = []
        for new_room_idx in room_order:
            room = new_rooms_data[new_room_idx]
            room_id = room["id"]

            # 合并后的房间需要重新计算标记点
            if new_room_idx == merged_room_idx:
                markers_polygons = []  # TODO: 接入家具/标记物多边形避障
                new_pose = landmark_builder._find_center(
                    room["geometry"], markers_polygons
                )
                new_pose = [new_pose[0], new_pose[1], 0]
                new_landmarks_data.append({
                    "geometry": new_pose,
                    "id": f"PLATFORM_LANDMARK_{len(new_landmarks_data) + 1:03d}",
                    "roomId": room_id,
                    "name": room["name"],
                    "type": "pose",
                })
                continue

            # 非合并房间：复用旧标记点
            matched = old_landmarks_by_room.get(room_id, [])
            if matched:
                for landmark in matched:
                    new_landmarks_data.append({
                        "geometry": landmark["geometry"],
                        "id": f"PLATFORM_LANDMARK_{len(new_landmarks_data) + 1:03d}",
                        "roomId": room_id,
                        "name": room["name"],
                        "type": "pose",
                    })
            else:
                markers_polygons = []
                new_pose = landmark_builder._find_center(
                    room["geometry"], markers_polygons
                )
                new_pose = [new_pose[0], new_pose[1], 0]
                new_landmarks_data.append({
                    "geometry": new_pose,
                    "id": f"PLATFORM_LANDMARK_{len(new_landmarks_data) + 1:03d}",
                    "roomId": room_id,
                    "name": room["name"],
                    "type": "pose",
                })

        return new_landmarks_data

    def process(self,
                map_data: Dict[str, Any],
                roomid_merge_list: List[str],
                transformer: CoordinateTransformer,
                graph_builder: RoomGraph,
                landmark_builder: LandmarkManager,
                ) -> Dict[str, Any]:
        """
        处理房间合并：完整流程（含图着色和标记点重建）

        Args:
            map_data: 地图数据
                "map_img": 地图图像 (H, W, 3) uint8
                "resolution": 地图分辨率 float
                "origin": 地图原点 [x, y] float
                "labels_json": 房间标注数据 json
                "robot_model": 机器人型号 str
                "uuid": 机器人 UUID str
                "markers_json": 标记信息 json
                "world_charge_pose": 充电桩世界坐标 [x, y, z] float
            roomid_merge_list: 待合并房间 ID 列表，如 ["ROOM_001", "ROOM_002"]
            transformer: 坐标变换器
            graph_builder: 邻接图构建器
            landmark_builder: 标记点管理器

        Returns:
            Dict[str, Any]: labels_json 格式
        """
        # step1 分割房间数据和标记点数据
        rooms_data, landmarks_data = split_labels_data(map_data["labels_json"])
        original_rooms_data = [r.copy() for r in rooms_data]

        # step2 校验参数并检查连通性
        merge_indices = self._extract_merge_params(rooms_data, roomid_merge_list)
        self._check_connectivity(rooms_data, merge_indices, transformer, map_data["map_img"])

        # step3 对合并列表进行合并
        new_rooms_data, merged_idx = self.merge_rooms(rooms_data, merge_indices)

        # step4 邻接图 + 五色地图
        new_rooms_data, contours_list, graph = self.build_graph_and_colors(
            new_rooms_data, transformer, graph_builder,
            map_data["map_img"], merged_idx
        )

        # step5 美化框
        if self.beautifier_status:
            pass  # TODO: 美化框

        # step6 平台点标记点
        new_landmarks_data = self.build_landmarks(
            landmarks_data=landmarks_data,
            new_rooms_data=new_rooms_data,
            merged_room_idx=merged_idx,
            graph=graph,
            contours_list=contours_list,
            world_charge_pose=map_data["world_charge_pose"],
            transformer=transformer,
            graph_builder=graph_builder,
            landmark_builder=landmark_builder,
        )

        logger.info(f">>>> old rooms: {len(original_rooms_data)}, "
                     f"new rooms: {len(new_rooms_data)}")
        logger.info(f">>>> old landmarks: {len(landmarks_data)}, "
                     f"new landmarks: {len(new_landmarks_data)}")

        # step7 回写结果
        labels = {
            "version": self.config.get("labels_version", f"v{self.config.get('service_version', '4.0.2')}"),
            "uuid": map_data["uuid"],
            "data": new_rooms_data + new_landmarks_data,
        }

        return labels
