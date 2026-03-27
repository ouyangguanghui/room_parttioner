"""手动划分模块。

当前阶段仅保留世界坐标(world)分割流程，作为模块内重构的中间态：
    - 入口：process(labels, division_croods_dict, map_img)
    - 核心：split_room(...)
"""

import logging
from typing import Dict, Any, List, Tuple
from cv2 import log
from shapely.geometry import Polygon

import numpy as np

from app.utils.coordinate import CoordinateTransformer
from app.utils.graph import RoomGraph
from app.utils.landmark import LandmarkManager
from app.core.errors import (
    InvalidParameterError,
    InsufficientIntersectionsError,
    RoomIndexOutOfRangeError,
    RoomTooSmallError,
)
from app.utils.geometry_ops import (
    get_room_index_by_id,
    split_labels_data,
    next_room_id,
    next_room_name,
    flatten_geometry,
    find_split_points,
)

logger = logging.getLogger(__name__)


class ManualPartitioner:
    """
    手动划分：
    用户指定世界坐标分割线，拆分 labels_json 中的目标房间。
    """

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.line_thickness = self.config.get("line_thickness", 1)
        self.beautifier_status = False


    def set_beautifier_status(self, status: bool):
        self.beautifier_status = status

    def _extract_split_params(
        self,
        rooms_data: List[Dict[str, Any]],
        division_croods_dict: Dict[str, Any],
    ) -> Tuple[int, List[Any], List[Any]]:
        """
        校验分割参数并提取目标房间索引与分割点。
        """
        if not isinstance(division_croods_dict, dict):
            raise InvalidParameterError("division_croods_dict 必须是 dict")
        for k in ("id", "A", "B"):
            if k not in division_croods_dict:
                raise InvalidParameterError(f"缺少参数: {k}")

        A = division_croods_dict["A"]
        B = division_croods_dict["B"]
        room_id = division_croods_dict["id"]

        if not isinstance(A, list) or not isinstance(B, list):
            raise InvalidParameterError("A/B 必须是列表")
        if len(A) != 2 or len(B) != 2:
            raise InvalidParameterError("A/B 必须是 [x, y]")
        if not isinstance(room_id, str):
            raise InvalidParameterError("id 必须是字符串")

        target_room_idx = get_room_index_by_id(rooms_data, room_id)
        if target_room_idx < 0:
            raise RoomIndexOutOfRangeError()

        return target_room_idx, A, B

    def split_room(
        self,
        rooms_data: List[Dict[str, Any]],
        target_room_idx: int,
        A: List[float],
        B: List[float],
    ) -> List[Dict[str, Any]]:
        """
        房间分割： 根据用户指定的分割线和房间ID，将房间拆分为两个房间。并更新房间数据。
        Args:
            rooms_data: 房间数据列表
            target_room_idx: 目标房间索引
            A: 分割线起点
            B: 分割线终点
            map_img: 地图图像

        Returns:
            Tuple[List[Dict[str, Any]], int]: 更新后的房间数据列表和新增房间的索引
        """

        geometry = rooms_data[target_room_idx]['geometry']
        if not geometry or len(geometry) < 8:
            raise InvalidParameterError("目标房间 geometry 非法")
        ok, result = find_split_points(A, B, geometry)
        if not ok:
            raise InsufficientIntersectionsError()

        poly_a, poly_b, _ = result

        area_a = float(Polygon(poly_a).area)
        area_b = float(Polygon(poly_b).area)
        if area_a < 0.25 or area_b < 0.25:
            raise RoomTooSmallError()

        geom_a = flatten_geometry(poly_a)
        geom_b = flatten_geometry(poly_b)

        new_id = next_room_id(rooms_data)
        new_name = next_room_name(rooms_data)

        if area_a >= area_b:
            rooms_data[target_room_idx]['geometry'] = geom_a
            rooms_data.append({
                "name": new_name,
                "id": new_id,
                "type": rooms_data[target_room_idx].get('type', 'polygon'),
                "geometry": geom_b,
                "colorType": None,
                "graph": None,
                "groundMaterial": rooms_data[target_room_idx].get('groundMaterial', None),
            })
        else:
            rooms_data[target_room_idx]['geometry'] = geom_b
            rooms_data.append({
                "name": new_name,
                "id": new_id,
                "type": rooms_data[target_room_idx].get('type', 'polygon'),
                "geometry": geom_a,
                "colorType": None,
                "graph": None,
                "groundMaterial": rooms_data[target_room_idx].get('groundMaterial', None),
            })

        return rooms_data

    # 邻接图 + 五色地图
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
        # 构建邻接图
        contours_list = transformer.rooms_data_to_contours(rooms_data)
        graph = graph_builder.build_graph(contours_list, map_img)
        
        logger.info(f">>>> graph: {graph}")
        for room_idx in range(len(rooms_data)):
            # 打印一下graph是否有变化
            old_graph_id = [rooms_data[i]['id'] for i in (rooms_data[room_idx].get('graph') or []) if i is not None]
            new_graph_id = [rooms_data[i]['id'] for i in (graph[room_idx] or []) if i is not None]
            logger.info(f">>>> room_id : {rooms_data[room_idx]['id']}： {old_graph_id} ---> {new_graph_id}")
            rooms_data[room_idx]["graph"] = graph[room_idx]
        
        # 着色
        # 取出所有房间的colorType
        current_colors = {room_idx: rooms_data[room_idx]["colorType"] for room_idx in range(len(rooms_data))}
        logger.info(f">>>> current_colors: {current_colors}")
        if rooms_data[target_room_idx]["colorType"] is None:
            logger.info(f">>> set target room colorType")
            target_room_color = graph_builder.assign_color_for_room(target_room_idx, graph, current_colors)
            logger.info(f">>>> target room colorType: {target_room_color}")
            rooms_data[target_room_idx]["colorType"] = target_room_color
        if rooms_data[len(rooms_data) - 1]["colorType"] is None:
            logger.info(f">>> set new room colorType")
            new_room_color = graph_builder.assign_color_for_room(len(rooms_data) - 1, graph, current_colors)
            logger.info(f">>>> new room colorType: {new_room_color}")
            rooms_data[len(rooms_data) - 1]["colorType"] = new_room_color
        return rooms_data, contours_list, graph


    def build_landmarks(
        self,
        landmarks_data: List[Dict[str, Any]],
        rooms_data: List[Dict[str, Any]],
        new_rooms_data: List[Dict[str, Any]],
        target_room_idx: int,
        graph: Dict[int, List[int]],
        contours_list: List[np.ndarray],
        world_charge_pose: List[float],
        transformer: CoordinateTransformer,
        graph_builder: RoomGraph,
        landmark_builder: LandmarkManager,
    ) -> List[Dict[str, Any]]:
        """
        构建平台点标记点
        """
        world_charge_pixel = transformer.world_to_pixel(world_charge_pose[0], world_charge_pose[1])
        start_room_idx = graph_builder.find_start_room(
            contours_list, world_charge_pixel, max_area_start=True
        )
        room_order = graph_builder.dfs_sort(graph, start_room_idx)

        # 兜底：确保每个房间都会被遍历到
        if not room_order:
            room_order = list(range(len(new_rooms_data)))
        else:
            missing = [idx for idx in range(len(new_rooms_data)) if idx not in room_order]
            room_order.extend(missing)

        old_landmarks_by_room = {}
        for landmark in landmarks_data:
            room_id = landmark.get("roomId")
            if room_id:
                old_landmarks_by_room.setdefault(room_id, []).append(landmark)

        new_landmarks_data = []
        for new_room_idx, old_room_idx in enumerate(room_order):
            old_room_id = rooms_data[old_room_idx]["id"]
            matched_landmarks = old_landmarks_by_room.get(old_room_id, [])
            if matched_landmarks and (
                old_room_idx != target_room_idx and old_room_idx != (len(rooms_data) - 1)
            ):
                for landmark in matched_landmarks:
                    new_landmarks_data.append({
                        "geometry": landmark["geometry"],
                        "id": f"PLATFORM_LANDMARK_{len(new_landmarks_data) + 1:03d}",
                        "roomId": new_rooms_data[new_room_idx]["id"],
                        "name": new_rooms_data[new_room_idx]["name"],
                        "type": "pose",
                    })
                continue

            markers_polygons = []  # TODO: 接入家具/标记物多边形避障
            new_pose = landmark_builder._find_center(
                new_rooms_data[new_room_idx]["geometry"],
                markers_polygons,
            )
            new_pose = [new_pose[0], new_pose[1], 0]
            new_landmarks_data.append({
                "geometry": new_pose,
                "id": f"PLATFORM_LANDMARK_{len(new_landmarks_data) + 1:03d}",
                "roomId": new_rooms_data[new_room_idx]["id"],
                "name": new_rooms_data[new_room_idx]["name"],
                "type": "pose",
            })

        return new_landmarks_data
    def process(self, 
                map_data: Dict[str, Any],
                division_croods_dict: Dict,
                transformer: CoordinateTransformer,
                graph_builder: RoomGraph,
                landmark_builder: LandmarkManager,

    ) -> Dict[str, Any]:
        """
        处理房间数据：根据用户指定的分割线或区域，手动拆分房间。
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
                
            division_croods_dict: 分割线坐标字典
            transformer: 坐标变换器
            graph_builder: 邻接图构建器
        Returns:
            List[Dict[str, Any]]: 更新后的房间数据列表
        """
        # step1 分割房间数据和标记点数据
        rooms_data, landmarks_data = split_labels_data(map_data["labels_json"])

        # step2 校验参数并提取目标房间与分割线端点
        target_room_idx, A, B = self._extract_split_params(rooms_data, division_croods_dict)

        # step3 对目标房间进行分割
        new_rooms_data = self.split_room(rooms_data, target_room_idx, A, B)

        # step4 邻接图 + 五色地图
        new_rooms_data, contours_list, graph = self.build_graph_and_colors(new_rooms_data, 
                                                                            transformer, 
                                                                            graph_builder, 
                                                                            map_data["map_img"], 
                                                                            target_room_idx)

        # step5 美化框
        if self.beautifier_status:
            pass # TODO: 美化框

        # step6 平台点标记点
        new_landmarks_data = self.build_landmarks(
            landmarks_data=landmarks_data,
            rooms_data=rooms_data,
            target_room_idx=target_room_idx,
            new_rooms_data=new_rooms_data,
            graph=graph,
            contours_list=contours_list,
            world_charge_pose=map_data["world_charge_pose"],
            transformer=transformer,
            graph_builder=graph_builder,
            landmark_builder=landmark_builder,
        )
        
        logger.info(f">>>> old rooms number: {len(rooms_data)}, new rooms number: {len(new_rooms_data)}")
        logger.info(f">>>> old landmarks number: {len(landmarks_data)}, new landmarks number: {len(new_landmarks_data)}")
            
        # step7 回写结果（先保持原始顺序：ROOM + LANDMARK）
        labels = {"version": self.config.get("labels_version", f"v{self.config.get('service_version', '4.0.2')}"), "uuid": map_data["uuid"], "data": new_rooms_data + new_landmarks_data}
    
        return labels
