"""房间编辑器 —— 对接 Lambda handler, 薄编排层

职责:
1. 从 S3 加载数据
2. 路由操作到对应服务
3. 返回 labels_json (成功) / 抛出异常 (失败)
"""

import time
import logging
from typing import Dict, Any, List, Optional

import numpy as np
import cv2

from app.core.config import load_config
from app.core.partitioner import RoomPartitioner
from app.core.errors import (
    DataLoadError,
    InvalidResolutionError,
    NoLabelsError,
    InvalidParameterError,
    OperationFailedError,
)
from app.utils.s3_loader import S3DataLoader
from app.utils.coordinate import CoordinateTransformer
from app.utils.serializer import LabelSerializer
from app.utils.graph import RoomGraph
from app.services.geometry_splitter import GeometrySplitter
from app.services.geometry_merger import GeometryMerger

logger = logging.getLogger(__name__)

VERSION = "online_4.0.2"


class RoomEditor:
    """
    房间编辑器: Lambda handler 的业务入口

    支持操作:
    - split:       首次自动分割
    - repartition: 清空重新分割
    - division:    用户画线分割
    - merge:       合并房间
    """

    def __init__(self, bucket: str, key: str):
        self.loader = S3DataLoader(bucket, key)
        self.config = load_config()
        self.partitioner = RoomPartitioner(self.config)
        self.splitter = GeometrySplitter(self.config)
        self.geometry_merger = GeometryMerger(self.config)

        # 运行时数据 (load 后填充)
        self.map_img: Optional[np.ndarray] = None
        self.resolution: float = 0.05
        self.origin: List[float] = [0.0, 0.0]
        self.labels_json: Optional[Dict] = None
        self.robot_model: str = "s10"
        self.uuid: Optional[str] = None
        self.markers_json: Optional[Dict] = None
        self.world_charge_pose: List[float] = [0, 0, 0]

    # ==================== 数据加载 ====================

    def _load_data(self):
        """从 S3 加载所有数据"""
        try:
            data = self.loader.load()
            self.map_img = data["map_img"]
            self.resolution = data["resolution"]
            self.origin = data["origin"]
            self.labels_json = data["labels_json"]
            self.robot_model = data["robot_model"]
            self.uuid = data["uuid"]
            self.markers_json = data["markers_json"]
            self.world_charge_pose = data["world_charge_pose"]

            # 更新 config
            self.config["resolution"] = self.resolution
        except Exception as e:
            logger.error(f"加载数据失败: {e}")
            raise DataLoadError(str(e))

    # ==================== 主入口 ====================

    def room_edit(
        self,
        operation: str,
        division_croods_dict: Dict = None,
        room_merge_list: List = None,
    ) -> Dict[str, Any]:
        """
        主编辑入口

        Args:
            operation: "split" | "repartition" | "division" | "merge"
            division_croods_dict: 分割参数 {"id": "ROOM_001", "A": [x,y], "B": [x,y]}
            room_merge_list: 合并列表 ["ROOM_001", "ROOM_002"]

        Returns:
            labels_json dict

        Raises:
            RoomPartitionerError 子类
        """
        self._load_data()

        if self.resolution == 0:
            raise InvalidResolutionError()

        if operation in ('split', 'repartition'):
            return self._room_detect(repartition=(operation == 'repartition'))

        if operation == 'division':
            if not self.labels_json:
                raise NoLabelsError()
            if not division_croods_dict or len(division_croods_dict) != 3:
                raise InvalidParameterError()
            return self._room_divide(division_croods_dict)

        if operation == 'merge':
            if not self.labels_json:
                raise NoLabelsError()
            if not room_merge_list or len(room_merge_list) == 1:
                raise InvalidParameterError()
            return self._room_merge(room_merge_list)

        raise InvalidParameterError(f"未知操作: {operation}")

    # ==================== 自动检测 ====================

    def _room_detect(self, repartition: bool = False) -> Dict[str, Any]:
        """自动房间分割"""
        t0 = time.time()
        h, w = self.map_img.shape[:2]
        logger.info(f"robot_model: {self.robot_model}")

        need_detect = not self.labels_json or repartition

        if need_detect:
            logger.info("重新分区" if repartition else "首次分区 (labels 为空)")

            # 初始化 labels
            if self.labels_json is None:
                self.labels_json = {"data": []}
            self.labels_json["data"] = []

            # 自动分区
            gray = cv2.cvtColor(self.map_img, cv2.COLOR_BGR2GRAY)
            self.partitioner.auto_partition(gray, extend=True)
        else:
            logger.info("扩展分区 (已有 labels)")
            # 从已有 labels 恢复轮廓
            transformer = self._make_transformer()
            serializer = LabelSerializer(transformer)
            existing_contours = serializer.deserialize_contours(self.labels_json)

            gray = cv2.cvtColor(self.map_img, cv2.COLOR_BGR2GRAY)
            # 用已有轮廓构建 label_map
            label_map = np.zeros((h, w), dtype=np.int32)
            for i, cnt in enumerate(existing_contours):
                cv2.drawContours(label_map, [cnt], -1, i + 1, -1)

            self.partitioner.load_state(label_map, gray)
            self.partitioner.extend_partition()

        # 轮廓外扩
        self.partitioner.expand_contours(self.map_img)

        # 序列化输出
        charge_pixel = self._charge_to_pixel()
        labels = self.partitioner.serialize(
            resolution=self.resolution,
            origin=self.origin,
            uuid=self.uuid,
            robot_model=self.robot_model,
            existing_labels=self.labels_json if not need_detect else None,
            marker_polygons=self._get_marker_polygons(),
        )

        # 排序 (k20 从充电桩开始)
        if self.robot_model == "S-K20PRO" and charge_pixel:
            self.partitioner.sort_rooms(charge_pixel)
            labels = self.partitioner.serialize(
                resolution=self.resolution,
                origin=self.origin,
                uuid=self.uuid,
                robot_model=self.robot_model,
                existing_labels=self.labels_json if not need_detect else None,
                marker_polygons=self._get_marker_polygons(),
            )

        self.labels_json = labels
        logger.info(f"分区完成, 耗时 {time.time() - t0:.2f}s")
        return self.labels_json

    # ==================== 画线分割 ====================

    def _room_divide(self, division_croods_dict: Dict) -> Dict[str, Any]:
        """用户画线分割房间 —— 委托给 GeometrySplitter"""
        t0 = time.time()
        transformer = self._make_transformer()

        rooms_data = [d for d in self.labels_json['data'] if 'ROOM' in d.get('id', '')]

        rooms_data, select_idx = self.splitter.split(
            rooms_data, division_croods_dict,
            transformer, self.map_img, self.resolution,
        )

        # 重建图 + 着色
        self.labels_json['data'] = rooms_data
        self._rebuild_graph_and_colors(transformer, select_idx)

        self.labels_json['version'] = VERSION
        self.labels_json['uuid'] = self.uuid
        logger.info(f"分割完成, 耗时 {time.time() - t0:.2f}s")
        return self.labels_json

    # ==================== 合并 ====================

    def _room_merge(self, room_merge_list: List[str]) -> Dict[str, Any]:
        """合并房间 —— 委托给 GeometryMerger"""
        t0 = time.time()
        transformer = self._make_transformer()

        rooms_data = [d for d in self.labels_json['data'] if 'ROOM' in d.get('id', '')]

        rooms_data, select_idx = self.geometry_merger.merge(
            rooms_data, room_merge_list,
            transformer, self.map_img,
        )

        # 重建图 + 着色
        self.labels_json['data'] = rooms_data
        self._rebuild_graph_and_colors(transformer, select_idx)

        self.labels_json['version'] = VERSION
        self.labels_json['uuid'] = self.uuid
        logger.info(f"合并完成, 耗时 {time.time() - t0:.2f}s")
        return self.labels_json

    # ==================== 内部工具 ====================

    def _make_transformer(self) -> CoordinateTransformer:
        h = self.map_img.shape[0]
        return CoordinateTransformer(self.resolution, self.origin, h)

    def _charge_to_pixel(self) -> Optional[tuple]:
        """充电桩世界坐标 → 像素坐标"""
        if self.world_charge_pose == [0, 0, 0]:
            return None
        t = self._make_transformer()
        return t.world_to_pixel(self.world_charge_pose[0],
                                self.world_charge_pose[1])

    def _get_marker_polygons(self) -> Optional[List]:
        """从 markers_json 提取家具标记多边形"""
        if not self.markers_json:
            return None
        polys = []
        for item in self.markers_json.get('data', []):
            if item.get('name') == '家具' and item.get('type') == 'polygon':
                polys.append(item['geometry'])
        return polys if polys else None

    def _rebuild_graph_and_colors(self, transformer: CoordinateTransformer,
                                  select_idx: int):
        """重建邻接图和颜色 (分割/合并后)"""
        rooms_data = [d for d in self.labels_json['data'] if 'ROOM' in d.get('id', '')]
        pixel_contours = [
            transformer.world_to_contour(r['geometry']) for r in rooms_data
        ]

        # 重建图
        graph_builder = RoomGraph(self.config)
        for i, cnt1 in enumerate(pixel_contours):
            neighbors = []
            for j, cnt2 in enumerate(pixel_contours):
                if i != j and graph_builder.check_connectivity(cnt1, cnt2, self.map_img):
                    neighbors.append(j)
            rooms_data[i]['graph'] = neighbors

        # 着色
        graph = {i: rooms_data[i]['graph'] for i in range(len(rooms_data))}
        colors = graph_builder.assign_colors(graph)
        for i in range(len(rooms_data)):
            if rooms_data[i]['colorType'] is None or i == select_idx:
                rooms_data[i]['colorType'] = colors.get(i, 0)

        self.labels_json['data'] = rooms_data
