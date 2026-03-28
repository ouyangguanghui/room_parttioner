"""RoomService —— 房间分区总接口

AWS Lambda handler 直接调用本模块，统一管理四种核心操作：
  - split:       首次分区 (无 labels) → AutoPartitioner
                 增量扩展 (有 labels) → ExtendedPartitioner
  - repartition: 强制重新分区         → AutoPartitioner(repartition=True)
  - division:    手动划分             → ManualPartitioner
  - merge:       手动合并             → ManualMerger

预处理 (Preprocessor) 也在本模块内完成，各服务 process() 接收统一的
map_data + meta + 公共工具 (transformer, graph_builder, landmark_builder)。
"""

import logging
from typing import Dict, Any, List, Optional

import cv2
import numpy as np

from app.core.config import load_config
from app.core.errors import (
    InvalidResolutionError,
    NoLabelsError,
    OperationFailedError,
)
from app.pipeline.preprocessor import Preprocessor
from app.services.auto_partition import AutoPartitioner
from app.services.extended_partition import ExtendedPartitioner
from app.services.manual_partition import ManualPartitioner
from app.services.manual_merge import ManualMerger
from app.utils.coordinate import CoordinateTransformer
from app.utils.graph import RoomGraph
from app.utils.landmark import LandmarkManager

logger = logging.getLogger(__name__)


class RoomService:
    """
    房间分区总接口

    用法::

        config = load_config()
        service = RoomService(config)
        labels_json = service.room_edit(
            map_data,
            operation="split",
        )
    """

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config if config is not None else load_config()

        # 预处理器
        self.preprocessor = Preprocessor(self.config)

        # 四个核心服务
        self.auto_partitioner = AutoPartitioner(self.config)
        self.extended_partitioner = ExtendedPartitioner(self.config)
        self.manual_partitioner = ManualPartitioner(self.config)
        self.manual_merger = ManualMerger(self.config)

    # ==================== 总入口 ====================

    def room_edit(
        self,
        map_data: Dict[str, Any],
        operation: str,
        division_croods_dict: Optional[Dict[str, Any]] = None,
        room_merge_list: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        总入口 —— Lambda handler 直接调用

        Args:
            map_data: S3DataLoader.load() 返回的字典
                "map_img": (H, W, 3) uint8
                "resolution": float
                "origin": [x, y]
                "labels_json": Dict 或 None
                "robot_model": str
                "uuid": str
                "markers_json": Dict 或 None
                "world_charge_pose": [x, y, z]
            operation: "split" | "repartition" | "division" | "merge"
            division_croods_dict: division 操作所需 {"id", "A", "B"}
            room_merge_list: merge 操作所需 ["ROOM_001", "ROOM_002"]

        Returns:
            labels_json 格式 {"version", "uuid", "data": [...]}

        Raises:
            InvalidResolutionError: resolution 为 0
            NoLabelsError: division/merge 操作缺少 labels
            RoomPartitionerError 子类: 各服务内部业务错误
        """
        # step1: 校验基础参数
        logger.info(f"step1 : start check base parameters")
        resolution = map_data.get("resolution", 0)
        if not resolution or resolution <= 0:
            raise InvalidResolutionError()
        logger.info(f">>>>>>>> step1 : check base parameters success")
        labels_json = map_data.get("labels_json")
        logger.info(f">>>>>>>> step1 : check labels_json success")
        # step2: 预处理

        logger.info(f">>>>>>>> step2 : start preprocess")
        meta = self._preprocess(map_data["map_img"])
        logger.info(f">>>>>>>> step2 : preprocess success, operation={operation}")

        # step3: 创建公共工具
        logger.info(f">>>>>>>> step3 : start create public tools")
        map_img = map_data["map_img"]
        height = map_img.shape[0]
        origin = map_data.get("origin", [0, 0])
        map_data["cleaned_img"] = meta["cleaned_img"]
        map_data["cleaned_img2"] = meta["cleaned_img2"]
        map_data["input_img"] = meta["input_img"]

        transformer = CoordinateTransformer(resolution, origin, height)
        graph_builder = RoomGraph(self.config)
        landmark_builder = LandmarkManager(self.config)
        logger.info(f">>>>>>>> step3 : create public tools success")
        # step4: 按操作路由
        if operation == "split":
            return self._handle_split(
                map_data, labels_json,
                transformer, graph_builder, landmark_builder,
            )

        if operation == "repartition":
            return self._handle_repartition(
                map_data, transformer, graph_builder, landmark_builder,
            )

        if operation == "division":
            return self._handle_division(
                map_data, labels_json, division_croods_dict,
                transformer, graph_builder, landmark_builder,
            )

        if operation == "merge":
            return self._handle_merge(
                map_data, labels_json, room_merge_list,
                transformer, graph_builder, landmark_builder,
            )

        raise OperationFailedError(f"不支持的操作: {operation}")

    # ==================== 预处理 ====================

    def _preprocess(self, map_img: np.ndarray) -> Dict[str, Any]:
        """
        地图预处理：BGR → 灰度 → Preprocessor.process()

        Returns:
            {"cleaned_img": ..., "cleaned_img2": ..., "input_img": ...}
        """
        if map_img.ndim == 3:
            gray = cv2.cvtColor(map_img, cv2.COLOR_BGR2GRAY)
        else:
            gray = map_img.copy()

        return self.preprocessor.process(gray)

    # ==================== 操作路由 ====================

    def _handle_split(
        self,
        map_data: Dict[str, Any],
        labels_json: Optional[Dict[str, Any]],
        transformer: CoordinateTransformer,
        graph_builder: RoomGraph,
        landmark_builder: LandmarkManager,
    ) -> Dict[str, Any]:
        """
        split 操作：
        - 无 labels → AutoPartitioner (首次分区)
        - 有 labels → ExtendedPartitioner (增量扩展)
        """
        has_labels = (
            labels_json
            and isinstance(labels_json.get("data"), list)
            and len(labels_json["data"]) > 0
        )

        if has_labels:
            logger.info("split: 已有 labels, 走扩展分区")
            return self.extended_partitioner.process(
                map_data, transformer, graph_builder, landmark_builder,
            )

        logger.info("split: 无 labels, 走自动分区")
        return self.auto_partitioner.process(
            map_data, transformer, graph_builder, landmark_builder,
            repartition=False,
        )

    def _handle_repartition(
        self,
        map_data: Dict[str, Any],
        transformer: CoordinateTransformer,
        graph_builder: RoomGraph,
        landmark_builder: LandmarkManager,
    ) -> Dict[str, Any]:
        """repartition 操作：强制重新分区 (忽略已有 labels)"""
        logger.info("repartition: 强制重新分区")
        return self.auto_partitioner.process(
            map_data, transformer, graph_builder, landmark_builder,
            repartition=True,
        )

    def _handle_division(
        self,
        map_data: Dict[str, Any],
        labels_json: Optional[Dict[str, Any]],
        division_croods_dict: Optional[Dict[str, Any]],
        transformer: CoordinateTransformer,
        graph_builder: RoomGraph,
        landmark_builder: LandmarkManager,
    ) -> Dict[str, Any]:
        """division 操作：手动划分单个房间"""
        if not labels_json:
            raise NoLabelsError()

        return self.manual_partitioner.process(
            map_data, division_croods_dict, transformer,
            graph_builder, landmark_builder,
        )

    def _handle_merge(
        self,
        map_data: Dict[str, Any],
        labels_json: Optional[Dict[str, Any]],
        room_merge_list: Optional[List[str]],
        transformer: CoordinateTransformer,
        graph_builder: RoomGraph,
        landmark_builder: LandmarkManager,
    ) -> Dict[str, Any]:
        """merge 操作：手动合并多个房间"""
        if not labels_json:
            raise NoLabelsError()

        return self.manual_merger.process(
            map_data, room_merge_list, transformer,
            graph_builder, landmark_builder,
        )
