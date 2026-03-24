"""几何级房间合并服务 —— 在世界坐标 geometry 上执行房间合并"""

import logging
from typing import Dict, Any, List, Tuple

import numpy as np
import cv2

from app.utils.coordinate import CoordinateTransformer
from app.core.errors import RoomsNotConnectedError

logger = logging.getLogger(__name__)


class GeometryMerger:
    """
    世界坐标级的房间合并

    用于 Lambda 场景：合并多个房间，操作直接在 labels_json 的
    geometry 多边形上进行（而非像素级 label_map）。
    """

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}

    def merge(
        self,
        rooms_data: List[Dict[str, Any]],
        room_merge_list: List[str],
        transformer: CoordinateTransformer,
        map_img: np.ndarray,
    ) -> Tuple[List[Dict[str, Any]], int]:
        """
        执行合并操作

        Args:
            rooms_data: labels_json 中的 ROOM 数据列表
            room_merge_list: 待合并房间 ID 列表 ["ROOM_001", "ROOM_002"]
            transformer: 坐标转换器
            map_img: 原始地图 (BGR)

        Returns:
            (更新后的 rooms_data, 合并后房间的索引)

        Raises:
            RoomsNotConnectedError
        """
        h, w = map_img.shape[:2]

        # 解析索引
        merge_indices = sorted(
            int(r.split("_")[-1]) - 1 for r in room_merge_list
        )

        # 合并轮廓
        mask = np.zeros((h, w), dtype=np.uint8)
        max_area = -1
        max_idx = merge_indices[0]

        for idx in merge_indices:
            cnt = transformer.world_to_contour(rooms_data[idx]['geometry'])
            cv2.drawContours(mask, [cnt], -1, 255, -1)
            area = cv2.contourArea(cnt)
            if area > max_area:
                max_area = area
                max_idx = idx

        # 连通性检查
        n_obj, _ = cv2.connectedComponents(mask)
        if n_obj != 2:
            raise RoomsNotConnectedError()

        # 去除墙壁区域, 提取合并后轮廓
        gray = cv2.cvtColor(map_img, cv2.COLOR_BGR2GRAY)
        merge_mask = mask.copy()
        merge_mask[mask != 255] = 127
        merge_mask[(gray == 0) & (mask == 0)] = 127
        merge_mask[(gray == 255) | (mask == 255)] = 255

        _, thresh = cv2.threshold(merge_mask, 235, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        merged_cnt = max(contours, key=cv2.contourArea)

        # 更新 labels
        min_idx = min(merge_indices)
        new_geom = transformer.contour_to_geometry(merged_cnt)
        rooms_data[min_idx]['geometry'] = new_geom

        # 保留最大房间的名称
        if min_idx != max_idx:
            rooms_data[min_idx]['name'] = rooms_data[max_idx].get(
                'name', rooms_data[min_idx]['name']
            )
            rooms_data[min_idx]['groundMaterial'] = rooms_data[max_idx].get(
                'groundMaterial'
            )

        # 删除被合并的 (从大到小删)
        for idx in sorted(merge_indices, reverse=True):
            if idx != min_idx:
                del rooms_data[idx]

        # 重编号
        for i, room in enumerate(rooms_data):
            room['id'] = f"ROOM_{i + 1:03d}"

        return rooms_data, min_idx
