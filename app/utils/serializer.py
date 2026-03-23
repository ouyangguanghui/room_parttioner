"""标签数据序列化 —— 生成 labels.json 格式的输出"""

from typing import Dict, Any, List, Optional

import numpy as np

from app.utils.coordinate import CoordinateTransformer


class LabelSerializer:
    """
    将像素级的房间分区结果序列化为 labels.json 格式

    labels.json 结构:
    {
        "version": "...",
        "uuid": "...",
        "data": [
            {
                "name": "A",
                "id": "ROOM_001",
                "type": "polygon",
                "geometry": [x0, y0, x1, y1, ..., x0, y0],
                "colorType": 0,
                "graph": [1, 2],
                "groundMaterial": null
            },
            ...
        ]
    }
    """

    VERSION = "online_4.0.2"

    def __init__(self, transformer: CoordinateTransformer):
        self.transformer = transformer

    def serialize(
        self,
        contours: List[np.ndarray],
        graph: Dict[int, List[int]],
        colors: Dict[int, int],
        room_order: List[int] = None,
        existing_labels: Dict[str, Any] = None,
        uuid: str = None,
        bbox_list: List[List] = None,
        threshold_list: List[List] = None,
    ) -> Dict[str, Any]:
        """
        序列化房间数据

        Args:
            contours: 各房间轮廓 [(N,1,2), ...]
            graph: 邻接图 {room_idx: [neighbors]}
            colors: 颜色分配 {room_idx: color_id}
            room_order: 房间排列顺序 (None=按索引)
            existing_labels: 已有的 labels (保留 name / groundMaterial)
            uuid: 设备 UUID
            bbox_list: 美化框列表 (s10)
            threshold_list: 门槛线列表 (s10)

        Returns:
            labels.json 格式的 dict
        """
        if room_order is None:
            room_order = list(range(len(contours)))

        existing_data = []
        if existing_labels and "data" in existing_labels:
            existing_data = [
                item for item in existing_labels["data"]
                if "ROOM" in item.get("id", "")
            ]

        data = []
        for new_idx, cnt_idx in enumerate(room_order):
            cnt = contours[cnt_idx]
            geometry = self.transformer.contour_to_geometry(cnt)

            room_data = {
                "name": chr(ord('A') + new_idx),
                "id": f"ROOM_{new_idx + 1:03d}",
                "type": "polygon",
                "geometry": geometry,
                "colorType": colors.get(cnt_idx, 0),
                "graph": self._remap_graph(graph.get(cnt_idx, []), room_order),
                "groundMaterial": None,
            }

            # 保留已有属性
            if new_idx < len(existing_data):
                old = existing_data[new_idx]
                room_data["groundMaterial"] = old.get("groundMaterial")
                if old.get("name"):
                    room_data["name"] = old["name"]

            # 美化框 (s10)
            if bbox_list and cnt_idx < len(bbox_list) and bbox_list[cnt_idx]:
                bbox_world = [
                    coord
                    for pt in bbox_list[cnt_idx]
                    for coord in self.transformer.pixel_to_world(pt[0], pt[1])
                ]
                room_data["test_bbox"] = bbox_world

            if threshold_list and cnt_idx < len(threshold_list) and threshold_list[cnt_idx]:
                world_thresholds = []
                for threshold in threshold_list[cnt_idx]:
                    wt = [
                        list(self.transformer.pixel_to_world(pt[0], pt[1]))
                        for pt in threshold
                    ]
                    if wt:
                        world_thresholds.append(wt)
                room_data["test_threshold"] = world_thresholds

            data.append(room_data)

        result = {
            "version": self.VERSION,
            "uuid": uuid,
            "data": data,
        }
        return result

    def _remap_graph(self, neighbors: List[int],
                     room_order: List[int]) -> List[int]:
        """将旧索引映射到新排序后的索引"""
        order_map = {old: new for new, old in enumerate(room_order)}
        return sorted(order_map[n] for n in neighbors if n in order_map)

    def deserialize_contours(
        self, labels_json: Dict[str, Any]
    ) -> List[np.ndarray]:
        """
        从 labels.json 反序列化轮廓 (世界坐标 → 像素坐标)

        Returns:
            轮廓列表 [(N,1,2), ...]
        """
        contours = []
        for item in labels_json.get("data", []):
            if "ROOM" not in item.get("id", ""):
                continue
            geometry = item.get("geometry", [])
            if len(geometry) < 6:
                continue
            cnt = self.transformer.world_to_contour(geometry)
            contours.append(cnt)
        return contours
