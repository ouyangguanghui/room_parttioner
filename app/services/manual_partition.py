"""手动划分模块"""

from typing import Dict, Any, List, Tuple
import numpy as np
import cv2


class ManualPartitioner:
    """
    手动划分：用户指定分割线或区域，手动拆分房间

    支持的操作：
    - 画线分割：指定两点画分割线，将一个房间拆为两个
    - 多边形划分：指定多边形区域，标记为新房间
    - 点选分割：指定分割点序列，沿路径切割
    """

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.line_thickness = self.config.get("line_thickness", 3)

    def split_by_line(
        self,
        label_map: np.ndarray,
        pt1: Tuple[int, int],
        pt2: Tuple[int, int],
    ) -> np.ndarray:
        """
        画线分割：在两点之间画线，将被切割的房间拆分

        Args:
            label_map: 当前标签图
            pt1: 起点 (x, y)
            pt2: 终点 (x, y)

        Returns:
            更新后的标签图
        """
        result = label_map.copy()
        next_label = result.max() + 1

        # 在线段经过位置设为 0（切断）
        cut_mask = np.zeros(result.shape[:2], dtype=np.uint8)
        cv2.line(cut_mask, pt1, pt2, 1, self.line_thickness)

        # 找出被线段穿过的所有房间ID
        affected_ids = set(result[cut_mask > 0].flatten())
        affected_ids.discard(0)

        # 将切割线设为背景
        result[cut_mask > 0] = 0

        # 对每个受影响的房间重新连通域标记
        for lid in affected_ids:
            mask = (result == lid).astype(np.uint8)
            num, sub_labels = cv2.connectedComponents(mask)
            if num <= 2:
                continue
            # 保留最大连通域为原ID，其余分配新ID
            sizes = [(sub_labels == i).sum() for i in range(1, num)]
            keep_sub = np.argmax(sizes) + 1
            for sub_id in range(1, num):
                if sub_id == keep_sub:
                    continue
                result[sub_labels == sub_id] = next_label
                next_label += 1

        return result

    def split_by_polyline(
        self,
        label_map: np.ndarray,
        points: List[Tuple[int, int]],
    ) -> np.ndarray:
        """
        多段线分割：沿多个点的折线切割

        Args:
            label_map: 当前标签图
            points: 折线顶点列表 [(x1,y1), (x2,y2), ...]
        """
        result = label_map.copy()
        for i in range(len(points) - 1):
            result = self.split_by_line(result, points[i], points[i + 1])
        return result

    def assign_polygon(
        self,
        label_map: np.ndarray,
        polygon: List[Tuple[int, int]],
        room_id: int = -1,
    ) -> np.ndarray:
        """
        多边形划分：将指定多边形区域标记为新房间

        Args:
            label_map: 当前标签图
            polygon: 多边形顶点列表
            room_id: 指定房间ID，-1 表示自动分配新ID
        """
        result = label_map.copy()
        if room_id == -1:
            room_id = result.max() + 1

        mask = np.zeros(result.shape[:2], dtype=np.uint8)
        pts = np.array(polygon, dtype=np.int32).reshape((-1, 1, 2))
        cv2.fillPoly(mask, [pts], 1)

        result[mask > 0] = room_id
        return result
