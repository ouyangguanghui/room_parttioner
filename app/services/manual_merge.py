"""手动合并模块"""

from typing import Dict, Any, List, Tuple
import numpy as np


class ManualMerger:
    """
    手动合并：将多个房间区域合并为一个

    支持的操作：
    - 指定ID合并：合并指定的房间ID列表
    - 点选合并：点击两个相邻房间进行合并
    - 标签重编号：合并后整理标签序号
    """

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}

    def merge_rooms(
        self,
        label_map: np.ndarray,
        room_ids: List[int],
        target_id: int = -1,
    ) -> np.ndarray:
        """
        合并多个房间

        Args:
            label_map: 当前标签图
            room_ids: 要合并的房间ID列表
            target_id: 合并后的ID，-1 表示使用列表中最小的ID

        Returns:
            更新后的标签图
        """
        if len(room_ids) < 2:
            return label_map

        result = label_map.copy()
        if target_id == -1:
            target_id = min(room_ids)

        for rid in room_ids:
            if rid != target_id:
                result[result == rid] = target_id

        return result

    def merge_by_point(
        self,
        label_map: np.ndarray,
        pt1: Tuple[int, int],
        pt2: Tuple[int, int],
    ) -> np.ndarray:
        """
        点选合并：通过两个点所在的房间进行合并

        Args:
            pt1: 第一个点 (x, y)
            pt2: 第二个点 (x, y)
        """
        id1 = label_map[pt1[1], pt1[0]]
        id2 = label_map[pt2[1], pt2[0]]

        if id1 == 0 or id2 == 0:
            return label_map  # 点在背景上，不合并
        if id1 == id2:
            return label_map  # 同一房间

        return self.merge_rooms(label_map, [int(id1), int(id2)])

    def merge_adjacent(
        self,
        label_map: np.ndarray,
        room_id: int,
    ) -> np.ndarray:
        """
        合并指定房间与所有相邻房间

        Args:
            room_id: 目标房间ID，将与其所有相邻房间合并
        """
        mask = (label_map == room_id).astype(np.uint8)

        import cv2
        dilated = cv2.dilate(mask, np.ones((3, 3), np.uint8))
        border = (dilated > 0) & (mask == 0) & (label_map > 0)
        neighbors = set(label_map[border].flatten())
        neighbors.discard(0)

        if not neighbors:
            return label_map

        all_ids = [room_id] + list(neighbors)
        return self.merge_rooms(label_map, all_ids, target_id=room_id)

    @staticmethod
    def relabel(label_map: np.ndarray) -> np.ndarray:
        """重新编号标签：确保ID从1开始连续"""
        result = np.zeros_like(label_map)
        unique_ids = sorted(set(label_map.flatten()))
        new_id = 0
        for old_id in unique_ids:
            if old_id == 0:
                continue
            new_id += 1
            result[label_map == old_id] = new_id
        return result
