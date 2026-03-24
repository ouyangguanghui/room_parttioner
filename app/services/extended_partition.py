"""扩展分区模块"""

from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import cv2


class ExtendedPartitioner:
    """
    扩展分区：在自动分区结果基础上进行进一步优化

    功能：
    - 过道/开放区域的二次分割
    - 基于门口检测的房间拆分
    - 区域生长扩展未分配像素
    """

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.door_width = self.config.get("door_width", 20)  # 门口宽度阈值 (像素)
        self.grow_iterations = self.config.get("grow_iterations", 10)

    def split_by_doorway(self, label_map: np.ndarray, grid_map: np.ndarray) -> np.ndarray:
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
        free_space = grid_map >= self.config.get("wall_threshold", 128)
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

    def extend(self, label_map: np.ndarray, grid_map: np.ndarray) -> np.ndarray:
        """
        完整扩展分区流程

        1. 门口检测拆分
        2. 未分配区域生长
        """
        result = self.split_by_doorway(label_map, grid_map)
        result = self.grow_unassigned(result, grid_map)
        return result

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
