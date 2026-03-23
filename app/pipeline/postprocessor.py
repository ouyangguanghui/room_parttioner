"""模型后处理模块"""

from typing import Dict, Any, Tuple
import numpy as np
import cv2


class Postprocessor:
    """模型输出后处理：将原始推理结果转为房间标签图"""

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.min_room_area = self.config.get("min_room_area", 1.0)  # m²
        self.resolution = self.config.get("resolution", 0.05)  # m/pixel
        self.morph_kernel_size = self.config.get("morph_kernel_size", 5)

    def process(self, raw_output: np.ndarray, meta: Dict[str, Any]) -> np.ndarray:
        """
        后处理流水线

        Args:
            raw_output: 模型原始输出 (N,C,H,W) 或 (N,H,W)
            meta: 前处理返回的元信息

        Returns:
            label_map: 原始尺寸的房间标签图 (H, W) int32
        """
        # argmax 得到标签
        if raw_output.ndim == 4:
            label_map = np.argmax(raw_output[0], axis=0).astype(np.int32)
        elif raw_output.ndim == 3:
            label_map = raw_output[0].astype(np.int32)
        else:
            label_map = raw_output.astype(np.int32)

        # 形态学优化：去噪 + 平滑边界
        label_map = self._morphology_refine(label_map)

        # 还原到原始尺寸
        orig_h, orig_w = meta["orig_shape"]
        if label_map.shape != (orig_h, orig_w):
            label_map = cv2.resize(label_map, (orig_w, orig_h),
                                   interpolation=cv2.INTER_NEAREST)

        # 过滤小区域
        label_map = self._filter_small_rooms(label_map)

        return label_map

    def _morphology_refine(self, label_map: np.ndarray) -> np.ndarray:
        """形态学优化：对每个标签做开运算去噪"""
        refined = np.zeros_like(label_map)
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (self.morph_kernel_size, self.morph_kernel_size)
        )
        for lid in range(1, label_map.max() + 1):
            mask = (label_map == lid).astype(np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            refined[mask > 0] = lid
        return refined

    def _filter_small_rooms(self, label_map: np.ndarray) -> np.ndarray:
        """过滤面积过小的区域"""
        filtered = label_map.copy()
        pixel_area = self.resolution ** 2
        for lid in range(1, label_map.max() + 1):
            mask = label_map == lid
            area = mask.sum() * pixel_area
            if area < self.min_room_area:
                filtered[mask] = 0
        return filtered
