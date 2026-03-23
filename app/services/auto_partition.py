"""自动分区模块"""

from typing import Dict, Any, List, Optional
import numpy as np

from app.pipeline.preprocessor import Preprocessor
from app.pipeline.inferencer import Inferencer
from app.pipeline.postprocessor import Postprocessor


class AutoPartitioner:
    """
    自动分区：前处理 → 模型推理 → 后处理 完整流水线

    当 Triton 服务不可用时，fallback 到传统连通域方法
    """

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}

        self.preprocessor = Preprocessor(self.config)
        self.postprocessor = Postprocessor(self.config)

        # Triton 推理（可选）
        self.inferencer: Optional[Inferencer] = None
        if self.config.get("triton_url"):
            self.inferencer = Inferencer(self.config)

        self.wall_threshold = self.config.get("wall_threshold", 128)

    def run(self, image: np.ndarray) -> np.ndarray:
        """
        自动分区

        Args:
            image: 原始栅格地图 (H, W) 灰度图

        Returns:
            label_map: 房间标签图 (H, W) int32, 0=背景
        """
        # 优先模型推理
        if self.inferencer and self.inferencer.is_ready():
            tensor, meta = self.preprocessor.process(image)
            raw_output = self.inferencer.run(tensor)
            label_map = self.postprocessor.process(raw_output, meta)
        else:
            label_map = self._fallback_partition(image)

        return label_map

    def _fallback_partition(self, image: np.ndarray) -> np.ndarray:
        """传统连通域分割（Triton 不可用时的兜底方案）"""
        import cv2
        binary = (image < self.wall_threshold).astype(np.uint8)
        num_labels, labels = cv2.connectedComponents(binary)

        # 复用后处理的小区域过滤
        labels = self.postprocessor._filter_small_rooms(labels.astype(np.int32))
        return labels

    @staticmethod
    def get_room_info(label_map: np.ndarray, resolution: float = 0.05) -> List[Dict[str, Any]]:
        """获取各房间统计信息"""
        rooms = []
        for lid in range(1, label_map.max() + 1):
            mask = label_map == lid
            if not mask.any():
                continue
            ys, xs = np.where(mask)
            rooms.append({
                "id": int(lid),
                "area": float(mask.sum() * resolution ** 2),
                "center": (float(xs.mean()), float(ys.mean())),
                "bbox": (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())),
            })
        return rooms
