"""自动分区模块"""

from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import cv2

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

    def partition_pixel(self, image: np.ndarray) -> np.ndarray:
        """
        像素级自动分区主入口。

        Args:
            image: 原始栅格地图 (H, W) 灰度图

        Returns:
            label_map: 房间标签图 (H, W) int32, 0=背景
        """
        if self.inferencer and self.inferencer.is_ready():
            meta = self.preprocessor.process(image)
            tensor = self._prepare_tensor(meta["map_data"], meta)
            raw_output = self.inferencer.run(tensor)
            label_map = self.postprocessor.process(raw_output, meta)
        else:
            label_map = self._fallback_partition(image)

        return label_map

    def run(self, image: np.ndarray) -> np.ndarray:
        """兼容旧接口：等价于 partition_pixel。"""
        return self.partition_pixel(image)

    def _prepare_tensor(
        self,
        map_data: np.ndarray,
        meta: Dict[str, Any],
    ) -> np.ndarray:
        """
        将预处理后的灰度地图转换为 Triton 模型输入张量。

        流程:
            1. 灰度 → BGR (3 通道复制)
            2. Letterbox resize 到 target_size，保持长宽比，填充 114 灰色
            3. uint8 → float32 归一化 [0, 1]，可选 ImageNet 均值/方差标准化
            4. HWC → NCHW

        同时将 scale / pad 写入 meta，供 postprocessor 做坐标逆映射。

        Args:
            map_data: 前处理后的灰度地图 (H, W) uint8
            meta:     由 preprocessor 生成的 meta 字典（本方法会向其追加 key）

        Returns:
            tensor: (1, 3, target_h, target_w) float32
        """
        target_size: List[int] = self.config.get("target_size", [512, 512])
        normalize: bool = self.config.get("normalize", True)
        th, tw = target_size[0], target_size[1]

        # 灰度 → BGR
        if map_data.ndim == 2:
            img_bgr = cv2.cvtColor(map_data, cv2.COLOR_GRAY2BGR)
        else:
            img_bgr = map_data.copy()

        h, w = img_bgr.shape[:2]

        # Letterbox: 等比缩放，短边 pad 至 target
        scale = min(tw / w, th / h)
        nw, nh = int(round(w * scale)), int(round(h * scale))
        resized = cv2.resize(img_bgr, (nw, nh), interpolation=cv2.INTER_LINEAR)

        pad_top = (th - nh) // 2
        pad_left = (tw - nw) // 2
        canvas = np.full((th, tw, 3), 114, dtype=np.uint8)
        canvas[pad_top:pad_top + nh, pad_left:pad_left + nw] = resized

        # 归一化
        tensor = canvas.astype(np.float32) / 255.0
        if normalize:
            mean = np.array(self.config.get("mean", [0.485, 0.456, 0.406]), dtype=np.float32)
            std  = np.array(self.config.get("std",  [0.229, 0.224, 0.225]), dtype=np.float32)
            tensor = (tensor - mean) / std

        # HWC → NCHW
        tensor = np.ascontiguousarray(tensor.transpose(2, 0, 1)[np.newaxis, ...], dtype=np.float32)

        # 将坐标映射参数写入 meta，供 postprocessor 逆映射用
        meta["tensor_scale"] = scale
        meta["tensor_pad"]   = (pad_top, pad_left)
        meta["tensor_size"]  = (th, tw)

        return tensor

    def _fallback_partition(self, image: np.ndarray) -> np.ndarray:
        """传统连通域分割（Triton 不可用时的兜底方案）"""
        meta = self.preprocessor.process(image)
        free = (meta["map_data"] >= 200).astype(np.uint8)
        _, labels = cv2.connectedComponents(free, connectivity=8)
        labels = labels.astype(np.int32)
        normal_map, small_map = self.postprocessor._split_by_area(labels, meta["map_data"])
        labels = self.postprocessor._merge_fragments(normal_map, small_map)
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
