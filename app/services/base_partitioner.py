"""分区器基类

提供 AutoPartitioner 和 ExtendedPartitioner 共享的能力：
  - 模型推理流程：prepare_tensor → inferencer.run → postprocessor.process
  - Fallback 连通域分割
  - 通用工具方法（轮廓提取、重编号、坐标转换等）
"""

import logging
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
import cv2

from app.pipeline.inferencer import Inferencer
from app.pipeline.postprocessor import Postprocessor
from app.utils.coordinate import CoordinateTransformer

logger = logging.getLogger(f"{__name__} [BasePartitioner]")


class BasePartitioner:
    """
    分区器基类

    子类：AutoPartitioner, ExtendedPartitioner
    """

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}

        # 内置处理器
        self.inferencer: Optional[Inferencer] = None
        if self.config.get("triton_url"):
            self.inferencer = Inferencer(self.config)
        self.postprocessor = Postprocessor(self.config)

        # 张量准备参数
        self.target_size: List[int] = self.config.get("target_size", [512, 512])
        self.min_input_size: int = self.config.get("min_input_size", 416)
        self.normalize: bool = self.config.get("normalize", True)
        self.mean = self.config.get("mean", [0.485, 0.456, 0.406])
        self.std = self.config.get("std", [0.229, 0.224, 0.225])

        self.beautifier_status = False

    def set_beautifier_status(self, status: bool):
        self.beautifier_status = status

    # ==================== 核心推理分区 ====================

    def partition(self, map_data: Dict[str, Any],
                  extend: bool = True) -> List:
        """
        核心分区: 推理 → 后处理 → room_polygons

        Args:
            map_data: 地图数据字典，至少包含:
                "input_img": 补墙平滑后的灰度地图 (H, W) uint8
            extend: 是否执行扩展分区

        Returns:
            room_polygons: 房间多边形列表
        """
        input_img = map_data["input_img"]

        if self.inferencer and self.inferencer.is_ready():
            logger.info("start infer partition")
            room_polygons = self._infer_partition(input_img, map_data)
        else:
            logger.info("start fallback partition")
            room_polygons = self._fallback_partition(input_img)

        return room_polygons

    def _infer_partition(self, input_img: np.ndarray,
                         map_data: Dict[str, Any]) -> List:
        """模型推理分区路径"""
        tensor = self._prepare_tensor(input_img, map_data)
        raw_output = self.inferencer.run(tensor)
        room_polygons = self.postprocessor.process(raw_output, map_data)
        return room_polygons

    def _fallback_partition(self, input_img: np.ndarray) -> List:
        """传统连通域分割 (Triton 不可用时的兜底方案)"""
        free = (input_img >= 200).astype(np.uint8)
        if not free.any():
            free = (input_img > 0).astype(np.uint8)

        _, labels = cv2.connectedComponents(free, connectivity=8)
        labels = labels.astype(np.int32)

        normal_map, small_map = self.postprocessor._split_by_area(labels, input_img)
        room_map = self.postprocessor._merge_fragments(normal_map, small_map)
        room_polygons = self.postprocessor._convert_to_polygons(input_img, room_map)
        return room_polygons

    # ==================== 张量准备 ====================

    def _prepare_tensor(self, input_img: np.ndarray,
                        map_data: Dict[str, Any]) -> np.ndarray:
        """
        将预处理后的灰度地图转换为 Triton 模型输入张量

        流程:
            1. 灰度 → BGR (3 通道复制)
            2. Letterbox resize 到 target_size, 保持长宽比, 填充 127 灰色
            3. uint8 → float32 归一化 [0, 1], 可选 ImageNet 均值/方差标准化
            4. HWC → NCHW

        同时将 scale / pad 写入 map_data, 供 postprocessor 做坐标逆映射。

        Args:
            input_img: 前处理后的灰度地图 (H, W) uint8
            map_data: 会追加 tensor_scale / tensor_pad / tensor_size / pre_pad

        Returns:
            tensor: (1, 3, target_h, target_w) float32
        """
        th, tw = self.target_size[0], self.target_size[1]

        # 灰度 → BGR
        if input_img.ndim == 2:
            img_bgr = cv2.cvtColor(input_img, cv2.COLOR_GRAY2BGR)
        else:
            img_bgr = input_img.copy()

        h, w = img_bgr.shape[:2]

        min_sz = self.min_input_size
        pre_pad = (0, 0)
        if h < min_sz and w < min_sz:
            # 小图先居中 pad 到 min_input_size
            pad_top_pre = (min_sz - h) // 2
            pad_left_pre = (min_sz - w) // 2
            padded = np.full((min_sz, min_sz, 3), 127, dtype=img_bgr.dtype)
            padded[pad_top_pre:pad_top_pre + h, pad_left_pre:pad_left_pre + w] = img_bgr
            img_bgr = padded
            pre_pad = (pad_top_pre, pad_left_pre)

        h, w = img_bgr.shape[:2]

        # Letterbox: 等比缩放, 短边 pad 至 target
        scale = min(tw / w, th / h)
        nw, nh = int(round(w * scale)), int(round(h * scale))
        resized = cv2.resize(img_bgr, (nw, nh), interpolation=cv2.INTER_NEAREST)

        pad_top = (th - nh) // 2
        pad_left = (tw - nw) // 2
        canvas = np.full((th, tw, 3), 127, dtype=np.uint8)
        canvas[pad_top:pad_top + nh, pad_left:pad_left + nw] = resized

        # 归一化
        tensor = canvas.astype(np.float32) / 255.0
        if self.normalize:
            mean = np.array(self.mean, dtype=np.float32)
            std = np.array(self.std, dtype=np.float32)
            tensor = (tensor - mean) / std

        # HWC → NCHW
        tensor = np.ascontiguousarray(
            tensor.transpose(2, 0, 1)[np.newaxis, ...], dtype=np.float32
        )

        # 写入 map_data, 供 postprocessor 逆映射
        map_data["tensor_scale"] = scale
        map_data["tensor_pad"] = (pad_top, pad_left)
        map_data["tensor_size"] = (th, tw)
        map_data["pre_pad"] = pre_pad

        return tensor

    # ==================== 通用工具方法 ====================

    @staticmethod
    def _extract_contours(label_map: np.ndarray) -> List[np.ndarray]:
        """从 label_map 提取各房间轮廓"""
        contours = []
        if label_map is None or label_map.max() == 0:
            return contours
        for lid in range(1, label_map.max() + 1):
            mask = (label_map == lid).astype(np.uint8)
            if mask.sum() == 0:
                continue
            cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
            if cnts:
                contours.append(max(cnts, key=cv2.contourArea))
        return contours

    @staticmethod
    def _relabel(label_map: np.ndarray) -> np.ndarray:
        """重编号使 label ID 从 1 开始连续"""
        unique_ids = sorted(set(label_map.flat) - {0})
        new_map = np.zeros_like(label_map)
        for new_id, old_id in enumerate(unique_ids, start=1):
            new_map[label_map == old_id] = new_id
        return new_map

    @staticmethod
    def _contours_to_label_map(contours: List[np.ndarray],
                               map_img: np.ndarray,
                               shape: Tuple[int, int]) -> np.ndarray:
        """轮廓列表 → label_map，仅标记当前地图上的自由区域(255)"""
        h, w = shape[:2]
        label_map = np.zeros((h, w), dtype=np.int32)
        for i, cnt in enumerate(contours):
            cv2.drawContours(label_map, [cnt], -1, i + 1, -1)
        return label_map 

    @staticmethod
    def _to_gray(img: np.ndarray) -> np.ndarray:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img.copy()

    @staticmethod
    def _get_charge_pixel(
        map_data: Dict[str, Any],
        transformer: CoordinateTransformer,
    ) -> Optional[Tuple[int, int]]:
        """充电桩世界坐标 → 像素坐标"""
        pose = map_data.get("world_charge_pose", [0, 0, 0])
        if pose == [0, 0, 0]:
            return None
        return transformer.world_to_pixel(pose[0], pose[1])

    @staticmethod
    def _get_marker_polygons(map_data: Dict[str, Any]) -> Optional[List]:
        """提取家具标记多边形"""
        markers_json = map_data.get("markers_json")
        if not markers_json:
            return None
        polys = []
        for item in markers_json.get("data", []):
            if item.get("name") == "家具" and item.get("type") == "polygon":
                polys.append(item["geometry"])
        return polys if polys else None

    # ==================== 房间统计 ====================

    @staticmethod
    def get_room_info(label_map: np.ndarray,
                      resolution: float = 0.05) -> List[Dict[str, Any]]:
        """获取各房间统计信息"""
        rooms = []
        if label_map is None or label_map.max() == 0:
            return rooms
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

    
