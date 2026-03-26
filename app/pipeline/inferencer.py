"""模型推理模块 — 封装 Triton 调用 + 输出解码

职责:
  1. 管理 TritonClient 连接
  2. 发送张量 → 获取原始输出
  3. 将原始输出解码为 OBB 列表 (供 Postprocessor 消费)

支持两种模型输出格式:
  - "xyxyxyxy": 4 点坐标 [x1,y1,x2,y2,x3,y3,x4,y4, conf, cls]
  - "xywha":    YOLO-OBB [cx, cy, w, h, angle, conf, cls]
"""

import math
import logging
from typing import Dict, Any, List

import numpy as np

from app.pipeline.triton_client import TritonClient

logger = logging.getLogger(__name__)


class Inferencer:
    """模型推理 + 输出解码

    Config keys:
        triton_url:       Triton 地址 (default: localhost:8001)
        model_name:       模型名 (default: room_seg)
        model_version:    版本 (default: "" = latest)
        triton_protocol:  grpc | http (default: grpc)
        triton_timeout:   推理超时秒数 (default: 30)
        triton_retries:   重试次数 (default: 3)
        input_name:       模型输入节点名 (default: input)
        output_name:      模型输出节点名 (default: output)
        conf_threshold:   置信度过滤阈值 (default: 0.5)
        nms_threshold:    NMS IoU 阈值 (default: 0.45)
        output_format:    "xyxyxyxy" | "xywha" (default: xyxyxyxy)
    """

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}

        self.client = TritonClient(
            url=self.config.get("triton_url", "localhost:8001"),
            model_name=self.config.get("model_name", "room_seg"),
            model_version=self.config.get("model_version", ""),
            protocol=self.config.get("triton_protocol", "grpc"),
            timeout=self.config.get("triton_timeout", 30.0),
            max_retries=self.config.get("triton_retries", 3),
        )

        self.input_name = self.config.get("input_name", "input")
        self.output_name = self.config.get("output_name", "output")

        # 解码参数
        self.conf_threshold = self.config.get("conf_threshold", 0.5)
        self.nms_threshold = self.config.get("nms_threshold", 0.45)
        self.output_format = self.config.get("output_format", "xyxyxyxy")

    # ==================== 健康检查 ====================

    def is_ready(self) -> bool:
        """模型是否可用"""
        return self.client.is_ready()

    # ==================== 推理主入口 ====================

    def run(self, tensor: np.ndarray) -> List[List[List[float]]]:
        """
        执行推理并解码输出为 OBB 列表

        Args:
            tensor: 预处理后的输入 (N, C, H, W) float32

        Returns:
            OBB 列表, 每个 OBB 为 4 个顶点 [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
        """
        raw = self.client.infer(tensor, self.input_name, self.output_name)
        return self.decode(raw)

    def run_raw(self, tensor: np.ndarray) -> np.ndarray:
        """仅执行推理, 返回原始张量 (不解码)"""
        return self.client.infer(tensor, self.input_name, self.output_name)

    # ==================== 输出解码 ====================

    def decode(self, raw_output: np.ndarray) -> List[List[List[float]]]:
        """
        将模型原始输出解码为 OBB 四顶点列表

        Args:
            raw_output: 模型原始输出张量

        Returns:
            [[[x1,y1],[x2,y2],[x3,y3],[x4,y4]], ...]
        """
        if self.output_format == "xywha":
            return self._decode_xywha(raw_output)
        else:
            return self._decode_xyxyxyxy(raw_output)

    def _decode_xyxyxyxy(self, raw: np.ndarray) -> List[List[List[float]]]:
        """
        解码 4 点坐标格式

        预期 raw shape: (1, N, 10+) 或 (N, 10+)
        每行: [x1, y1, x2, y2, x3, y3, x4, y4, conf, cls, ...]
        """
        preds = raw.reshape(-1, raw.shape[-1])  # (N, D)
        if preds.shape[0] == 0 or preds.shape[1] < 10:
            logger.warning("模型输出为空或维度不足: shape=%s", preds.shape)
            return []

        # 置信度过滤
        confs = preds[:, 8]
        mask = confs >= self.conf_threshold
        preds = preds[mask]

        if len(preds) == 0:
            return []

        # 提取 OBB 四顶点
        obbs = []
        for row in preds:
            vertices = [
                [float(row[0]), float(row[1])],
                [float(row[2]), float(row[3])],
                [float(row[4]), float(row[5])],
                [float(row[6]), float(row[7])],
            ]
            obbs.append(vertices)

        # NMS
        if len(obbs) > 1:
            scores = preds[:, 8].tolist()
            obbs = self._nms_obb(obbs, scores, self.nms_threshold)

        logger.info("解码完成: %d 个 OBB (conf >= %.2f)",
                     len(obbs), self.conf_threshold)
        return obbs

    def _decode_xywha(self, raw: np.ndarray) -> List[List[List[float]]]:
        """
        解码 YOLO-OBB [cx, cy, w, h, angle, conf, cls, ...] 格式

        预期 raw shape: (1, N, 7+) 或 (N, 7+)
        angle 为弧度
        """
        preds = raw.reshape(-1, raw.shape[-1])
        if preds.shape[0] == 0 or preds.shape[1] < 7:
            logger.warning("模型输出为空或维度不足: shape=%s", preds.shape)
            return []

        # 置信度过滤
        confs = preds[:, 5]
        mask = confs >= self.conf_threshold
        preds = preds[mask]

        if len(preds) == 0:
            return []

        # xywha → 4 顶点
        obbs = []
        for row in preds:
            cx, cy, w, h, angle = float(row[0]), float(row[1]), float(row[2]), float(row[3]), float(row[4])
            vertices = self._xywha_to_vertices(cx, cy, w, h, angle)
            obbs.append(vertices)

        # NMS
        if len(obbs) > 1:
            scores = preds[:, 5].tolist()
            obbs = self._nms_obb(obbs, scores, self.nms_threshold)

        logger.info("解码完成: %d 个 OBB (conf >= %.2f)",
                     len(obbs), self.conf_threshold)
        return obbs

    # ==================== 几何工具 ====================

    @staticmethod
    def _xywha_to_vertices(
        cx: float, cy: float, w: float, h: float, angle: float,
    ) -> List[List[float]]:
        """
        (cx, cy, w, h, angle) → 4 个顶点 [[x,y], ...]

        角度为弧度, 逆时针为正。
        顶点顺序与 line_to_obb 一致:
            v0 = center + R @ (+w/2, +h/2)
            v1 = center + R @ (-w/2, +h/2)
            v2 = center + R @ (-w/2, -h/2)
            v3 = center + R @ (+w/2, -h/2)
        """
        cos_a = math.cos(angle)
        sin_a = math.sin(angle)

        hw, hh = w / 2, h / 2
        # 旋转矩阵乘以四个角偏移
        corners = [
            (+hw, +hh),
            (-hw, +hh),
            (-hw, -hh),
            (+hw, -hh),
        ]
        vertices = []
        for dx, dy in corners:
            x = cx + dx * cos_a - dy * sin_a
            y = cy + dx * sin_a + dy * cos_a
            vertices.append([x, y])

        return vertices

    @staticmethod
    def _polygon_area(vertices: List[List[float]]) -> float:
        """Shoelface 公式计算多边形面积"""
        n = len(vertices)
        area = 0.0
        for i in range(n):
            j = (i + 1) % n
            area += vertices[i][0] * vertices[j][1]
            area -= vertices[j][0] * vertices[i][1]
        return abs(area) / 2.0

    @staticmethod
    def _polygon_intersection_area(
        poly_a: List[List[float]],
        poly_b: List[List[float]],
    ) -> float:
        """
        使用 Shapely 计算两个凸多边形的交集面积
        (OBB 都是凸四边形)
        """
        try:
            from shapely.geometry import Polygon
            a = Polygon(poly_a)
            b = Polygon(poly_b)
            if not a.is_valid or not b.is_valid:
                return 0.0
            return a.intersection(b).area
        except Exception:
            return 0.0

    @classmethod
    def _nms_obb(
        cls,
        obbs: List[List[List[float]]],
        scores: List[float],
        iou_threshold: float,
    ) -> List[List[List[float]]]:
        """
        OBB 非极大值抑制 (NMS)

        Args:
            obbs: OBB 列表 (4 顶点)
            scores: 对应置信度
            iou_threshold: IoU 阈值

        Returns:
            过滤后的 OBB 列表
        """
        if len(obbs) == 0:
            return []

        # 按分数降序排列
        indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)

        areas = [cls._polygon_area(obb) for obb in obbs]
        keep = []
        suppressed = set()

        for i in indices:
            if i in suppressed:
                continue
            keep.append(i)

            for j in indices:
                if j in suppressed or j == i:
                    continue
                inter = cls._polygon_intersection_area(obbs[i], obbs[j])
                union = areas[i] + areas[j] - inter
                if union > 0 and inter / union >= iou_threshold:
                    suppressed.add(j)

        return [obbs[i] for i in keep]
