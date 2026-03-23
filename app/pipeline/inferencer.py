"""模型推理模块 — 封装 Triton 调用"""

from typing import Dict, Any
import numpy as np

from app.pipeline.triton_client import TritonClient


class Inferencer:
    """模型推理：调用 Triton 模型服务"""

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.client = TritonClient(
            url=self.config.get("triton_url", "localhost:8001"),
            model_name=self.config.get("model_name", "room_seg"),
            model_version=self.config.get("model_version", ""),
            protocol=self.config.get("triton_protocol", "grpc"),
        )
        self.input_name = self.config.get("input_name", "input")
        self.output_name = self.config.get("output_name", "output")

    def is_ready(self) -> bool:
        return self.client.is_ready()

    def run(self, tensor: np.ndarray) -> np.ndarray:
        """
        执行模型推理

        Args:
            tensor: 前处理后的输入 (N, C, H, W) float32

        Returns:
            模型原始输出
        """
        return self.client.infer(tensor, self.input_name, self.output_name)
