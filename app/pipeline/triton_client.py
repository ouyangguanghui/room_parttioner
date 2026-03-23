"""Triton Inference Server 客户端封装"""

from typing import List, Optional, Tuple
import numpy as np

try:
    import tritonclient.grpc as grpcclient
    import tritonclient.http as httpclient
except ImportError:
    grpcclient = None
    httpclient = None


class TritonClient:
    """Triton 模型推理客户端"""

    def __init__(
        self,
        url: str = "localhost:8001",
        model_name: str = "room_seg",
        model_version: str = "",
        protocol: str = "grpc",
    ):
        self.model_name = model_name
        self.model_version = model_version
        self.protocol = protocol

        if grpcclient is None or httpclient is None:
            raise ImportError("tritonclient 未安装，请安装: pip install tritonclient[all]")

        if protocol == "grpc":
            self.client = grpcclient.InferenceServerClient(url=url)
        else:
            self.client = httpclient.InferenceServerClient(url=url)

    def is_ready(self) -> bool:
        """检查模型是否就绪"""
        try:
            return self.client.is_model_ready(self.model_name, self.model_version)
        except Exception:
            return False

    def infer(
        self,
        image: np.ndarray,
        input_name: str = "input",
        output_name: str = "output",
    ) -> np.ndarray:
        """
        推理单张图片

        Args:
            image: 预处理后的输入 (N, C, H, W) 或 (H, W, C)
            input_name: 模型输入节点名
            output_name: 模型输出节点名

        Returns:
            模型输出 numpy 数组
        """
        if image.ndim == 3:
            image = np.expand_dims(image, axis=0)

        if self.protocol == "grpc":
            inputs = [grpcclient.InferInput(input_name, image.shape, self._np_to_triton_dtype(image.dtype))]
            inputs[0].set_data_from_numpy(image)
            outputs = [grpcclient.InferRequestedOutput(output_name)]
            result = self.client.infer(
                model_name=self.model_name,
                model_version=self.model_version,
                inputs=inputs,
                outputs=outputs,
            )
        else:
            inputs = [httpclient.InferInput(input_name, image.shape, self._np_to_triton_dtype(image.dtype))]
            inputs[0].set_data_from_numpy(image)
            outputs = [httpclient.InferRequestedOutput(output_name)]
            result = self.client.infer(
                model_name=self.model_name,
                model_version=self.model_version,
                inputs=inputs,
                outputs=outputs,
            )

        return result.as_numpy(output_name)

    @staticmethod
    def _np_to_triton_dtype(dtype: np.dtype) -> str:
        mapping = {
            np.float32: "FP32",
            np.float16: "FP16",
            np.int32: "INT32",
            np.int64: "INT64",
            np.uint8: "UINT8",
            np.int8: "INT8",
        }
        return mapping.get(dtype.type, "FP32")
