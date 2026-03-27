"""Triton Inference Server 客户端封装

功能:
  - 支持 gRPC / HTTP 双协议
  - 超时控制 + 指数退避重试
  - 多输入/多输出推理
  - 模型元数据查询 (输入输出 shape / dtype)
  - 健康检查 (带缓存, 避免频繁探测)
"""

import time
import logging
from typing import Dict, List, Optional

import numpy as np

try:
    import tritonclient.grpc as grpcclient
except (ImportError, RuntimeError):
    grpcclient = None

try:
    import tritonclient.http as httpclient
except (ImportError, RuntimeError):
    httpclient = None

logger = logging.getLogger(__name__)

# numpy dtype → Triton dtype 字符串
_NP_TO_TRITON: Dict[type, str] = {
    np.float32: "FP32",
    np.float16: "FP16",
    np.float64: "FP64",
    np.int32:   "INT32",
    np.int64:   "INT64",
    np.int16:   "INT16",
    np.int8:    "INT8",
    np.uint8:   "UINT8",
    np.uint16:  "UINT16",
    np.uint32:  "UINT32",
    np.uint64:  "UINT64",
    np.bool_:   "BOOL",
}


class TritonClient:
    """Triton 模型推理客户端

    Args:
        url: Triton 服务地址, gRPC 默认 8001, HTTP 默认 8000
        model_name: 模型仓库中的名称
        model_version: 版本号 (空字符串 = latest)
        protocol: "grpc" 或 "http"
        timeout: 单次推理超时 (秒)
        max_retries: 最大重试次数 (含首次)
        retry_backoff: 重试退避基数 (秒), 实际等待 = backoff * 2^attempt
    """

    def __init__(
        self,
        url: str = "localhost:8001",
        model_name: str = "room_seg",
        model_version: str = "",
        protocol: str = "grpc",
        timeout: float = 30.0,
        max_retries: int = 3,
        retry_backoff: float = 0.5,
    ):
        self.model_name = model_name
        self.model_version = model_version
        self.protocol = protocol.lower()
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_backoff = retry_backoff

        # 健康检查缓存
        self._ready_cache: Optional[bool] = None
        self._ready_cache_ts: float = 0.0
        self._ready_cache_ttl: float = 10.0  # 缓存 10 秒

        self._client = self._create_client(url)

    # ==================== 客户端创建 ====================

    def _create_client(self, url: str):
        if self.protocol == "grpc":
            if grpcclient is None:
                raise ImportError(
                    "tritonclient[grpc] 未安装, 请执行: pip install tritonclient[grpc]"
                )
            return grpcclient.InferenceServerClient(
                url=url,
                verbose=False,
            )
        else:
            if httpclient is None:
                raise ImportError(
                    "tritonclient[http] 未安装, 请执行: pip install tritonclient[http]"
                )
            return httpclient.InferenceServerClient(
                url=url,
                verbose=False,
            )

    # ==================== 健康检查 ====================

    def is_server_live(self) -> bool:
        """服务器是否存活"""
        try:
            return self._client.is_server_live()
        except Exception:
            return False

    def is_ready(self) -> bool:
        """模型是否就绪 (带缓存)"""
        now = time.monotonic()
        if (self._ready_cache is not None
                and now - self._ready_cache_ts < self._ready_cache_ttl):
            return self._ready_cache

        try:
            ready = self._client.is_model_ready(
                self.model_name, self.model_version
            )
        except Exception:
            ready = False

        self._ready_cache = ready
        self._ready_cache_ts = now
        return ready

    def invalidate_cache(self):
        """手动使健康检查缓存失效"""
        self._ready_cache = None

    # ==================== 模型元数据 ====================

    def get_model_metadata(self) -> Dict:
        """
        查询模型输入/输出元数据

        Returns:
            {
                "name": str,
                "inputs":  [{"name": str, "shape": list, "datatype": str}, ...],
                "outputs": [{"name": str, "shape": list, "datatype": str}, ...],
            }
        """
        meta = self._client.get_model_metadata(
            self.model_name, self.model_version
        )

        if self.protocol == "grpc":
            return {
                "name": meta.name,
                "inputs": [
                    {"name": inp.name, "shape": list(inp.shape), "datatype": inp.datatype}
                    for inp in meta.inputs
                ],
                "outputs": [
                    {"name": out.name, "shape": list(out.shape), "datatype": out.datatype}
                    for out in meta.outputs
                ],
            }
        else:
            return {
                "name": meta["name"],
                "inputs": [
                    {"name": inp["name"], "shape": inp["shape"], "datatype": inp["datatype"]}
                    for inp in meta["inputs"]
                ],
                "outputs": [
                    {"name": out["name"], "shape": out["shape"], "datatype": out["datatype"]}
                    for out in meta["outputs"]
                ],
            }

    # ==================== 推理 ====================

    def infer(
        self,
        input_data: np.ndarray,
        input_name: str = "input",
        output_name: str = "output",
    ) -> np.ndarray:
        """
        单输入/单输出推理

        Args:
            input_data: 输入张量, 3D 时自动补 batch 维
            input_name: 模型输入节点名
            output_name: 模型输出节点名

        Returns:
            输出 numpy 数组
        """
        results = self.infer_multi(
            inputs={input_name: input_data},
            output_names=[output_name],
        )
        return results[output_name]

    def infer_multi(
        self,
        inputs: Dict[str, np.ndarray],
        output_names: List[str],
    ) -> Dict[str, np.ndarray]:
        """
        多输入/多输出推理 (带重试)

        Args:
            inputs: {input_name: ndarray}
            output_names: 要获取的输出名列表

        Returns:
            {output_name: ndarray}
        """
        last_error: Optional[Exception] = None

        for attempt in range(self.max_retries):
            try:
                return self._do_infer(inputs, output_names)
            except Exception as e:
                last_error = e
                if attempt < self.max_retries - 1:
                    wait = self.retry_backoff * (2 ** attempt)
                    logger.warning(
                        "推理失败 (attempt %d/%d): %s, %.1fs 后重试",
                        attempt + 1, self.max_retries, e, wait,
                    )
                    time.sleep(wait)
                    # 重试前使缓存失效
                    self.invalidate_cache()

        raise RuntimeError(
            f"推理失败, 已重试 {self.max_retries} 次: {last_error}"
        ) from last_error

    def _do_infer(
        self,
        inputs: Dict[str, np.ndarray],
        output_names: List[str],
    ) -> Dict[str, np.ndarray]:
        """执行一次推理调用"""
        if self.protocol == "grpc":
            return self._infer_grpc(inputs, output_names)
        else:
            return self._infer_http(inputs, output_names)

    def _infer_grpc(
        self,
        inputs: Dict[str, np.ndarray],
        output_names: List[str],
    ) -> Dict[str, np.ndarray]:
        triton_inputs = []
        for name, data in inputs.items():
            if data.ndim == 3:
                data = np.expand_dims(data, axis=0)
            inp = grpcclient.InferInput(
                name, list(data.shape), self._np_to_triton_dtype(data.dtype)
            )
            inp.set_data_from_numpy(data)
            triton_inputs.append(inp)

        triton_outputs = [grpcclient.InferRequestedOutput(n) for n in output_names]

        result = self._client.infer(
            model_name=self.model_name,
            model_version=self.model_version,
            inputs=triton_inputs,
            outputs=triton_outputs,
            client_timeout=self.timeout,
        )

        return {name: result.as_numpy(name) for name in output_names}

    def _infer_http(
        self,
        inputs: Dict[str, np.ndarray],
        output_names: List[str],
    ) -> Dict[str, np.ndarray]:
        triton_inputs = []
        for name, data in inputs.items():
            if data.ndim == 3:
                data = np.expand_dims(data, axis=0)
            inp = httpclient.InferInput(
                name, list(data.shape), self._np_to_triton_dtype(data.dtype)
            )
            inp.set_data_from_numpy(data)
            triton_inputs.append(inp)

        triton_outputs = [httpclient.InferRequestedOutput(n) for n in output_names]

        result = self._client.infer(
            model_name=self.model_name,
            model_version=self.model_version,
            inputs=triton_inputs,
            outputs=triton_outputs,
            timeout=self.timeout,
        )

        return {name: result.as_numpy(name) for name in output_names}

    # ==================== 工具 ====================

    @staticmethod
    def _np_to_triton_dtype(dtype: np.dtype) -> str:
        """numpy dtype → Triton dtype 字符串"""
        return _NP_TO_TRITON.get(dtype.type, "FP32")
