"""TritonClient 单元测试（pytest 风格, mock tritonclient）"""

import time
from unittest.mock import MagicMock, patch, PropertyMock

import numpy as np
import pytest

from app.pipeline.triton_client import TritonClient, _NP_TO_TRITON


# ==================== fixtures ====================

@pytest.fixture
def mock_grpc_module():
    """创建完整的 mock grpcclient 模块"""
    mock_module = MagicMock()
    mock_client_instance = MagicMock()
    mock_module.InferenceServerClient.return_value = mock_client_instance
    mock_module.InferInput = MagicMock()
    mock_module.InferRequestedOutput = MagicMock()
    return mock_module, mock_client_instance


@pytest.fixture
def grpc_client(mock_grpc_module):
    """创建使用 mock grpc 的 TritonClient"""
    mock_module, mock_instance = mock_grpc_module
    with patch("app.pipeline.triton_client.grpcclient", mock_module):
        client = TritonClient(
            url="localhost:8001",
            model_name="room_seg",
            protocol="grpc",
            timeout=10.0,
            max_retries=2,
            retry_backoff=0.01,
        )
    # 将 mock_module 挂在 client 上方便后续 patch
    client._mock_module = mock_module
    return client, mock_instance


@pytest.fixture
def mock_http_module():
    """创建完整的 mock httpclient 模块"""
    mock_module = MagicMock()
    mock_client_instance = MagicMock()
    mock_module.InferenceServerClient.return_value = mock_client_instance
    mock_module.InferInput = MagicMock()
    mock_module.InferRequestedOutput = MagicMock()
    return mock_module, mock_client_instance


@pytest.fixture
def http_client(mock_http_module):
    """创建使用 mock http 的 TritonClient"""
    mock_module, mock_instance = mock_http_module
    with patch("app.pipeline.triton_client.httpclient", mock_module):
        client = TritonClient(
            url="localhost:8000",
            model_name="room_seg",
            protocol="http",
            timeout=10.0,
            max_retries=2,
            retry_backoff=0.01,
        )
    client._mock_module = mock_module
    return client, mock_instance


# ==================== 健康检查 ====================

class TestHealthCheck:

    def test_is_server_live_true(self, grpc_client):
        client, mock_inst = grpc_client
        mock_inst.is_server_live.return_value = True
        assert client.is_server_live() is True

    def test_is_server_live_exception(self, grpc_client):
        client, mock_inst = grpc_client
        mock_inst.is_server_live.side_effect = ConnectionError("refused")
        assert client.is_server_live() is False

    def test_is_ready_true(self, grpc_client):
        client, mock_inst = grpc_client
        mock_inst.is_model_ready.return_value = True
        assert client.is_ready() is True

    def test_is_ready_false_on_exception(self, grpc_client):
        client, mock_inst = grpc_client
        mock_inst.is_model_ready.side_effect = Exception("timeout")
        assert client.is_ready() is False

    def test_is_ready_cache(self, grpc_client):
        client, mock_inst = grpc_client
        mock_inst.is_model_ready.return_value = True

        # 首次调用
        assert client.is_ready() is True
        assert mock_inst.is_model_ready.call_count == 1

        # 缓存命中, 不再调用
        assert client.is_ready() is True
        assert mock_inst.is_model_ready.call_count == 1

    def test_invalidate_cache(self, grpc_client):
        client, mock_inst = grpc_client
        mock_inst.is_model_ready.return_value = True

        client.is_ready()
        assert mock_inst.is_model_ready.call_count == 1

        client.invalidate_cache()
        client.is_ready()
        assert mock_inst.is_model_ready.call_count == 2


# ==================== dtype 转换 ====================

class TestDtypeConversion:

    @pytest.mark.parametrize("np_type,expected", [
        (np.float32, "FP32"),
        (np.float16, "FP16"),
        (np.int32,   "INT32"),
        (np.int64,   "INT64"),
        (np.uint8,   "UINT8"),
        (np.int8,    "INT8"),
    ])
    def test_np_to_triton_dtype(self, np_type, expected):
        arr = np.zeros(1, dtype=np_type)
        assert TritonClient._np_to_triton_dtype(arr.dtype) == expected

    def test_unknown_dtype_defaults_to_fp32(self):
        arr = np.zeros(1, dtype=np.complex128)
        assert TritonClient._np_to_triton_dtype(arr.dtype) == "FP32"


# ==================== 推理 ====================

class TestInfer:

    def test_infer_grpc_single_io(self, grpc_client):
        client, mock_inst = grpc_client
        fake_result = MagicMock()
        fake_result.as_numpy.return_value = np.zeros((1, 10, 10), dtype=np.float32)
        mock_inst.infer.return_value = fake_result

        tensor = np.random.randn(1, 3, 512, 512).astype(np.float32)
        with patch("app.pipeline.triton_client.grpcclient", client._mock_module):
            result = client.infer(tensor, "input", "output")

        assert result.shape == (1, 10, 10)
        mock_inst.infer.assert_called_once()

    def test_infer_http_single_io(self, http_client):
        client, mock_inst = http_client
        fake_result = MagicMock()
        fake_result.as_numpy.return_value = np.ones((1, 5, 10), dtype=np.float32)
        mock_inst.infer.return_value = fake_result

        tensor = np.random.randn(1, 3, 512, 512).astype(np.float32)
        with patch("app.pipeline.triton_client.httpclient", client._mock_module):
            result = client.infer(tensor, "input", "output")

        assert result.shape == (1, 5, 10)

    def test_infer_auto_expand_3d_input(self, grpc_client):
        """3D 输入自动补 batch 维"""
        client, mock_inst = grpc_client
        fake_result = MagicMock()
        fake_result.as_numpy.return_value = np.zeros((1, 10), dtype=np.float32)
        mock_inst.infer.return_value = fake_result

        mock_infer_input = MagicMock()
        client._mock_module.InferInput.return_value = mock_infer_input

        tensor = np.random.randn(3, 512, 512).astype(np.float32)
        with patch("app.pipeline.triton_client.grpcclient", client._mock_module):
            client.infer(tensor, "input", "output")

        # 验证 InferInput 被调用时 shape 是 4D
        call_args = client._mock_module.InferInput.call_args
        shape = call_args[0][1]
        assert len(shape) == 4

    def test_infer_multi_outputs(self, grpc_client):
        client, mock_inst = grpc_client
        fake_result = MagicMock()
        fake_result.as_numpy.side_effect = lambda name: {
            "boxes": np.zeros((1, 10, 10), dtype=np.float32),
            "scores": np.ones((1, 10), dtype=np.float32),
        }[name]
        mock_inst.infer.return_value = fake_result

        tensor = np.random.randn(1, 3, 512, 512).astype(np.float32)
        with patch("app.pipeline.triton_client.grpcclient", client._mock_module):
            results = client.infer_multi(
                {"input": tensor},
                ["boxes", "scores"],
            )

        assert "boxes" in results
        assert "scores" in results
        assert results["boxes"].shape == (1, 10, 10)
        assert results["scores"].shape == (1, 10)


# ==================== 重试 ====================

class TestRetry:

    def test_retry_on_failure_then_success(self, grpc_client):
        client, mock_inst = grpc_client

        fake_result = MagicMock()
        fake_result.as_numpy.return_value = np.zeros((1,), dtype=np.float32)

        # 第一次失败, 第二次成功
        mock_inst.infer.side_effect = [
            ConnectionError("first fail"),
            fake_result,
        ]

        tensor = np.random.randn(1, 3, 512, 512).astype(np.float32)
        with patch("app.pipeline.triton_client.grpcclient", client._mock_module):
            result = client.infer(tensor, "input", "output")

        assert result.shape == (1,)
        assert mock_inst.infer.call_count == 2

    def test_retry_exhausted_raises(self, grpc_client):
        client, mock_inst = grpc_client
        mock_inst.infer.side_effect = ConnectionError("always fail")

        tensor = np.random.randn(1, 3, 512, 512).astype(np.float32)
        with patch("app.pipeline.triton_client.grpcclient", client._mock_module):
            with pytest.raises(RuntimeError, match="已重试"):
                client.infer(tensor, "input", "output")

        assert mock_inst.infer.call_count == 2  # max_retries=2


# ==================== 导入失败 ====================

class TestImportError:

    def test_grpc_import_error(self):
        with patch("app.pipeline.triton_client.grpcclient", None):
            with pytest.raises(ImportError, match="grpc"):
                TritonClient(protocol="grpc")

    def test_http_import_error(self):
        with patch("app.pipeline.triton_client.httpclient", None):
            with pytest.raises(ImportError, match="http"):
                TritonClient(protocol="http")
