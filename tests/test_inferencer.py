"""Inferencer 单元测试（pytest 风格, mock TritonClient）"""

import math
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from app.pipeline.inferencer import Inferencer


# ==================== fixtures ====================

@pytest.fixture
def config():
    return {
        "triton_url": "localhost:8001",
        "model_name": "room_seg",
        "conf_threshold": 0.5,
        "nms_threshold": 0.45,
        "output_format": "xyxyxyxy",
    }


@pytest.fixture
def inferencer(config):
    with patch("app.pipeline.inferencer.TritonClient") as MockClient:
        mock_client = MagicMock()
        MockClient.return_value = mock_client
        inf = Inferencer(config)
    return inf


@pytest.fixture
def inferencer_xywha():
    config = {
        "triton_url": "localhost:8001",
        "conf_threshold": 0.3,
        "output_format": "xywha",
    }
    with patch("app.pipeline.inferencer.TritonClient") as MockClient:
        MockClient.return_value = MagicMock()
        inf = Inferencer(config)
    return inf


def _make_xyxyxyxy_row(vertices, conf=0.9, cls=0):
    """构造一行 xyxyxyxy 格式数据: [x1,y1,x2,y2,x3,y3,x4,y4, conf, cls]"""
    row = []
    for v in vertices:
        row.extend(v)
    row.append(conf)
    row.append(cls)
    return row


def _make_xywha_row(cx, cy, w, h, angle, conf=0.9, cls=0):
    """构造一行 xywha 格式数据: [cx, cy, w, h, angle, conf, cls]"""
    return [cx, cy, w, h, angle, conf, cls]


# ==================== is_ready ====================

class TestIsReady:

    def test_delegates_to_client(self, inferencer):
        inferencer.client.is_ready.return_value = True
        assert inferencer.is_ready() is True

        inferencer.client.is_ready.return_value = False
        assert inferencer.is_ready() is False


# ==================== run ====================

class TestRun:

    def test_run_calls_client_and_decodes(self, inferencer):
        # 模拟 Triton 返回 1 个 OBB
        raw = np.array([[_make_xyxyxyxy_row(
            [[0, 0], [10, 0], [10, 5], [0, 5]], conf=0.9
        )]], dtype=np.float32)
        inferencer.client.infer.return_value = raw

        tensor = np.zeros((1, 3, 512, 512), dtype=np.float32)
        result = inferencer.run(tensor)

        assert len(result) == 1
        assert len(result[0]) == 4  # 4 顶点
        inferencer.client.infer.assert_called_once()

    def test_run_raw_returns_numpy(self, inferencer):
        raw = np.zeros((1, 5, 10), dtype=np.float32)
        inferencer.client.infer.return_value = raw

        tensor = np.zeros((1, 3, 512, 512), dtype=np.float32)
        result = inferencer.run_raw(tensor)

        assert isinstance(result, np.ndarray)
        assert result.shape == (1, 5, 10)


# ==================== decode xyxyxyxy ====================

class TestDecodeXyxyxyxy:

    def test_single_obb(self, inferencer):
        raw = np.array([[_make_xyxyxyxy_row(
            [[10, 20], [30, 20], [30, 40], [10, 40]], conf=0.8
        )]], dtype=np.float32)

        obbs = inferencer.decode(raw)
        assert len(obbs) == 1
        assert obbs[0] == [[10, 20], [30, 20], [30, 40], [10, 40]]

    def test_multiple_obbs(self, inferencer):
        rows = [
            _make_xyxyxyxy_row([[0, 0], [5, 0], [5, 5], [0, 5]], conf=0.9),
            _make_xyxyxyxy_row([[10, 10], [20, 10], [20, 20], [10, 20]], conf=0.7),
        ]
        raw = np.array([rows], dtype=np.float32)

        obbs = inferencer.decode(raw)
        assert len(obbs) >= 1  # 至少 1 个通过 NMS

    def test_conf_filtering(self, inferencer):
        """低置信度的 OBB 被过滤"""
        rows = [
            _make_xyxyxyxy_row([[0, 0], [5, 0], [5, 5], [0, 5]], conf=0.9),
            _make_xyxyxyxy_row([[10, 10], [20, 10], [20, 20], [10, 20]], conf=0.3),  # 低于 0.5 阈值
        ]
        raw = np.array([rows], dtype=np.float32)

        obbs = inferencer.decode(raw)
        assert len(obbs) == 1

    def test_empty_output(self, inferencer):
        raw = np.zeros((1, 0, 10), dtype=np.float32)
        obbs = inferencer.decode(raw)
        assert obbs == []

    def test_all_below_threshold(self, inferencer):
        rows = [
            _make_xyxyxyxy_row([[0, 0], [5, 0], [5, 5], [0, 5]], conf=0.1),
            _make_xyxyxyxy_row([[10, 10], [20, 10], [20, 20], [10, 20]], conf=0.2),
        ]
        raw = np.array([rows], dtype=np.float32)

        obbs = inferencer.decode(raw)
        assert obbs == []

    def test_insufficient_dimensions(self, inferencer):
        raw = np.zeros((1, 5, 8), dtype=np.float32)  # < 10 列
        obbs = inferencer.decode(raw)
        assert obbs == []


# ==================== decode xywha ====================

class TestDecodeXywha:

    def test_single_obb(self, inferencer_xywha):
        # 中心 (50, 50), 宽高 20x10, 角度 0
        rows = [_make_xywha_row(50, 50, 20, 10, 0.0, conf=0.8)]
        raw = np.array([rows], dtype=np.float32)

        obbs = inferencer_xywha.decode(raw)
        assert len(obbs) == 1
        # 角度 0: 顶点应为 (60,55), (40,55), (40,45), (60,45)
        for v in obbs[0]:
            assert len(v) == 2

    def test_rotated_obb(self, inferencer_xywha):
        # 中心 (0, 0), w=4, h=2, 角度 pi/2
        rows = [_make_xywha_row(0, 0, 4, 2, math.pi / 2, conf=0.9)]
        raw = np.array([rows], dtype=np.float32)

        obbs = inferencer_xywha.decode(raw)
        assert len(obbs) == 1

        # 旋转 90 度: w 和 h 方向互换
        vertices = obbs[0]
        xs = [v[0] for v in vertices]
        ys = [v[1] for v in vertices]
        assert max(xs) - min(xs) == pytest.approx(2.0, abs=0.01)
        assert max(ys) - min(ys) == pytest.approx(4.0, abs=0.01)

    def test_conf_filtering(self, inferencer_xywha):
        rows = [
            _make_xywha_row(10, 10, 5, 5, 0, conf=0.5),
            _make_xywha_row(20, 20, 5, 5, 0, conf=0.1),  # 低于 0.3 阈值
        ]
        raw = np.array([rows], dtype=np.float32)

        obbs = inferencer_xywha.decode(raw)
        assert len(obbs) == 1

    def test_empty_output(self, inferencer_xywha):
        raw = np.zeros((1, 0, 7), dtype=np.float32)
        obbs = inferencer_xywha.decode(raw)
        assert obbs == []


# ==================== xywha → vertices ====================

class TestXywhaToVertices:

    def test_no_rotation(self):
        # 中心 (10, 10), 宽 4, 高 2, 角度 0
        vertices = Inferencer._xywha_to_vertices(10, 10, 4, 2, 0)
        assert len(vertices) == 4
        # 期望: (12,11), (8,11), (8,9), (12,9)
        xs = sorted([v[0] for v in vertices])
        ys = sorted([v[1] for v in vertices])
        assert xs[0] == pytest.approx(8.0)
        assert xs[-1] == pytest.approx(12.0)
        assert ys[0] == pytest.approx(9.0)
        assert ys[-1] == pytest.approx(11.0)

    def test_90_degree_rotation(self):
        vertices = Inferencer._xywha_to_vertices(0, 0, 4, 2, math.pi / 2)
        xs = sorted([v[0] for v in vertices])
        ys = sorted([v[1] for v in vertices])
        # 旋转后 w 和 h 交换
        assert xs[-1] - xs[0] == pytest.approx(2.0, abs=0.01)
        assert ys[-1] - ys[0] == pytest.approx(4.0, abs=0.01)

    def test_zero_size(self):
        vertices = Inferencer._xywha_to_vertices(5, 5, 0, 0, 0)
        for v in vertices:
            assert v[0] == pytest.approx(5.0)
            assert v[1] == pytest.approx(5.0)


# ==================== NMS ====================

class TestNMS:

    def test_no_overlap_keeps_all(self):
        obbs = [
            [[0, 0], [1, 0], [1, 1], [0, 1]],
            [[10, 10], [11, 10], [11, 11], [10, 11]],
        ]
        scores = [0.9, 0.8]
        result = Inferencer._nms_obb(obbs, scores, 0.5)
        assert len(result) == 2

    def test_full_overlap_suppresses(self):
        """完全重叠的两个 OBB, 低分被抑制"""
        obb = [[0, 0], [10, 0], [10, 10], [0, 10]]
        obbs = [obb, obb]
        scores = [0.9, 0.8]
        result = Inferencer._nms_obb(obbs, scores, 0.5)
        assert len(result) == 1

    def test_partial_overlap_below_threshold_keeps_both(self):
        """部分重叠但 IoU 低于阈值"""
        obb1 = [[0, 0], [5, 0], [5, 5], [0, 5]]
        obb2 = [[4, 0], [9, 0], [9, 5], [4, 5]]
        obbs = [obb1, obb2]
        scores = [0.9, 0.8]
        # 重叠区域 = 5, 面积各 25, IoU = 5/(25+25-5) = 0.11
        result = Inferencer._nms_obb(obbs, scores, 0.5)
        assert len(result) == 2

    def test_empty_input(self):
        result = Inferencer._nms_obb([], [], 0.5)
        assert result == []

    def test_single_obb(self):
        obbs = [[[0, 0], [1, 0], [1, 1], [0, 1]]]
        result = Inferencer._nms_obb(obbs, [0.9], 0.5)
        assert len(result) == 1


# ==================== polygon_area ====================

class TestPolygonArea:

    def test_unit_square(self):
        area = Inferencer._polygon_area([[0, 0], [1, 0], [1, 1], [0, 1]])
        assert area == pytest.approx(1.0)

    def test_rectangle(self):
        area = Inferencer._polygon_area([[0, 0], [4, 0], [4, 3], [0, 3]])
        assert area == pytest.approx(12.0)

    def test_triangle(self):
        area = Inferencer._polygon_area([[0, 0], [4, 0], [0, 3]])
        assert area == pytest.approx(6.0)
