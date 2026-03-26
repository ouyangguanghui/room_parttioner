"""RoomService 单元测试（pytest 风格）。"""

from unittest.mock import MagicMock, patch, PropertyMock

import numpy as np
import pytest

from app.services.services import RoomService
from app.core.errors import (
    InvalidResolutionError,
    NoLabelsError,
    OperationFailedError,
)


# ==================== fixtures ====================

@pytest.fixture
def config():
    return {
        "target_size": [512, 512],
        "resolution": 0.05,
        "wall_threshold": 128,
        "door_width": 20,
        "grow_iterations": 10,
        "min_room_area": 1.0,
        "line_thickness": 3,
    }


@pytest.fixture
def service(config):
    """使用 mock 替换预处理器，避免实际图像处理开销"""
    with patch("app.services.services.Preprocessor") as MockPP:
        mock_pp = MagicMock()
        mock_pp.process.return_value = {
            "map_data": np.full((20, 20), 200, dtype=np.uint8),
            "input_data": np.full((20, 20), 200, dtype=np.uint8),
        }
        MockPP.return_value = mock_pp
        svc = RoomService(config)
        svc._mock_preprocessor = mock_pp
        yield svc


@pytest.fixture
def map_data_no_labels():
    """无 labels 的 map_data"""
    return {
        "map_img": np.full((20, 20, 3), 200, dtype=np.uint8),
        "resolution": 0.05,
        "origin": [0.0, 0.0],
        "labels_json": None,
        "robot_model": "s10",
        "uuid": "test-uuid",
        "markers_json": None,
        "world_charge_pose": [0, 0, 0],
    }


@pytest.fixture
def map_data_with_labels():
    """有 labels 的 map_data"""
    return {
        "map_img": np.full((20, 20, 3), 200, dtype=np.uint8),
        "resolution": 0.05,
        "origin": [0.0, 0.0],
        "labels_json": {
            "version": "online_4.0.2",
            "uuid": "test-uuid",
            "data": [
                {
                    "name": "A", "id": "ROOM_001", "type": "polygon",
                    "geometry": [0, 0, 1, 0, 1, 1, 0, 1, 0, 0],
                    "colorType": 0, "graph": [], "groundMaterial": None,
                },
            ],
        },
        "robot_model": "s10",
        "uuid": "test-uuid",
        "markers_json": None,
        "world_charge_pose": [1.0, 2.0, 0],
    }


# ==================== TestInit ====================

class TestInit:
    def test_services_created(self, service):
        assert service.auto_partitioner is not None
        assert service.extended_partitioner is not None
        assert service.manual_partitioner is not None
        assert service.manual_merger is not None

    def test_preprocessor_created(self, service):
        assert service.preprocessor is not None

    def test_default_config(self):
        with patch("app.services.services.Preprocessor"):
            svc = RoomService()
        assert svc.config is not None


# ==================== TestPreprocess ====================

class TestPreprocess:
    def test_bgr_to_gray(self, service):
        bgr = np.full((10, 10, 3), 128, dtype=np.uint8)
        service._preprocess(bgr)
        service._mock_preprocessor.process.assert_called_once()
        # 检查传入的是灰度图
        call_arg = service._mock_preprocessor.process.call_args[0][0]
        assert call_arg.ndim == 2

    def test_gray_passthrough(self, service):
        gray = np.full((10, 10), 128, dtype=np.uint8)
        service._preprocess(gray)
        call_arg = service._mock_preprocessor.process.call_args[0][0]
        assert call_arg.ndim == 2

    def test_returns_meta(self, service):
        gray = np.full((10, 10), 128, dtype=np.uint8)
        meta = service._preprocess(gray)
        assert "map_data" in meta
        assert "input_data" in meta


# ==================== TestInvalidResolution ====================

class TestInvalidResolution:
    def test_zero_resolution(self, service, map_data_no_labels):
        map_data_no_labels["resolution"] = 0
        with pytest.raises(InvalidResolutionError):
            service.room_edit(map_data_no_labels, "split")

    def test_negative_resolution(self, service, map_data_no_labels):
        map_data_no_labels["resolution"] = -0.05
        with pytest.raises(InvalidResolutionError):
            service.room_edit(map_data_no_labels, "split")

    def test_missing_resolution(self, service, map_data_no_labels):
        del map_data_no_labels["resolution"]
        with pytest.raises(InvalidResolutionError):
            service.room_edit(map_data_no_labels, "split")


# ==================== TestInvalidOperation ====================

class TestInvalidOperation:
    def test_unknown_operation(self, service, map_data_no_labels):
        with pytest.raises(OperationFailedError):
            service.room_edit(map_data_no_labels, "unknown_op")


# ==================== TestSplitRoute ====================

class TestSplitRoute:
    def test_no_labels_routes_to_auto(self, service, map_data_no_labels):
        """无 labels 时走 AutoPartitioner"""
        with patch.object(service.auto_partitioner, "process") as mock_auto:
            mock_auto.return_value = {"version": "v", "uuid": "", "data": []}
            result = service.room_edit(map_data_no_labels, "split")
            mock_auto.assert_called_once()
            # 验证 repartition=False
            assert mock_auto.call_args[1].get("repartition") is False \
                or mock_auto.call_args.kwargs.get("repartition") is False

    def test_with_labels_routes_to_extended(self, service, map_data_with_labels):
        """有 labels 时走 ExtendedPartitioner"""
        with patch.object(service.extended_partitioner, "process") as mock_ext:
            mock_ext.return_value = {"version": "v", "uuid": "", "data": []}
            result = service.room_edit(map_data_with_labels, "split")
            mock_ext.assert_called_once()

    def test_empty_labels_data_routes_to_auto(self, service, map_data_with_labels):
        """labels_json 存在但 data 为空时走 AutoPartitioner"""
        map_data_with_labels["labels_json"] = {"data": []}
        with patch.object(service.auto_partitioner, "process") as mock_auto:
            mock_auto.return_value = {"version": "v", "uuid": "", "data": []}
            service.room_edit(map_data_with_labels, "split")
            mock_auto.assert_called_once()


# ==================== TestRepartitionRoute ====================

class TestRepartitionRoute:
    def test_always_routes_to_auto(self, service, map_data_with_labels):
        """repartition 始终走 AutoPartitioner"""
        with patch.object(service.auto_partitioner, "process") as mock_auto:
            mock_auto.return_value = {"version": "v", "uuid": "", "data": []}
            service.room_edit(map_data_with_labels, "repartition")
            mock_auto.assert_called_once()
            assert mock_auto.call_args[1].get("repartition") is True \
                or mock_auto.call_args.kwargs.get("repartition") is True

    def test_no_labels_still_works(self, service, map_data_no_labels):
        """无 labels 时 repartition 也能正常工作"""
        with patch.object(service.auto_partitioner, "process") as mock_auto:
            mock_auto.return_value = {"version": "v", "uuid": "", "data": []}
            service.room_edit(map_data_no_labels, "repartition")
            mock_auto.assert_called_once()


# ==================== TestDivisionRoute ====================

class TestDivisionRoute:
    def test_routes_to_manual_partitioner(self, service, map_data_with_labels):
        div_dict = {"id": "ROOM_001", "A": [0.5, 0], "B": [0.5, 1]}
        with patch.object(service.manual_partitioner, "process") as mock_div:
            mock_div.return_value = {"version": "v", "uuid": "", "data": []}
            service.room_edit(
                map_data_with_labels, "division",
                division_croods_dict=div_dict,
            )
            mock_div.assert_called_once()

    def test_no_labels_raises(self, service, map_data_no_labels):
        with pytest.raises(NoLabelsError):
            service.room_edit(
                map_data_no_labels, "division",
                division_croods_dict={"id": "ROOM_001", "A": [0, 0], "B": [1, 1]},
            )


# ==================== TestMergeRoute ====================

class TestMergeRoute:
    def test_routes_to_merger(self, service, map_data_with_labels):
        merge_list = ["ROOM_001", "ROOM_002"]
        with patch.object(service.manual_merger, "process") as mock_merge:
            mock_merge.return_value = {"version": "v", "uuid": "", "data": []}
            service.room_edit(
                map_data_with_labels, "merge",
                room_merge_list=merge_list,
            )
            mock_merge.assert_called_once()

    def test_no_labels_raises(self, service, map_data_no_labels):
        with pytest.raises(NoLabelsError):
            service.room_edit(
                map_data_no_labels, "merge",
                room_merge_list=["ROOM_001", "ROOM_002"],
            )


# ==================== TestToolsCreation ====================

class TestToolsCreation:
    def test_transformer_receives_correct_params(self, service, map_data_no_labels):
        """验证 CoordinateTransformer 接收正确的参数"""
        with patch("app.services.services.CoordinateTransformer") as MockCT, \
             patch.object(service.auto_partitioner, "process") as mock_auto:
            mock_auto.return_value = {"version": "v", "uuid": "", "data": []}
            service.room_edit(map_data_no_labels, "split")
            MockCT.assert_called_once_with(0.05, [0.0, 0.0], 20)

    def test_graph_builder_receives_config(self, service, map_data_no_labels):
        """验证 RoomGraph 使用 config 初始化"""
        with patch("app.services.services.RoomGraph") as MockRG, \
             patch.object(service.auto_partitioner, "process") as mock_auto:
            mock_auto.return_value = {"version": "v", "uuid": "", "data": []}
            service.room_edit(map_data_no_labels, "split")
            MockRG.assert_called_once_with(service.config)


# ==================== TestEndToEnd ====================

class TestEndToEnd:
    def test_split_returns_labels_json(self, service, map_data_no_labels):
        """完整 split 流程返回 labels_json 格式"""
        expected = {"version": "online_4.0.2", "uuid": "test-uuid", "data": []}
        with patch.object(service.auto_partitioner, "process", return_value=expected):
            result = service.room_edit(map_data_no_labels, "split")
        assert result["version"] == "online_4.0.2"
        assert "data" in result

    def test_preprocessor_called_before_service(self, service, map_data_no_labels):
        """预处理器在服务之前被调用"""
        call_order = []

        def mock_preprocess(gray):
            call_order.append("preprocess")
            return {
                "map_data": np.full((20, 20), 200, dtype=np.uint8),
                "input_data": np.full((20, 20), 200, dtype=np.uint8),
            }

        def mock_auto_process(*args, **kwargs):
            call_order.append("auto_partition")
            return {"version": "v", "uuid": "", "data": []}

        service._mock_preprocessor.process.side_effect = mock_preprocess
        with patch.object(service.auto_partitioner, "process", side_effect=mock_auto_process):
            service.room_edit(map_data_no_labels, "split")

        assert call_order == ["preprocess", "auto_partition"]
