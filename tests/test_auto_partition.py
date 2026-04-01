"""AutoPartitioner 单元测试（pytest 风格）。"""

from unittest.mock import MagicMock, patch, PropertyMock

import numpy as np
import pytest

from app.services.auto_partition import AutoPartitioner


# ==================== fixtures ====================

@pytest.fixture
def config():
    return {
        "target_size": [512, 512],
        "normalize": True,
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225],
        "thickness": 2,
    }


@pytest.fixture
def partitioner(config):
    """无 Triton 的 AutoPartitioner"""
    return AutoPartitioner(config)


@pytest.fixture
def partitioner_with_triton(config):
    """带 Triton 的 AutoPartitioner"""
    config["triton_url"] = "localhost:8001"
    with patch("app.services.base_partitioner.Inferencer") as MockInf:
        mock_inf = MagicMock()
        MockInf.return_value = mock_inf
        p = AutoPartitioner(config)
        p._mock_inferencer = mock_inf  # 方便测试中访问
        yield p


@pytest.fixture
def simple_label_map():
    """简单的 3x3 label_map，两个房间"""
    lm = np.zeros((6, 6), dtype=np.int32)
    lm[0:3, 0:3] = 1
    lm[0:3, 3:6] = 2
    return lm


@pytest.fixture
def square_contour():
    """一个 10x10 正方形轮廓"""
    return np.array([[[0, 0]], [[10, 0]], [[10, 10]], [[0, 10]]], dtype=np.int32)


@pytest.fixture
def mock_transformer():
    t = MagicMock()
    t.contour_to_geometry.return_value = [0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0]
    t.world_to_pixel.return_value = (5, 5)
    t.rooms_data_to_contours.return_value = []
    return t


@pytest.fixture
def mock_graph_builder():
    g = MagicMock()
    g.build_graph.return_value = {0: []}
    g.assign_colors.return_value = {0: 0}
    g.find_start_room.return_value = 0
    g.dfs_sort.return_value = [0]
    return g


@pytest.fixture
def mock_landmark_builder():
    lb = MagicMock()
    lb.generate_landmarks.return_value = []
    return lb


# ==================== TestInit ====================

class TestInit:
    def test_default_config(self):
        p = AutoPartitioner()
        assert p.inferencer is None
        assert p.target_size == [512, 512]

    def test_custom_config(self, config):
        p = AutoPartitioner(config)
        assert p.normalize is True
        assert p.mean == [0.485, 0.456, 0.406]

    def test_triton_inferencer_created(self):
        with patch("app.services.base_partitioner.Inferencer") as MockInf:
            AutoPartitioner({"triton_url": "localhost:8001"})
            MockInf.assert_called_once()

    def test_no_triton_no_inferencer(self, partitioner):
        assert partitioner.inferencer is None


# ==================== TestPrepareTensor ====================

class TestPrepareTensor:
    def test_output_shape(self, partitioner):
        gray = np.full((100, 200), 128, dtype=np.uint8)
        meta = {}
        tensor = partitioner._prepare_tensor(gray, meta)
        assert tensor.shape == (1, 3, 512, 512)
        assert tensor.dtype == np.float32

    def test_meta_updated(self, partitioner):
        gray = np.full((100, 200), 128, dtype=np.uint8)
        meta = {}
        partitioner._prepare_tensor(gray, meta)
        assert "tensor_scale" in meta
        assert "tensor_pad" in meta
        assert "tensor_size" in meta
        assert meta["tensor_size"] == (512, 512)

    def test_scale_calculation(self, partitioner):
        gray = np.full((256, 512), 128, dtype=np.uint8)
        meta = {}
        partitioner._prepare_tensor(gray, meta)
        # scale = min(512/512, 512/256) = 1.0
        assert meta["tensor_scale"] == 1.0

    def test_bgr_input(self, partitioner):
        bgr = np.full((100, 100, 3), 128, dtype=np.uint8)
        meta = {}
        tensor = partitioner._prepare_tensor(bgr, meta)
        assert tensor.shape == (1, 3, 512, 512)

    def test_no_normalize(self, config):
        config["normalize"] = False
        p = AutoPartitioner(config)
        gray = np.full((100, 100), 128, dtype=np.uint8)
        meta = {}
        tensor = p._prepare_tensor(gray, meta)
        # 值应在 [0, 1] 范围 (只做 /255 没有减均值)
        assert tensor.min() >= 0.0
        assert tensor.max() <= 1.0


# ==================== TestRelabel ====================

class TestRelabel:
    def test_basic_relabel(self):
        lm = np.array([[0, 3, 0], [5, 5, 0], [0, 0, 0]], dtype=np.int32)
        result = AutoPartitioner._relabel(lm)
        assert set(result.flat) == {0, 1, 2}
        assert result[0, 1] == 1  # 原 3 → 1
        assert result[1, 0] == 2  # 原 5 → 2

    def test_already_continuous(self):
        lm = np.array([[0, 1], [2, 0]], dtype=np.int32)
        result = AutoPartitioner._relabel(lm)
        np.testing.assert_array_equal(result, lm)

    def test_empty_map(self):
        lm = np.zeros((3, 3), dtype=np.int32)
        result = AutoPartitioner._relabel(lm)
        assert result.max() == 0

    def test_single_room(self):
        lm = np.array([[0, 7], [7, 0]], dtype=np.int32)
        result = AutoPartitioner._relabel(lm)
        assert result.max() == 1
        assert result[0, 1] == 1


# ==================== TestExtractContours ====================

class TestExtractContours:
    def test_basic_extraction(self, simple_label_map):
        contours = AutoPartitioner._extract_contours(simple_label_map)
        assert len(contours) == 2

    def test_empty_map(self):
        lm = np.zeros((10, 10), dtype=np.int32)
        contours = AutoPartitioner._extract_contours(lm)
        assert contours == []

    def test_none_map(self):
        contours = AutoPartitioner._extract_contours(None)
        assert contours == []

    def test_contours_are_ndarray(self, simple_label_map):
        contours = AutoPartitioner._extract_contours(simple_label_map)
        for cnt in contours:
            assert isinstance(cnt, np.ndarray)


# ==================== TestContoursToLabelMap ====================

class TestContoursToLabelMap:
    def test_roundtrip(self):
        """轮廓 → label_map → 轮廓 数量应一致"""
        lm = np.zeros((50, 50), dtype=np.int32)
        lm[5:20, 5:20] = 1
        lm[5:20, 25:40] = 2
        contours = AutoPartitioner._extract_contours(lm)
        new_lm = AutoPartitioner._contours_to_label_map(contours, np.full((50, 50), 255, dtype=np.uint8), (50, 50))
        assert new_lm.max() >= 2

    def test_empty_contours(self):
        result = AutoPartitioner._contours_to_label_map([], np.full((10, 10), 255, dtype=np.uint8), (10, 10))
        assert result.max() == 0
        assert result.shape == (10, 10)


# ==================== TestToGray ====================

class TestToGray:
    def test_gray_passthrough(self):
        gray = np.full((10, 10), 128, dtype=np.uint8)
        result = AutoPartitioner._to_gray(gray)
        np.testing.assert_array_equal(result, gray)
        assert result is not gray  # 应该是 copy

    def test_bgr_conversion(self):
        bgr = np.full((10, 10, 3), 128, dtype=np.uint8)
        result = AutoPartitioner._to_gray(bgr)
        assert result.ndim == 2


# ==================== TestFallbackPartition ====================

class TestFallbackPartition:
    def test_basic_fallback(self, partitioner):
        """两个白色区域应被分割为两个连通域"""
        map_data = np.zeros((20, 40), dtype=np.uint8)
        map_data[2:8, 2:8] = 255
        map_data[2:8, 12:18] = 255
        with patch.object(partitioner.postprocessor, "_split_by_area") as mock_split, \
             patch.object(partitioner.postprocessor, "_merge_fragments") as mock_merge:
            mock_split.return_value = (
                np.zeros((20, 40), dtype=np.int32),
                np.zeros((20, 40), dtype=np.int32),
            )
            mock_merge.return_value = np.zeros((20, 40), dtype=np.int32)
            partitioner._fallback_partition(map_data)
            mock_split.assert_called_once()
            mock_merge.assert_called_once()

    def test_all_black_fallback(self, partitioner):
        """全黑地图: 使用 >0 的 fallback 阈值"""
        map_data = np.zeros((10, 10), dtype=np.uint8)
        with patch.object(partitioner.postprocessor, "_split_by_area") as mock_split, \
             patch.object(partitioner.postprocessor, "_merge_fragments") as mock_merge:
            mock_split.return_value = (
                np.zeros((10, 10), dtype=np.int32),
                np.zeros((10, 10), dtype=np.int32),
            )
            mock_merge.return_value = np.zeros((10, 10), dtype=np.int32)
            partitioner._fallback_partition(map_data)
            mock_split.assert_called_once()


# ==================== TestPartition ====================

class TestPartition:
    def test_partition_fallback_path(self, partitioner):
        """无 inferencer 时走 fallback"""
        map_data = np.zeros((20, 20), dtype=np.uint8)
        map_data[2:8, 2:8] = 255
        map_data_dict = {"input_img": map_data}

        mock_polygons = [[[2, 2], [8, 2], [8, 8], [2, 8]]]
        with patch.object(partitioner, "_fallback_partition") as mock_fb:
            mock_fb.return_value = mock_polygons
            result = partitioner.partition(map_data_dict, extend=False)
            mock_fb.assert_called_once()
            assert isinstance(result, list)

    def test_partition_infer_path(self, partitioner_with_triton):
        p = partitioner_with_triton
        p._mock_inferencer.is_ready.return_value = True

        mock_polygons = [[[0, 0], [10, 0], [10, 5], [0, 5]],
                         [[0, 5], [10, 5], [10, 10], [0, 10]]]
        with patch.object(p, "_infer_partition") as mock_infer:
            mock_infer.return_value = mock_polygons
            map_data_dict = {"input_img": np.zeros((10, 10), dtype=np.uint8)}
            result = p.partition(map_data_dict, extend=False)
            mock_infer.assert_called_once()
            assert isinstance(result, list)
            assert len(result) == 2

    def test_partition_with_extend(self, partitioner):
        """extend=True 时走 fallback 路径"""
        map_data_dict = {"input_img": np.zeros((10, 10), dtype=np.uint8)}
        mock_polygons = [[[0, 0], [1, 0], [1, 1], [0, 1]]]
        with patch.object(partitioner, "_fallback_partition") as mock_fb:
            mock_fb.return_value = mock_polygons
            result = partitioner.partition(map_data_dict, extend=True)
            mock_fb.assert_called_once()
            assert isinstance(result, list)


# ==================== TestExpandContours ====================

class TestExpandContours:
    def test_delegates_to_expander(self, partitioner, square_contour):
        map_img = np.zeros((20, 20), dtype=np.uint8)
        with patch("app.services.auto_partition.ContourExpander") as MockExp:
            mock_exp = MagicMock()
            mock_exp.expand.return_value = [square_contour]
            MockExp.return_value = mock_exp
            result = partitioner.expand_contours([square_contour], map_img)
            mock_exp.expand.assert_called_once()
            assert len(result) == 1


# ==================== TestBuildGraphAndColors ====================

class TestBuildGraphAndColors:
    def test_returns_graph_and_colors(self, partitioner, mock_graph_builder):
        contours = [np.array([[[0, 0]], [[5, 5]]], dtype=np.int32)]
        map_img = np.zeros((10, 10), dtype=np.uint8)
        graph, colors = partitioner.build_graph_and_colors(
            contours, map_img, mock_graph_builder
        )
        assert isinstance(graph, dict)
        assert isinstance(colors, dict)
        mock_graph_builder.build_graph.assert_called_once()
        mock_graph_builder.assign_colors.assert_called_once()


# ==================== TestSortContours ====================

class TestSortContours:
    def test_basic_sort(self, partitioner, mock_graph_builder):
        cnt0 = np.array([[[0, 0]], [[5, 0]], [[5, 5]], [[0, 5]]], dtype=np.int32)
        cnt1 = np.array([[[10, 0]], [[15, 0]], [[15, 5]], [[10, 5]]], dtype=np.int32)
        contours = [cnt0, cnt1]
        graph = {0: [1], 1: [0]}

        sorted_c, order = partitioner.sort_contours(
            contours, graph, mock_graph_builder
        )
        assert len(sorted_c) == 2
        assert len(order) == 2

    def test_with_charge_pixel(self, partitioner, mock_graph_builder):
        cnt0 = np.array([[[0, 0]], [[5, 5]]], dtype=np.int32)
        contours = [cnt0]
        graph = {0: []}
        mock_graph_builder.dfs_sort.return_value = [0]

        _, order = partitioner.sort_contours(
            contours, graph, mock_graph_builder,
            charge_pixel=(3, 3),
        )
        mock_graph_builder.find_start_room.assert_called_once()
        assert 0 in order

    def test_empty_dfs_fallback(self, partitioner, mock_graph_builder):
        """dfs_sort 返回空列表时的兜底"""
        cnt0 = np.array([[[0, 0]], [[5, 5]]], dtype=np.int32)
        cnt1 = np.array([[[10, 0]], [[15, 5]]], dtype=np.int32)
        contours = [cnt0, cnt1]
        graph = {0: [], 1: []}
        mock_graph_builder.dfs_sort.return_value = []

        _, order = partitioner.sort_contours(contours, graph, mock_graph_builder)
        assert order == [0, 1]

    def test_missing_rooms_appended(self, partitioner, mock_graph_builder):
        """dfs_sort 遗漏的房间追加到末尾"""
        cnt0 = np.array([[[0, 0]], [[5, 5]]], dtype=np.int32)
        cnt1 = np.array([[[10, 0]], [[15, 5]]], dtype=np.int32)
        cnt2 = np.array([[[20, 0]], [[25, 5]]], dtype=np.int32)
        contours = [cnt0, cnt1, cnt2]
        graph = {0: [1], 1: [0], 2: []}
        mock_graph_builder.dfs_sort.return_value = [0, 1]

        _, order = partitioner.sort_contours(contours, graph, mock_graph_builder)
        assert 2 in order
        assert len(order) == 3


# ==================== TestSerializeContours ====================

class TestSerializeContours:
    def test_basic_serialization(self, partitioner, mock_transformer):
        cnt0 = np.array([[[0, 0]], [[5, 0]], [[5, 5]], [[0, 5]]], dtype=np.int32)
        cnt1 = np.array([[[10, 0]], [[15, 0]], [[15, 5]], [[10, 5]]], dtype=np.int32)
        contours = [cnt0, cnt1]
        graph = {0: [1], 1: [0]}
        colors = {0: 0, 1: 1}
        order = [0, 1]

        result = partitioner.serialize_contours(
            contours, graph, colors, order, mock_transformer
        )
        assert len(result) == 2
        assert result[0]["name"] == "A"
        assert result[0]["id"] == "ROOM_001"
        assert result[1]["name"] == "B"
        assert result[1]["id"] == "ROOM_002"
        assert result[0]["type"] == "polygon"
        assert "geometry" in result[0]
        assert "colorType" in result[0]
        assert "graph" in result[0]

    def test_reordered_graph(self, partitioner, mock_transformer):
        """order=[1,0] 时 graph 索引应正确重映射"""
        cnt0 = np.array([[[0, 0]], [[5, 5]]], dtype=np.int32)
        cnt1 = np.array([[[10, 0]], [[15, 5]]], dtype=np.int32)
        contours = [cnt1, cnt0]  # 排序后: old_1 在前, old_0 在后
        graph = {0: [1], 1: [0]}
        colors = {0: 2, 1: 3}
        order = [1, 0]

        result = partitioner.serialize_contours(
            contours, graph, colors, order, mock_transformer
        )
        # new_idx=0 对应 old_idx=1, 邻居 old_0 → new_1
        assert result[0]["graph"] == [1]
        assert result[0]["colorType"] == 3  # colors[old_idx=1]

    def test_name_wrapping(self, partitioner, mock_transformer):
        """超过 26 个房间时名称循环"""
        contours = [np.array([[[0, 0]], [[1, 1]]], dtype=np.int32)] * 27
        graph = {i: [] for i in range(27)}
        colors = {i: 0 for i in range(27)}
        order = list(range(27))

        result = partitioner.serialize_contours(
            contours, graph, colors, order, mock_transformer
        )
        assert result[0]["name"] == "A"
        assert result[25]["name"] == "Z"
        assert result[26]["name"] == "A1"


# ==================== TestBeautifyContours ====================

class TestBeautifyContours:
    def test_disabled_by_default(self, partitioner):
        result = partitioner.beautify_contours([], np.zeros((10, 10), dtype=np.uint8))
        assert result == (None, None)

    def test_enabled_delegates(self, partitioner):
        partitioner.set_beautifier_status(True)
        with patch("app.services.auto_partition.ContourBeautifier") as MockB:
            mock_b = MagicMock()
            mock_b.beautify.return_value = ([[1, 2]], [[3, 4]])
            MockB.return_value = mock_b
            bbox, thresh = partitioner.beautify_contours(
                [], np.zeros((10, 10), dtype=np.uint8)
            )
            assert bbox == [[1, 2]]
            assert thresh == [[3, 4]]


# ==================== TestBuildLandmarks ====================

class TestBuildLandmarks:
    def test_basic(self, partitioner, mock_landmark_builder):
        rooms_data = [
            {"name": "A", "id": "ROOM_001", "geometry": [0, 0, 1, 1]},
        ]
        result = partitioner.build_landmarks(rooms_data, mock_landmark_builder)
        mock_landmark_builder.generate_landmarks.assert_called_once()
        assert isinstance(result, list)

    def test_with_marker_polygons(self, partitioner, mock_landmark_builder):
        rooms_data = [
            {"name": "A", "id": "ROOM_001", "geometry": [0, 0, 1, 1]},
        ]
        markers = [[0.5, 0.5, 0.6, 0.6]]
        partitioner.build_landmarks(rooms_data, mock_landmark_builder, marker_polygons=markers)
        call_kwargs = mock_landmark_builder.generate_landmarks.call_args
        assert call_kwargs[1]["marker_polygons"] == markers


# ==================== TestGetChargePixel ====================

class TestGetChargePixel:
    def test_no_charge_pose(self):
        result = AutoPartitioner._get_charge_pixel({}, MagicMock())
        assert result is None

    def test_zero_pose(self):
        result = AutoPartitioner._get_charge_pixel(
            {"world_charge_pose": [0, 0, 0]}, MagicMock()
        )
        assert result is None

    def test_valid_pose(self, mock_transformer):
        result = AutoPartitioner._get_charge_pixel(
            {"world_charge_pose": [1.0, 2.0, 0.0]}, mock_transformer
        )
        assert result == (5, 5)
        mock_transformer.world_to_pixel.assert_called_once_with(1.0, 2.0)


# ==================== TestGetMarkerPolygons ====================

class TestGetMarkerPolygons:
    def test_no_markers(self):
        assert AutoPartitioner._get_marker_polygons({}) is None

    def test_empty_data(self):
        assert AutoPartitioner._get_marker_polygons({"markers_json": {"data": []}}) is None

    def test_furniture_markers(self):
        markers_json = {
            "data": [
                {"name": "家具", "type": "polygon", "geometry": [1, 2, 3, 4]},
                {"name": "其他", "type": "point", "geometry": [5, 6]},
            ]
        }
        result = AutoPartitioner._get_marker_polygons({"markers_json": markers_json})
        assert result == [[1, 2, 3, 4]]


# ==================== TestGetRoomInfo ====================

class TestGetRoomInfo:
    def test_basic_info(self, simple_label_map):
        rooms = AutoPartitioner.get_room_info(simple_label_map, resolution=1.0)
        assert len(rooms) == 2
        assert rooms[0]["id"] == 1
        assert rooms[0]["area"] == 9.0  # 3x3 pixels

    def test_empty_map(self):
        rooms = AutoPartitioner.get_room_info(np.zeros((5, 5), dtype=np.int32))
        assert rooms == []

    def test_none_map(self):
        rooms = AutoPartitioner.get_room_info(None)
        assert rooms == []


# ==================== TestProcess ====================

class TestProcess:
    def test_full_pipeline_first_partition(
        self, partitioner, mock_transformer, mock_graph_builder, mock_landmark_builder
    ):
        """首次分区完整流程"""
        map_img = np.full((20, 20), 200, dtype=np.uint8)
        map_data = {
            "map_img": map_img,
            "resolution": 0.05,
            "origin": [0, 0],
            "uuid": "test-uuid",
            "robot_model": "s10",
            "cleaned_img": np.full((20, 20), 200, dtype=np.uint8),
            "cleaned_img2": np.full((20, 20), 200, dtype=np.uint8),
            "input_img": np.full((20, 20), 200, dtype=np.uint8),
        }

        # Mock partition 返回多边形列表
        polygon = [[2, 2], [8, 2], [8, 8], [2, 8]]
        with patch.object(partitioner, "partition") as mock_part, \
             patch.object(partitioner, "beautify_contours") as mock_beauty:
            mock_part.return_value = [polygon]
            mock_beauty.return_value = (None, None)

            result = partitioner.process(
                map_data, mock_transformer,
                mock_graph_builder, mock_landmark_builder,
            )

        assert "version" in result
        assert "uuid" in result
        assert result["uuid"] == "test-uuid"
        assert "data" in result
        assert len(result["data"]) > 0

    def test_repartition_forces_detect(
        self, partitioner, mock_transformer, mock_graph_builder, mock_landmark_builder
    ):
        """repartition=True 时即使有 labels_json 也重新分区"""
        map_img = np.full((20, 20), 200, dtype=np.uint8)
        map_data = {
            "map_img": map_img,
            "labels_json": {"data": [{"id": "ROOM_001"}]},
            "robot_model": "s10",
            "input_img": map_img,
            "cleaned_img": map_img,
            "cleaned_img2": map_img,
        }

        polygon = [[2, 2], [8, 2], [8, 8], [2, 8]]
        with patch.object(partitioner, "partition") as mock_part, \
             patch.object(partitioner, "beautify_contours") as mock_beauty:
            mock_part.return_value = [polygon]
            mock_beauty.return_value = (None, None)

            partitioner.process(
                map_data, mock_transformer,
                mock_graph_builder, mock_landmark_builder,
                repartition=True,
            )
            mock_part.assert_called_once()

    def test_existing_labels_still_partitions(
        self, partitioner, mock_transformer, mock_graph_builder, mock_landmark_builder
    ):
        """有 labels_json 时 AutoPartitioner 仍执行 partition()（扩展分区由 ExtendedPartitioner 负责）"""
        map_img = np.full((20, 20), 200, dtype=np.uint8)
        labels_json = {
            "data": [
                {"id": "ROOM_001", "name": "A", "geometry": [0, 0, 1, 0, 1, 1, 0, 1, 0, 0]},
            ]
        }
        map_data = {
            "map_img": map_img,
            "labels_json": labels_json,
            "robot_model": "s10",
            "input_img": map_img,
            "cleaned_img": map_img,
            "cleaned_img2": map_img,
        }

        polygon = [[2, 2], [8, 2], [8, 8], [2, 8]]

        with patch.object(partitioner, "partition") as mock_part, \
             patch.object(partitioner, "beautify_contours") as mock_beauty:
            mock_part.return_value = [polygon]
            mock_beauty.return_value = (None, None)

            partitioner.process(
                map_data, mock_transformer,
                mock_graph_builder, mock_landmark_builder,
                extend=True,
            )
            mock_part.assert_called_once()

    def test_k20pro_generates_landmarks(
        self, partitioner, mock_transformer, mock_graph_builder, mock_landmark_builder
    ):
        """K20PRO 机器人生成 landmark"""
        map_img = np.full((20, 20), 200, dtype=np.uint8)
        map_data = {
            "map_img": map_img,
            "robot_model": "S-K20PRO",
            "uuid": "k20-uuid",
            "input_img": map_img,
            "cleaned_img": map_img,
            "cleaned_img2": map_img,
        }

        polygon = [[2, 2], [8, 2], [8, 8], [2, 8]]
        mock_landmark_builder.generate_landmarks.return_value = [
            {"type": "pose", "id": "PLATFORM_LANDMARK_001"}
        ]

        with patch.object(partitioner, "partition") as mock_part, \
             patch.object(partitioner, "beautify_contours") as mock_beauty:
            mock_part.return_value = [polygon]
            mock_beauty.return_value = (None, None)

            result = partitioner.process(
                map_data, mock_transformer,
                mock_graph_builder, mock_landmark_builder,
            )

        mock_landmark_builder.generate_landmarks.assert_called_once()
        # data 中应包含 landmark
        has_landmark = any(
            d.get("type") == "pose" for d in result["data"]
        )
        assert has_landmark
