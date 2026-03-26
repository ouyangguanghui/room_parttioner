"""ExtendedPartitioner 单元测试（pytest 风格）。"""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from app.services.extended_partition import ExtendedPartitioner


# ==================== fixtures ====================

@pytest.fixture
def config():
    return {
        "door_width": 20,
        "grow_iterations": 10,
        "wall_threshold": 128,
        "resolution": 0.05,
        "min_room_area": 1.0,
        "min_new_region_area": 50,
        "merge_area_threshold": 0,
        "merge_ratio_threshold": 0.6,
    }


@pytest.fixture
def ep(config):
    return ExtendedPartitioner(config)


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


def _make_two_room_label_map(h=50, w=100):
    """两个房间的 label_map: room1 左半, room2 右半"""
    lm = np.zeros((h, w), dtype=np.int32)
    lm[5:45, 5:45] = 1
    lm[5:45, 55:95] = 2
    return lm


def _make_grid_map(h=50, w=100, free_val=255):
    """全自由空间 grid_map"""
    return np.full((h, w), free_val, dtype=np.uint8)


# ==================== TestInit ====================

class TestInit:
    def test_default_config(self):
        ep = ExtendedPartitioner()
        assert ep.door_width == 20
        assert ep.grow_iterations == 10
        assert ep.wall_threshold == 128

    def test_custom_config(self, config):
        config["door_width"] = 30
        ep = ExtendedPartitioner(config)
        assert ep.door_width == 30

    def test_merge_thresholds(self, config):
        config["merge_area_threshold"] = 2.0
        config["merge_ratio_threshold"] = 0.8
        ep = ExtendedPartitioner(config)
        assert ep.merge_area_threshold == 2.0
        assert ep.merge_ratio_threshold == 0.8


# ==================== TestSplitByDoorway ====================

class TestSplitByDoorway:
    def test_no_split_solid_region(self, ep):
        """实心区域不应被分裂"""
        lm = np.zeros((100, 100), dtype=np.int32)
        lm[10:90, 10:90] = 1
        gm = np.full((100, 100), 255, dtype=np.uint8)
        result = ep.split_by_doorway(lm, gm)
        # 应仍然只有 1 个区域
        unique = set(result.flat) - {0}
        assert len(unique) == 1

    def test_narrow_passage_splits(self):
        """窄通道连接的两个大区域应被分裂"""
        ep = ExtendedPartitioner({"door_width": 10, "grow_iterations": 20})
        lm = np.zeros((100, 200), dtype=np.int32)
        # 左侧大区域
        lm[10:90, 10:80] = 1
        # 窄通道 (5px 宽)
        lm[45:50, 80:120] = 1
        # 右侧大区域
        lm[10:90, 120:190] = 1
        gm = np.full((100, 200), 255, dtype=np.uint8)
        result = ep.split_by_doorway(lm, gm)
        unique = set(result.flat) - {0}
        assert len(unique) >= 2

    def test_empty_map(self, ep):
        lm = np.zeros((10, 10), dtype=np.int32)
        gm = np.full((10, 10), 255, dtype=np.uint8)
        result = ep.split_by_doorway(lm, gm)
        assert result.max() == 0


# ==================== TestGrowUnassigned ====================

class TestGrowUnassigned:
    def test_fills_gap(self, ep):
        """两个房间之间的小间隙应被填充"""
        lm = np.zeros((20, 40), dtype=np.int32)
        lm[2:18, 2:15] = 1
        lm[2:18, 17:38] = 2  # 2px gap at x=15-17
        gm = np.full((20, 40), 255, dtype=np.uint8)

        result = ep.grow_unassigned(lm, gm)
        # gap 区域应被填充
        gap_values = result[5, 15:17]
        assert all(v > 0 for v in gap_values)

    def test_no_unassigned(self, ep):
        """无未分配像素时返回原图"""
        lm = np.ones((10, 10), dtype=np.int32)
        gm = np.full((10, 10), 255, dtype=np.uint8)
        result = ep.grow_unassigned(lm, gm)
        np.testing.assert_array_equal(result, lm)

    def test_respects_walls(self, ep):
        """墙壁像素不应被填充"""
        lm = np.zeros((20, 20), dtype=np.int32)
        lm[2:8, 2:8] = 1
        gm = np.zeros((20, 20), dtype=np.uint8)  # 全墙壁
        gm[2:8, 2:8] = 255  # 只有房间是自由空间

        result = ep.grow_unassigned(lm, gm)
        # 墙壁区域应仍为 0
        assert result[0, 0] == 0
        assert result[15, 15] == 0


# ==================== TestExtendPixel ====================

class TestExtendPixel:
    def test_combines_split_and_grow(self, ep):
        """extend_pixel 应依次执行 split + grow"""
        lm = np.zeros((20, 20), dtype=np.int32)
        lm[2:18, 2:18] = 1
        gm = np.full((20, 20), 255, dtype=np.uint8)

        with patch.object(ep, "split_by_doorway") as mock_split, \
             patch.object(ep, "grow_unassigned") as mock_grow:
            mock_split.return_value = lm
            mock_grow.return_value = lm
            ep.extend_pixel(lm, gm)
            mock_split.assert_called_once()
            mock_grow.assert_called_once()

    def test_extend_alias(self, ep):
        """extend() 应等价于 extend_pixel()"""
        lm = np.zeros((10, 10), dtype=np.int32)
        gm = np.full((10, 10), 255, dtype=np.uint8)
        with patch.object(ep, "extend_pixel") as mock_ext:
            mock_ext.return_value = lm
            ep.extend(lm, gm)
            mock_ext.assert_called_once_with(lm, gm)


# ==================== TestRegionGrow ====================

class TestRegionGrow:
    def test_grows_from_seeds(self, ep):
        """种子应向外膨胀填满 mask"""
        mask = np.zeros((20, 20), dtype=np.uint8)
        mask[5:15, 5:15] = 1

        seeds = np.zeros((20, 20), dtype=np.int32)
        seeds[9, 9] = 1  # 中心种子

        result = ep._region_grow(mask, seeds, 2)
        # mask 区域应被完全填充
        assert (result[mask > 0] > 0).all()


# ==================== TestDetectNewRegions ====================

class TestDetectNewRegions:
    def test_detects_new_area(self, ep):
        """label_map 未覆盖的自由区域应被检测到"""
        lm = np.zeros((50, 100), dtype=np.int32)
        lm[5:45, 5:45] = 1  # 已有一个房间

        gm = np.full((50, 100), 255, dtype=np.uint8)
        gm[0:5, :] = 0  # 顶部墙壁
        # x=55-95 有大片自由空间未被标记

        result = ep.detect_new_regions(lm, gm)
        assert result.max() >= 1  # 至少检测到 1 个新区域

    def test_no_new_regions(self, ep):
        """所有自由空间都已标记时不应检测到新区域"""
        lm = np.ones((20, 20), dtype=np.int32)
        gm = np.full((20, 20), 255, dtype=np.uint8)

        result = ep.detect_new_regions(lm, gm)
        assert result.max() == 0

    def test_filters_small_regions(self):
        """面积低于 min_new_region_area 的碎片应被过滤"""
        ep = ExtendedPartitioner({"min_new_region_area": 100, "wall_threshold": 128})
        lm = np.ones((50, 50), dtype=np.int32)
        # 在角落留一个 5x5 的小空白
        lm[0:5, 0:5] = 0
        gm = np.full((50, 50), 255, dtype=np.uint8)

        result = ep.detect_new_regions(lm, gm)
        assert result.max() == 0  # 5x5=25 < 100，应被过滤

    def test_all_walls(self, ep):
        """全墙壁不应检测到新区域"""
        lm = np.zeros((20, 20), dtype=np.int32)
        gm = np.zeros((20, 20), dtype=np.uint8)  # 全墙壁

        result = ep.detect_new_regions(lm, gm)
        assert result.max() == 0

    def test_multiple_regions(self):
        """多个不连通的新区域应分别标记"""
        ep = ExtendedPartitioner({"min_new_region_area": 10, "wall_threshold": 128})
        lm = np.zeros((50, 100), dtype=np.int32)
        lm[20:30, 40:60] = 1  # 中间有个已有房间

        gm = np.zeros((50, 100), dtype=np.uint8)  # 全墙壁
        # 左侧自由空间 (被墙壁隔开)
        gm[20:30, 10:30] = 255
        # 右侧自由空间
        gm[20:30, 70:90] = 255

        result = ep.detect_new_regions(lm, gm)
        assert result.max() >= 2  # 至少两个新区域


# ==================== TestClassifyRegion ====================

class TestClassifyRegion:
    def test_small_area_merges(self):
        """小面积区域应合并到相邻房间"""
        ep = ExtendedPartitioner({
            "resolution": 0.05,
            "min_room_area": 1.0,
            "merge_area_threshold": 0,
            "merge_ratio_threshold": 0.6,
            "wall_threshold": 128,
        })

        lm = np.zeros((100, 100), dtype=np.int32)
        lm[10:50, 10:50] = 1  # 大房间

        # 小新区域紧贴房间右边
        region_mask = np.zeros((100, 100), dtype=bool)
        region_mask[20:30, 50:55] = True  # 10x5 = 50 pixels = 0.125m²

        action, target = ep.classify_region(region_mask, lm)
        assert action == "merge"
        assert target == 1

    def test_large_isolated_area_new_room(self):
        """大面积且无显著接触的区域应创建新房间"""
        ep = ExtendedPartitioner({
            "resolution": 0.05,
            "min_room_area": 0.1,
            "merge_area_threshold": 0,
            "merge_ratio_threshold": 0.6,
            "wall_threshold": 128,
        })

        lm = np.zeros((200, 200), dtype=np.int32)
        lm[10:50, 10:50] = 1  # 房间在左上角

        # 大区域远离已有房间
        region_mask = np.zeros((200, 200), dtype=bool)
        region_mask[100:180, 100:180] = True  # 80x80 = 6400 pixels = 16m²

        action, target = ep.classify_region(region_mask, lm)
        assert action == "new"
        assert target == 0

    def test_no_adjacent_rooms(self):
        """完全孤立的区域应创建新房间"""
        ep = ExtendedPartitioner({
            "resolution": 0.05,
            "min_room_area": 1.0,
            "merge_area_threshold": 0,
            "merge_ratio_threshold": 0.6,
            "wall_threshold": 128,
        })

        lm = np.zeros((100, 100), dtype=np.int32)
        # 无已有房间

        region_mask = np.zeros((100, 100), dtype=bool)
        region_mask[40:60, 40:60] = True

        action, target = ep.classify_region(region_mask, lm)
        assert action == "new"
        assert target == 0

    def test_high_contact_ratio_merges(self):
        """高接触比例的区域应合并，即使面积大"""
        ep = ExtendedPartitioner({
            "resolution": 0.05,
            "min_room_area": 0.01,
            "merge_area_threshold": 0.01,
            "merge_ratio_threshold": 0.3,
            "wall_threshold": 128,
        })

        lm = np.zeros((100, 100), dtype=np.int32)
        lm[10:50, 10:50] = 1

        # 长条形新区域紧贴房间右侧（高接触比例）
        region_mask = np.zeros((100, 100), dtype=bool)
        region_mask[10:50, 50:55] = True  # 40x5 = 200px, 接触边 40px

        action, target = ep.classify_region(region_mask, lm)
        assert action == "merge"
        assert target == 1


# ==================== TestAssignNewRegions ====================

class TestAssignNewRegions:
    def test_merge_assignment(self, ep):
        """小区域应被合并到相邻房间"""
        lm = np.zeros((50, 50), dtype=np.int32)
        lm[5:25, 5:25] = 1

        new_regions = np.zeros((50, 50), dtype=np.int32)
        new_regions[15:20, 25:28] = 1  # 紧贴房间 1 的小区域

        with patch.object(ep, "classify_region", return_value=("merge", 1)):
            result = ep.assign_new_regions(lm, new_regions)
            assert result[17, 26] == 1  # 应被合并到房间 1

    def test_new_room_assignment(self, ep):
        """大区域应获得新 label"""
        lm = np.zeros((50, 50), dtype=np.int32)
        lm[5:20, 5:20] = 1

        new_regions = np.zeros((50, 50), dtype=np.int32)
        new_regions[30:45, 30:45] = 1

        with patch.object(ep, "classify_region", return_value=("new", 0)):
            result = ep.assign_new_regions(lm, new_regions)
            new_val = result[35, 35]
            assert new_val == 2  # 新 label = max(1) + 1 = 2

    def test_no_new_regions(self, ep):
        """无新区域时返回原 label_map"""
        lm = np.ones((10, 10), dtype=np.int32)
        new_regions = np.zeros((10, 10), dtype=np.int32)

        result = ep.assign_new_regions(lm, new_regions)
        np.testing.assert_array_equal(result, lm)

    def test_mixed_assignments(self, ep):
        """混合场景：部分合并 + 部分新建"""
        lm = np.zeros((60, 60), dtype=np.int32)
        lm[5:25, 5:25] = 1

        new_regions = np.zeros((60, 60), dtype=np.int32)
        new_regions[15:20, 25:28] = 1  # 区域 1: 小, 紧贴
        new_regions[40:55, 40:55] = 2  # 区域 2: 大, 远离

        def mock_classify(mask, label_map):
            if mask[17, 26]:
                return ("merge", 1)
            return ("new", 0)

        with patch.object(ep, "classify_region", side_effect=mock_classify):
            result = ep.assign_new_regions(lm, new_regions)
            assert result[17, 26] == 1  # 合并到房间 1
            assert result[45, 45] == 2  # 新房间 label=2


# ==================== TestBuildOldRoomMapping ====================

class TestBuildOldRoomMapping:
    def test_matching_rooms(self, ep):
        """重叠像素多的新 label 应映射到对应旧房间"""
        old_lm = np.zeros((50, 50), dtype=np.int32)
        old_lm[5:25, 5:25] = 1
        old_lm[5:25, 30:50] = 2

        new_lm = np.zeros((50, 50), dtype=np.int32)
        new_lm[3:27, 3:27] = 1  # 大部分重叠旧 room 1
        new_lm[3:27, 28:50] = 2  # 大部分重叠旧 room 2

        old_rooms = [
            {"name": "A", "id": "ROOM_001"},
            {"name": "B", "id": "ROOM_002"},
        ]

        mapping = ep._build_old_room_mapping(old_lm, new_lm, old_rooms)
        assert mapping[1] == 0  # new label 1 → old room idx 0
        assert mapping[2] == 1  # new label 2 → old room idx 1

    def test_new_room_not_in_mapping(self, ep):
        """完全新增的房间不应出现在映射中"""
        old_lm = np.zeros((50, 50), dtype=np.int32)
        old_lm[5:25, 5:25] = 1

        new_lm = np.zeros((50, 50), dtype=np.int32)
        new_lm[5:25, 5:25] = 1
        new_lm[30:45, 30:45] = 2  # 完全新增的区域

        old_rooms = [{"name": "A", "id": "ROOM_001"}]

        mapping = ep._build_old_room_mapping(old_lm, new_lm, old_rooms)
        assert 1 in mapping
        assert 2 not in mapping  # 新房间没有映射


# ==================== TestSerializeContours ====================

class TestSerializeContours:
    def test_preserves_old_metadata(self, ep, mock_transformer):
        """已有房间应保留原 name/id/groundMaterial"""
        cnt = np.array([[[5, 5]], [[20, 5]], [[20, 20]], [[5, 20]]], dtype=np.int32)
        contours = [cnt]
        graph = {0: []}
        colors = {0: 2}
        order = [0]
        old_rooms = [
            {"name": "客厅", "id": "ROOM_001", "type": "polygon",
             "colorType": 3, "groundMaterial": "wood"},
        ]
        old_mapping = {1: 0}  # label 1 → old room idx 0

        result = ep.serialize_contours(
            contours, graph, colors, order,
            mock_transformer, old_rooms, old_mapping,
        )
        assert len(result) == 1
        assert result[0]["name"] == "客厅"
        assert result[0]["id"] == "ROOM_001"
        assert result[0]["groundMaterial"] == "wood"

    def test_new_room_gets_new_id(self, ep, mock_transformer):
        """新房间应获得新分配的 name/id"""
        cnt0 = np.array([[[5, 5]], [[20, 5]], [[20, 20]], [[5, 20]]], dtype=np.int32)
        cnt1 = np.array([[[30, 5]], [[45, 5]], [[45, 20]], [[30, 20]]], dtype=np.int32)
        contours = [cnt0, cnt1]
        graph = {0: [1], 1: [0]}
        colors = {0: 0, 1: 1}
        order = [0, 1]
        old_rooms = [
            {"name": "A", "id": "ROOM_001", "type": "polygon",
             "colorType": 0, "groundMaterial": None},
        ]
        old_mapping = {1: 0}  # label 1 → old, label 2 → new

        result = ep.serialize_contours(
            contours, graph, colors, order,
            mock_transformer, old_rooms, old_mapping,
        )
        assert len(result) == 2
        assert result[0]["id"] == "ROOM_001"
        assert result[1]["id"] == "ROOM_002"  # 新分配
        assert result[1]["name"] != "A"  # 不同于已有名称


# ==================== TestRelabel ====================

class TestRelabel:
    def test_basic(self):
        lm = np.array([[0, 3, 0], [5, 5, 0]], dtype=np.int32)
        result = ExtendedPartitioner._relabel(lm)
        assert set(result.flat) == {0, 1, 2}

    def test_empty(self):
        lm = np.zeros((3, 3), dtype=np.int32)
        result = ExtendedPartitioner._relabel(lm)
        assert result.max() == 0


# ==================== TestStaticUtils ====================

class TestStaticUtils:
    def test_extract_contours(self):
        lm = np.zeros((30, 30), dtype=np.int32)
        lm[5:15, 5:15] = 1
        lm[5:15, 20:28] = 2
        contours = ExtendedPartitioner._extract_contours(lm)
        assert len(contours) == 2

    def test_extract_contours_empty(self):
        assert ExtendedPartitioner._extract_contours(np.zeros((10, 10), dtype=np.int32)) == []

    def test_contours_to_label_map(self):
        cnt = np.array([[[5, 5]], [[15, 5]], [[15, 15]], [[5, 15]]], dtype=np.int32)
        lm = ExtendedPartitioner._contours_to_label_map([cnt], (20, 20))
        assert lm.max() == 1
        assert lm[10, 10] == 1

    def test_to_gray_passthrough(self):
        gray = np.full((10, 10), 128, dtype=np.uint8)
        result = ExtendedPartitioner._to_gray(gray)
        assert result.ndim == 2

    def test_to_gray_bgr(self):
        bgr = np.full((10, 10, 3), 128, dtype=np.uint8)
        result = ExtendedPartitioner._to_gray(bgr)
        assert result.ndim == 2

    def test_get_charge_pixel_none(self):
        assert ExtendedPartitioner._get_charge_pixel({}, MagicMock()) is None

    def test_get_charge_pixel_valid(self):
        t = MagicMock()
        t.world_to_pixel.return_value = (10, 20)
        result = ExtendedPartitioner._get_charge_pixel(
            {"world_charge_pose": [1.0, 2.0, 0]}, t
        )
        assert result == (10, 20)

    def test_get_marker_polygons_none(self):
        assert ExtendedPartitioner._get_marker_polygons({}) is None

    def test_get_marker_polygons(self):
        md = {"markers_json": {"data": [
            {"name": "家具", "type": "polygon", "geometry": [1, 2]},
        ]}}
        assert ExtendedPartitioner._get_marker_polygons(md) == [[1, 2]]


# ==================== TestProcess ====================

class TestProcess:
    def test_full_pipeline_no_new_regions(
        self, ep, mock_transformer, mock_graph_builder, mock_landmark_builder
    ):
        """无新区域时仅执行扩展优化"""
        map_img = np.full((30, 30), 200, dtype=np.uint8)
        cnt = np.array([[[5, 5]], [[20, 5]], [[20, 20]], [[5, 20]]], dtype=np.int32)
        mock_transformer.rooms_data_to_contours.return_value = [cnt]

        map_data = {
            "map_img": map_img,
            "robot_model": "s10",
            "uuid": "test-uuid",
            "labels_json": {
                "data": [
                    {"name": "A", "id": "ROOM_001", "type": "polygon",
                     "geometry": [0, 0, 1, 0, 1, 1, 0, 1, 0, 0],
                     "colorType": 0, "graph": [], "groundMaterial": None},
                ]
            },
        }
        meta = {"map_data": map_img, "input_data": map_img}

        with patch.object(ep, "detect_new_regions") as mock_detect, \
             patch.object(ep, "extend_pixel") as mock_ext, \
             patch("app.services.extended_partition.ContourExpander") as MockExp:

            mock_detect.return_value = np.zeros((30, 30), dtype=np.int32)

            lm = np.zeros((30, 30), dtype=np.int32)
            lm[5:20, 5:20] = 1
            mock_ext.return_value = lm

            mock_exp = MagicMock()
            mock_exp.expand.return_value = [cnt]
            MockExp.return_value = mock_exp

            result = ep.process(
                map_data, meta, mock_transformer,
                mock_graph_builder, mock_landmark_builder,
            )

        assert "version" in result
        assert result["uuid"] == "test-uuid"
        assert "data" in result
        mock_detect.assert_called_once()

    def test_with_new_regions(
        self, ep, mock_transformer, mock_graph_builder, mock_landmark_builder
    ):
        """有新区域时应调用 assign_new_regions"""
        map_img = np.full((30, 30), 200, dtype=np.uint8)
        cnt = np.array([[[5, 5]], [[20, 5]], [[20, 20]], [[5, 20]]], dtype=np.int32)
        mock_transformer.rooms_data_to_contours.return_value = [cnt]

        map_data = {
            "map_img": map_img,
            "robot_model": "s10",
            "uuid": "test-uuid",
            "labels_json": {
                "data": [
                    {"name": "A", "id": "ROOM_001", "type": "polygon",
                     "geometry": [0, 0, 1, 0, 1, 1, 0, 1, 0, 0],
                     "colorType": 0, "graph": [], "groundMaterial": None},
                ]
            },
        }
        meta = {"map_data": map_img, "input_data": map_img}

        new_regions = np.zeros((30, 30), dtype=np.int32)
        new_regions[25:29, 25:29] = 1  # 一个新区域

        lm = np.zeros((30, 30), dtype=np.int32)
        lm[5:20, 5:20] = 1

        with patch.object(ep, "detect_new_regions", return_value=new_regions), \
             patch.object(ep, "assign_new_regions", return_value=lm) as mock_assign, \
             patch.object(ep, "extend_pixel", return_value=lm), \
             patch("app.services.extended_partition.ContourExpander") as MockExp:

            mock_exp = MagicMock()
            mock_exp.expand.return_value = [cnt]
            MockExp.return_value = mock_exp

            result = ep.process(
                map_data, meta, mock_transformer,
                mock_graph_builder, mock_landmark_builder,
            )

        mock_assign.assert_called_once()
        assert len(result["data"]) > 0

    def test_k20pro_landmarks(
        self, ep, mock_transformer, mock_graph_builder, mock_landmark_builder
    ):
        """K20PRO 应生成 landmark"""
        map_img = np.full((30, 30), 200, dtype=np.uint8)
        cnt = np.array([[[5, 5]], [[20, 5]], [[20, 20]], [[5, 20]]], dtype=np.int32)
        mock_transformer.rooms_data_to_contours.return_value = [cnt]
        mock_landmark_builder.generate_landmarks.return_value = [
            {"type": "pose", "id": "PLATFORM_LANDMARK_001"}
        ]

        map_data = {
            "map_img": map_img,
            "robot_model": "S-K20PRO",
            "uuid": "k20-uuid",
            "labels_json": {
                "data": [
                    {"name": "A", "id": "ROOM_001", "type": "polygon",
                     "geometry": [0, 0, 1, 0, 1, 1, 0, 1, 0, 0],
                     "colorType": 0, "graph": [], "groundMaterial": None},
                ]
            },
        }
        meta = {"map_data": map_img, "input_data": map_img}

        lm = np.zeros((30, 30), dtype=np.int32)
        lm[5:20, 5:20] = 1

        with patch.object(ep, "detect_new_regions", return_value=np.zeros((30, 30), dtype=np.int32)), \
             patch.object(ep, "extend_pixel", return_value=lm), \
             patch("app.services.extended_partition.ContourExpander") as MockExp:

            mock_exp = MagicMock()
            mock_exp.expand.return_value = [cnt]
            MockExp.return_value = mock_exp

            result = ep.process(
                map_data, meta, mock_transformer,
                mock_graph_builder, mock_landmark_builder,
            )

        mock_landmark_builder.generate_landmarks.assert_called_once()
        has_landmark = any(d.get("type") == "pose" for d in result["data"])
        assert has_landmark
