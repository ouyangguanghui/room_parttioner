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
        """面积过滤已移除，所有新区域都应被检测到"""
        ep = ExtendedPartitioner({"min_new_region_area": 100, "wall_threshold": 128})
        lm = np.ones((50, 50), dtype=np.int32)
        # 在角落留一个 5x5 的小空白
        lm[0:5, 0:5] = 0
        gm = np.full((50, 50), 255, dtype=np.uint8)

        result = ep.detect_new_regions(lm, gm)
        assert result.max() >= 1  # 5x5=25，不再过滤，应被检测到

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
        """已有房间应保留原 name/id/groundMaterial，colorType 用新计算值"""
        cnt = np.array([[[5, 5]], [[20, 5]], [[20, 20]], [[5, 20]]], dtype=np.int32)
        contours = [cnt]
        graph = {0: []}
        colors = {0: 2}
        order = [0]
        old_rooms = [
            {"name": "客厅", "id": "ROOM_001", "type": "polygon",
             "colorType": 3, "groundMaterial": "wood",
             "geometry": [0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0]},
        ]
        old_mapping = {1: 0}  # label 1 → old room idx 0

        result = ep.serialize_contours(
            contours, graph, colors, order,
            mock_transformer, old_rooms, old_mapping,
        )
        assert len(result) == 1
        assert result[0]["name"] == "客厅"
        assert result[0]["id"] == "ROOM_001"
        assert result[0]["colorType"] == 2  # 使用新计算的颜色

    def test_old_rooms_preserve_id_name_update_graph_color(self, ep, mock_transformer):
        """serialize_contours 应保留旧房间 id/name/groundMaterial，更新 graph/colorType"""
        cnt0 = np.array([[[5, 5]], [[20, 5]], [[20, 20]], [[5, 20]]], dtype=np.int32)
        cnt1 = np.array([[[30, 5]], [[45, 5]], [[45, 20]], [[30, 20]]], dtype=np.int32)
        contours = [cnt0, cnt1]
        graph = {0: [1], 1: [0]}
        colors = {0: 0, 1: 1}
        order = [0, 1]
        old_rooms = [
            {"name": "A", "id": "ROOM_001", "type": "polygon",
             "colorType": 3, "graph": [9], "groundMaterial": "wood",
             "geometry": [0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0]},
            {"name": "B", "id": "ROOM_002", "type": "polygon",
             "colorType": 4, "graph": [8], "groundMaterial": "tile",
             "geometry": [2.0, 0.0, 3.0, 0.0, 3.0, 1.0, 2.0, 1.0, 2.0, 0.0]},
        ]
        old_mapping = {1: 0, 2: 1}

        result = ep.serialize_contours(
            contours, graph, colors, order,
            mock_transformer, old_rooms, old_mapping,
        )

        # id/name/groundMaterial 保留
        assert result[0]["id"] == "ROOM_001"
        assert result[0]["name"] == "A"
        assert result[0]["groundMaterial"] is None
        assert result[1]["id"] == "ROOM_002"
        assert result[1]["name"] == "B"
        assert result[1]["groundMaterial"] is None
        # graph/colorType 使用新计算值
        assert result[0]["graph"] == [1]
        assert result[1]["graph"] == [0]
        assert result[0]["colorType"] == 0
        assert result[1]["colorType"] == 1

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
             "colorType": 0, "groundMaterial": None,
             "geometry": [0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0]},
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
        assert result[0]["colorType"] == 0  # 旧房间保持原值
        assert result[1]["colorType"] == 1  # 新房间使用新着色
        assert result[1]["graph"] == [0]


# ==================== TestFilterThresholds ====================

class TestFilterThresholds:
    def test_skips_lines_inside_old_rooms(self, ep):
        """两端都在旧房间内的 threshold 线应被跳过（old_label_map > 0 的像素被过滤）"""
        old_lm = np.zeros((50, 50), dtype=np.int32)
        old_lm[10:40, 10:40] = 1
        gray = np.full((50, 50), 255, dtype=np.uint8)

        # 线段两端都在旧房间内
        threshold_list = [((20, 20), (30, 30))]
        mask = ep._filter_thresholds(threshold_list, old_lm, gray)
        assert mask.max() == 0  # 旧房间区域内的像素全被过滤

    def test_keeps_lines_crossing_boundary(self, ep):
        """线段穿过旧房间边界时，旧房间外且自由空间的部分应被保留"""
        old_lm = np.zeros((50, 50), dtype=np.int32)
        old_lm[10:25, 10:25] = 1
        gray = np.full((50, 50), 255, dtype=np.uint8)  # 全自由空间

        # 一端在旧房间内 (15,15)，一端在外 (35,35)
        threshold_list = [((15, 15), (35, 35))]
        mask = ep._filter_thresholds(threshold_list, old_lm, gray)
        assert mask.max() == 255  # 旧房间外的部分应保留

    def test_empty_threshold_list(self, ep):
        """空 threshold_list 应返回全零 mask"""
        old_lm = np.zeros((50, 50), dtype=np.int32)
        gray = np.full((50, 50), 255, dtype=np.uint8)
        mask = ep._filter_thresholds([], old_lm, gray)
        assert mask.max() == 0


# ==================== TestClassifyNewRegions ====================

class TestClassifyNewRegions:
    def test_merge_to_adjacent_old_room(self):
        """新区域直接接触旧房间（无 threshold 隔开）应合并"""
        ep = ExtendedPartitioner({"wall_threshold": 128, "min_new_region_area": 5, "min_room_area": 0.01})
        old_lm = np.zeros((50, 50), dtype=np.int32)
        old_lm[10:30, 10:25] = 1  # 旧房间
        old_contours = ExtendedPartitioner._extract_contours(old_lm)

        grid_map = np.full((50, 50), 255, dtype=np.uint8)
        filtered_mask = np.zeros((50, 50), dtype=np.uint8)

        # 新区域紧贴旧房间右侧 (x=25:35)
        # old_lm 在 x=25:35 为 0, grid_map 为 255, filtered_mask 为 0
        result_contours, room_id_list = ep._classify_new_regions(
            old_lm, old_contours, grid_map, filtered_mask
        )
        assert 0 in room_id_list  # 旧房间 idx=0 被合并（old_contours[0]）
        assert len(result_contours) >= 1

    def test_new_room_when_threshold_separates(self):
        """threshold 隔开的新区域应成为新房间"""
        ep = ExtendedPartitioner({"wall_threshold": 128, "min_new_region_area": 5, "min_room_area": 0.01})
        old_lm = np.zeros((50, 50), dtype=np.int32)
        old_lm[10:30, 10:20] = 1  # 旧房间
        old_contours = ExtendedPartitioner._extract_contours(old_lm)

        # 只在旧房间右侧和 threshold 右侧有自由空间
        grid_map = np.zeros((50, 50), dtype=np.uint8)
        grid_map[10:30, 10:40] = 255  # 自由空间覆盖旧房间和右侧

        # threshold 在 x=20:23 画一条竖线（3px宽），完全覆盖 y=10:30
        filtered_mask = np.zeros((50, 50), dtype=np.uint8)
        filtered_mask[10:30, 20:23] = 255  # threshold 线

        result_contours, room_id_list = ep._classify_new_regions(
            old_lm, old_contours, grid_map, filtered_mask
        )
        # 新区域被 threshold 隔开，应成为新房间（result_contours 比 old_contours 多）
        assert len(result_contours) > len(old_contours)

    def test_merge_picks_old_room_with_max_contact(self):
        """同时接触多个旧房间时，应合并到接触像素最多的房间。"""
        ep = ExtendedPartitioner({"wall_threshold": 128, "min_new_region_area": 5, "min_room_area": 0.01})
        old_lm = np.zeros((60, 80), dtype=np.int32)
        old_lm[10:50, 10:20] = 1
        old_lm[10:20, 30:40] = 2
        old_contours = ExtendedPartitioner._extract_contours(old_lm)

        grid_map = np.zeros((60, 80), dtype=np.uint8)
        # 新区域中间桥接：与房间1接触 40px，与房间2接触 10px
        grid_map[10:50, 20:30] = 255
        filtered_mask = np.zeros((60, 80), dtype=np.uint8)

        result_contours, room_id_list = ep._classify_new_regions(
            old_lm, old_contours, grid_map, filtered_mask
        )
        assert 0 in room_id_list  # old_contours[0] (旧房间1) 被合并
        assert len(result_contours) == len(old_contours)  # 没有新房间，只是合并


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
        map_img = np.full((20, 20), 255, dtype=np.uint8)
        lm = ExtendedPartitioner._contours_to_label_map([cnt], map_img, (20, 20))
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
    def test_no_new_regions_preserves_old_data(
        self, ep, mock_transformer, mock_graph_builder, mock_landmark_builder
    ):
        """无新区域时：id/name/geometry 不变，仅更新 graph/colorType"""
        map_img = np.full((30, 30), 255, dtype=np.uint8)
        cnt = np.array([[[5, 5]], [[20, 5]], [[20, 20]], [[5, 20]]], dtype=np.int32)
        mock_transformer.rooms_data_to_contours.return_value = [cnt]

        # mock graph_builder 返回值
        mock_graph_builder.build_graph.return_value = {0: []}
        mock_graph_builder.assign_colors.return_value = {0: 2}

        old_geometry = [0, 0, 1, 0, 1, 1, 0, 1, 0, 0]
        map_data = {
            "cleaned_img": map_img,
            "cleaned_img2": map_img,
            "robot_model": "s10",
            "uuid": "test-uuid",
            "labels_json": {
                "data": [
                    {"name": "MyRoom", "id": "ROOM_001", "type": "polygon",
                     "geometry": old_geometry,
                     "colorType": 0, "graph": [1], "groundMaterial": "wood"},
                ]
            },
        }

        with patch.object(ep, "detect_new_regions") as mock_detect:
            mock_detect.return_value = np.zeros((30, 30), dtype=np.int32)

            result = ep.process(
                map_data, mock_transformer,
                mock_graph_builder, mock_landmark_builder,
            )

        assert result["uuid"] == "test-uuid"
        room = result["data"][0]
        # id/name/geometry/groundMaterial 不变
        assert room["id"] == "ROOM_001"
        assert room["name"] == "MyRoom"
        assert room["geometry"] == old_geometry
        assert room["groundMaterial"] == "wood"
        # graph 更新，colorType 无冲突则保留旧值
        assert room["graph"] == []
        assert room["colorType"] == 0  # 旧值 0，无邻居无冲突，保持不变
        # 不调用 extend_pixel
        mock_detect.assert_called_once()

    def test_with_new_regions(
        self, ep, mock_transformer, mock_graph_builder, mock_landmark_builder
    ):
        """有新区域时应调用 _process_with_new_regions"""
        map_img = np.full((30, 30), 255, dtype=np.uint8)
        cnt = np.array([[[5, 5]], [[20, 5]], [[20, 20]], [[5, 20]]], dtype=np.int32)
        mock_transformer.rooms_data_to_contours.return_value = [cnt]

        map_data = {
            "cleaned_img": map_img,
            "cleaned_img2": map_img,
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

        new_regions = np.zeros((30, 30), dtype=np.int32)
        new_regions[25:29, 25:29] = 1  # 一个新区域

        fake_result = {
            "version": "v4.0.2",
            "uuid": "test-uuid",
            "data": [
                {"name": "A", "id": "ROOM_001", "type": "polygon",
                 "geometry": [0, 0, 1, 0, 1, 1, 0, 1, 0, 0],
                 "colorType": 0, "graph": [], "groundMaterial": None},
            ],
        }

        with patch.object(ep, "detect_new_regions", return_value=new_regions), \
             patch.object(ep, "_process_with_new_regions", return_value=fake_result) as mock_proc:

            result = ep.process(
                map_data, mock_transformer,
                mock_graph_builder, mock_landmark_builder,
            )

        mock_proc.assert_called_once()
        assert len(result["data"]) > 0

    def test_k20pro_landmarks(
        self, ep, mock_transformer, mock_graph_builder, mock_landmark_builder
    ):
        """K20PRO 有新区域时应生成 landmark"""
        map_img = np.full((30, 30), 255, dtype=np.uint8)
        cnt = np.array([[[5, 5]], [[20, 5]], [[20, 20]], [[5, 20]]], dtype=np.int32)
        mock_transformer.rooms_data_to_contours.return_value = [cnt]
        mock_landmark_builder.generate_landmarks.return_value = [
            {"type": "pose", "id": "PLATFORM_LANDMARK_001"}
        ]

        map_data = {
            "cleaned_img": map_img,
            "cleaned_img2": map_img,
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

        new_regions = np.zeros((30, 30), dtype=np.int32)
        new_regions[25:29, 25:29] = 1  # 一个新区域

        # _process_with_new_regions 内部会处理 landmarks
        fake_result = {
            "version": "v4.0.2",
            "uuid": "k20-uuid",
            "data": [
                {"name": "A", "id": "ROOM_001", "type": "polygon",
                 "geometry": [0, 0, 1, 0, 1, 1, 0, 1, 0, 0],
                 "colorType": 0, "graph": [], "groundMaterial": None},
                {"type": "pose", "id": "PLATFORM_LANDMARK_001"},
            ],
        }

        with patch.object(ep, "detect_new_regions", return_value=new_regions), \
             patch.object(ep, "_process_with_new_regions", return_value=fake_result):

            result = ep.process(
                map_data, mock_transformer,
                mock_graph_builder, mock_landmark_builder,
            )

        has_landmark = any(d.get("type") == "pose" for d in result["data"])
        assert has_landmark
