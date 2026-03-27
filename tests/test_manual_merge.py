"""ManualMerger 单元测试（pytest 风格）。"""

import copy
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from app.services.manual_merge import ManualMerger
from app.core.config import load_config
from app.core.errors import (
    InvalidParameterError,
    RoomIndexOutOfRangeError,
    RoomsNotConnectedError,
)


# ==================== fixtures ====================

@pytest.fixture
def merger():
    return ManualMerger(load_config())


@pytest.fixture
def two_adjacent_rooms():
    """两个相邻矩形房间 (共享边 x=2)"""
    return [
        {
            "name": "A", "id": "ROOM_001", "type": "polygon",
            "geometry": [0.0, 0.0, 2.0, 0.0, 2.0, 2.0, 0.0, 2.0, 0.0, 0.0],
            "colorType": 0, "graph": [1], "groundMaterial": None,
        },
        {
            "name": "B", "id": "ROOM_002", "type": "polygon",
            "geometry": [2.0, 0.0, 4.0, 0.0, 4.0, 2.0, 2.0, 2.0, 2.0, 0.0],
            "colorType": 1, "graph": [0], "groundMaterial": None,
        },
    ]


@pytest.fixture
def three_rooms():
    """三个房间: A-B 相邻, B-C 相邻, A-C 不相邻"""
    return [
        {
            "name": "A", "id": "ROOM_001", "type": "polygon",
            "geometry": [0.0, 0.0, 2.0, 0.0, 2.0, 2.0, 0.0, 2.0, 0.0, 0.0],
            "colorType": 0, "graph": [1], "groundMaterial": None,
        },
        {
            "name": "B", "id": "ROOM_002", "type": "polygon",
            "geometry": [2.0, 0.0, 4.0, 0.0, 4.0, 2.0, 2.0, 2.0, 2.0, 0.0],
            "colorType": 1, "graph": [0, 2], "groundMaterial": None,
        },
        {
            "name": "C", "id": "ROOM_003", "type": "polygon",
            "geometry": [4.0, 0.0, 6.0, 0.0, 6.0, 2.0, 4.0, 2.0, 4.0, 0.0],
            "colorType": 2, "graph": [1], "groundMaterial": None,
        },
    ]


# ==================== _extract_merge_params 校验 ====================

class TestExtractMergeParams:

    def test_valid_params(self, merger, two_adjacent_rooms):
        indices = merger._extract_merge_params(
            two_adjacent_rooms, ["ROOM_001", "ROOM_002"]
        )
        assert indices == [0, 1]

    def test_not_list_raises(self, merger, two_adjacent_rooms):
        with pytest.raises(InvalidParameterError, match="必须是列表"):
            merger._extract_merge_params(two_adjacent_rooms, "ROOM_001")

    def test_non_string_elements_raises(self, merger, two_adjacent_rooms):
        with pytest.raises(InvalidParameterError, match="字符串"):
            merger._extract_merge_params(two_adjacent_rooms, [1, 2])

    def test_single_room_raises(self, merger, two_adjacent_rooms):
        with pytest.raises(InvalidParameterError, match="不足 2"):
            merger._extract_merge_params(two_adjacent_rooms, ["ROOM_001"])

    def test_nonexistent_room_raises(self, merger, two_adjacent_rooms):
        with pytest.raises(RoomIndexOutOfRangeError):
            merger._extract_merge_params(
                two_adjacent_rooms, ["ROOM_001", "ROOM_999"]
            )


# ==================== 像素级合并 ====================

class TestPixelMerge:

    def test_merge_rooms_pixel_basic(self):
        label_map = np.array([
            [0, 1, 1, 2, 2],
            [0, 1, 1, 2, 2],
            [0, 3, 3, 3, 0],
        ], dtype=np.int32)

        result = ManualMerger.merge_rooms_pixel(label_map, [1, 2])
        assert (result[0:2, 1:3] == 1).all()
        assert (result[0:2, 3:5] == 1).all()  # 原来的 2 变成 1
        assert (result[2, 1:4] == 3).all()    # 3 不变

    def test_merge_rooms_pixel_preserves_background(self):
        label_map = np.array([
            [0, 1, 0, 2, 0],
        ], dtype=np.int32)
        result = ManualMerger.merge_rooms_pixel(label_map, [1, 2])
        assert result[0, 0] == 0
        assert result[0, 2] == 0
        assert result[0, 4] == 0

    def test_merge_rooms_pixel_too_few_ids_raises(self):
        label_map = np.zeros((3, 3), dtype=np.int32)
        with pytest.raises(InvalidParameterError):
            ManualMerger.merge_rooms_pixel(label_map, [1])

    def test_merge_rooms_pixel_three_rooms(self):
        label_map = np.array([
            [1, 2, 3],
        ], dtype=np.int32)
        result = ManualMerger.merge_rooms_pixel(label_map, [1, 2, 3])
        assert (result == np.array([[1, 1, 1]])).all()

    def test_merge_by_point_pixel_basic(self):
        label_map = np.array([
            [1, 1, 2, 2],
            [1, 1, 2, 2],
        ], dtype=np.int32)
        result = ManualMerger.merge_by_point_pixel(label_map, (0, 0), (2, 0))
        assert (result == 1).all()

    def test_merge_by_point_pixel_background_raises(self):
        label_map = np.array([
            [0, 1, 2],
        ], dtype=np.int32)
        with pytest.raises(InvalidParameterError, match="不在任何房间"):
            ManualMerger.merge_by_point_pixel(label_map, (0, 0), (1, 0))

    def test_merge_by_point_pixel_same_room_raises(self):
        label_map = np.array([
            [1, 1, 2],
        ], dtype=np.int32)
        with pytest.raises(InvalidParameterError, match="同一房间"):
            ManualMerger.merge_by_point_pixel(label_map, (0, 0), (1, 0))


# ==================== relabel ====================

class TestRelabel:

    def test_relabel_compacts_ids(self):
        label_map = np.array([
            [0, 3, 3, 7, 7],
            [0, 3, 3, 7, 7],
        ], dtype=np.int32)
        result = ManualMerger.relabel(label_map)
        assert set(result.flat) == {0, 1, 2}
        assert (result[result > 0] > 0).all()

    def test_relabel_preserves_background(self):
        label_map = np.array([[0, 5, 0, 10]], dtype=np.int32)
        result = ManualMerger.relabel(label_map)
        assert result[0, 0] == 0
        assert result[0, 2] == 0

    def test_relabel_already_compact(self):
        label_map = np.array([[0, 1, 2, 3]], dtype=np.int32)
        result = ManualMerger.relabel(label_map)
        assert (result == label_map).all()

    def test_relabel_does_not_modify_input(self):
        label_map = np.array([[0, 5, 10]], dtype=np.int32)
        original = label_map.copy()
        ManualMerger.relabel(label_map)
        assert (label_map == original).all()


# ==================== 世界坐标级合并 ====================

class TestMergeRooms:

    def test_merge_two_adjacent_rooms(self, merger, two_adjacent_rooms):
        rooms = copy.deepcopy(two_adjacent_rooms)
        result, new_idx = merger.merge_rooms(rooms, [0, 1])

        assert len(result) == 1
        assert result[0]["id"] == "ROOM_001"
        assert result[0]["name"] == "A"
        assert result[0]["colorType"] is None  # 待重新着色
        assert result[0]["graph"] is None       # 待重建邻接图
        assert new_idx == 0

        # 合并后面积应为原来两个之和 (≈8.0)
        from shapely.geometry import Polygon
        geom = result[0]["geometry"]
        pts = [(geom[i], geom[i+1]) for i in range(0, len(geom)-1, 2)]
        if pts[0] == pts[-1]:
            pts = pts[:-1]
        area = Polygon(pts).area
        assert abs(area - 8.0) < 0.1

    def test_merge_preserves_unmerged_rooms(self, merger, three_rooms):
        rooms = copy.deepcopy(three_rooms)
        result, new_idx = merger.merge_rooms(rooms, [0, 1])

        assert len(result) == 2
        ids = sorted([r["id"] for r in result])
        assert "ROOM_001" in ids   # 保留的合并房间
        assert "ROOM_003" in ids   # 未参与合并

    def test_merge_three_rooms(self, merger, three_rooms):
        rooms = copy.deepcopy(three_rooms)
        result, new_idx = merger.merge_rooms(rooms, [0, 1, 2])

        assert len(result) == 1
        assert result[0]["id"] == "ROOM_001"

    def test_merge_new_idx_accounts_for_removals(self, merger, three_rooms):
        """合并 B+C (索引 1,2)，保留 B 的元数据，删除 C"""
        rooms = copy.deepcopy(three_rooms)
        result, new_idx = merger.merge_rooms(rooms, [1, 2])

        assert len(result) == 2
        assert result[new_idx]["id"] == "ROOM_002"


# ==================== merge_world ====================

class TestMergeWorld:

    def test_merge_world_calls_connectivity_check(self, merger, two_adjacent_rooms):
        rooms = copy.deepcopy(two_adjacent_rooms)
        transformer = MagicMock()
        map_img = MagicMock()

        with patch.object(merger, "_check_connectivity") as check_mock, \
             patch.object(merger, "merge_rooms", return_value=(rooms[:1], 0)):
            merger.merge_world(rooms, ["ROOM_001", "ROOM_002"],
                               transformer, map_img)
            check_mock.assert_called_once()

    def test_merge_world_returns_rooms_and_index(self, merger, two_adjacent_rooms):
        rooms = copy.deepcopy(two_adjacent_rooms)
        transformer = MagicMock()
        map_img = MagicMock()

        with patch.object(merger, "_check_connectivity"), \
             patch.object(merger, "merge_rooms", return_value=([rooms[0]], 0)):
            result, idx = merger.merge_world(
                rooms, ["ROOM_001", "ROOM_002"], transformer, map_img
            )
            assert len(result) == 1
            assert idx == 0


# ==================== _check_connectivity ====================

class TestCheckConnectivity:

    def test_connected_rooms_pass(self, merger, two_adjacent_rooms):
        graph_builder = MagicMock()
        graph_builder.check_connectivity.return_value = True
        transformer = MagicMock()
        map_img = MagicMock()

        with patch("app.services.manual_merge.RoomGraph", return_value=graph_builder):
            # 不抛异常即为通过
            merger._check_connectivity(
                two_adjacent_rooms, [0, 1], transformer, map_img
            )

    def test_disconnected_rooms_raise(self, merger, three_rooms):
        graph_builder = MagicMock()
        # A-C 不相邻
        graph_builder.check_connectivity.return_value = False
        transformer = MagicMock()
        map_img = MagicMock()

        with patch("app.services.manual_merge.RoomGraph", return_value=graph_builder):
            with pytest.raises(RoomsNotConnectedError):
                merger._check_connectivity(
                    three_rooms, [0, 2], transformer, map_img
                )


# ==================== build_landmarks ====================

class TestBuildLandmarks:

    def test_merged_room_gets_new_landmark(self, merger):
        new_rooms = [
            {"id": "ROOM_001", "name": "A",
             "geometry": [0, 0, 4, 0, 4, 2, 0, 2, 0, 0]},
        ]
        landmarks_data = [
            {"geometry": [1.0, 1.0, 0], "roomId": "ROOM_001"},
            {"geometry": [3.0, 1.0, 0], "roomId": "ROOM_002"},
        ]

        transformer = MagicMock()
        transformer.world_to_pixel.return_value = (10, 10)
        graph_builder = MagicMock()
        graph_builder.find_start_room.return_value = 0
        graph_builder.dfs_sort.return_value = [0]
        landmark_builder = MagicMock()
        landmark_builder._find_center.return_value = (2.0, 1.0)

        out = merger.build_landmarks(
            landmarks_data=landmarks_data,
            new_rooms_data=new_rooms,
            merged_room_idx=0,
            graph={0: []},
            contours_list=[MagicMock()],
            world_charge_pose=[0.0, 0.0, 0.0],
            transformer=transformer,
            graph_builder=graph_builder,
            landmark_builder=landmark_builder,
        )

        assert len(out) == 1
        assert out[0]["geometry"] == [2.0, 1.0, 0]  # 重新计算的中心点
        assert out[0]["roomId"] == "ROOM_001"
        assert out[0]["type"] == "pose"
        landmark_builder._find_center.assert_called_once()

    def test_non_merged_rooms_keep_old_landmarks(self, merger):
        new_rooms = [
            {"id": "ROOM_001", "name": "A",
             "geometry": [0, 0, 4, 0, 4, 2, 0, 2, 0, 0]},
            {"id": "ROOM_003", "name": "C",
             "geometry": [4, 0, 6, 0, 6, 2, 4, 2, 4, 0]},
        ]
        landmarks_data = [
            {"geometry": [2.0, 1.0, 0], "roomId": "ROOM_001"},
            {"geometry": [5.0, 1.0, 0], "roomId": "ROOM_003"},
        ]

        transformer = MagicMock()
        transformer.world_to_pixel.return_value = (10, 10)
        graph_builder = MagicMock()
        graph_builder.find_start_room.return_value = 0
        graph_builder.dfs_sort.return_value = [0, 1]
        landmark_builder = MagicMock()
        landmark_builder._find_center.return_value = (2.0, 1.0)

        out = merger.build_landmarks(
            landmarks_data=landmarks_data,
            new_rooms_data=new_rooms,
            merged_room_idx=0,
            graph={0: [1], 1: [0]},
            contours_list=[MagicMock(), MagicMock()],
            world_charge_pose=[0.0, 0.0, 0.0],
            transformer=transformer,
            graph_builder=graph_builder,
            landmark_builder=landmark_builder,
        )

        assert len(out) == 2
        # 合并房间重新计算
        assert out[0]["geometry"] == [2.0, 1.0, 0]
        # 非合并房间复用旧标记点
        assert out[1]["geometry"] == [5.0, 1.0, 0]


# ==================== process 集成 ====================

class TestProcess:

    def test_process_assembles_output(self, merger):
        rooms = [
            {"name": "A", "id": "ROOM_001", "type": "polygon",
             "geometry": [0, 0, 2, 0, 2, 2, 0, 2, 0, 0],
             "colorType": 0, "graph": [1], "groundMaterial": None},
            {"name": "B", "id": "ROOM_002", "type": "polygon",
             "geometry": [2, 0, 4, 0, 4, 2, 2, 2, 2, 0],
             "colorType": 1, "graph": [0], "groundMaterial": None},
        ]
        landmarks = [{
            "id": "PLATFORM_LANDMARK_001",
            "roomId": "ROOM_001",
            "geometry": [1.0, 1.0, 0],
            "name": "A",
            "type": "pose",
        }]
        map_data = {
            "labels_json": {"data": rooms + landmarks},
            "map_img": MagicMock(),
            "world_charge_pose": [0.0, 0.0, 0.0],
            "uuid": "u-001",
        }
        merge_list = ["ROOM_001", "ROOM_002"]

        fake_merged = [{"id": "ROOM_001", "name": "A",
                        "geometry": [0, 0, 4, 0, 4, 2, 0, 2, 0, 0],
                        "colorType": None, "graph": None}]
        fake_landmarks = [
            {"id": "PLATFORM_LANDMARK_001", "roomId": "ROOM_001",
             "geometry": [2, 1, 0], "name": "A", "type": "pose"},
        ]

        with patch.object(merger, "_check_connectivity"), \
             patch.object(merger, "merge_rooms", return_value=(fake_merged, 0)), \
             patch.object(merger, "build_graph_and_colors",
                          return_value=(fake_merged, [], {0: []})) as graph_mock, \
             patch.object(merger, "build_landmarks",
                          return_value=fake_landmarks) as lm_mock:
            out = merger.process(
                map_data=map_data,
                roomid_merge_list=merge_list,
                transformer=MagicMock(),
                graph_builder=MagicMock(),
                landmark_builder=MagicMock(),
            )

        assert out["version"] == "v4.0.2_0.0.1"
        assert out["uuid"] == "u-001"
        assert len(out["data"]) == 2  # 1 room + 1 landmark
        assert out["data"][0]["id"] == "ROOM_001"
        assert out["data"][1]["id"] == "PLATFORM_LANDMARK_001"
        graph_mock.assert_called_once()
        lm_mock.assert_called_once()
