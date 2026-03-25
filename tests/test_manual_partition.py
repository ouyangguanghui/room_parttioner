"""ManualPartitioner(world) 单元测试（pytest 风格）。"""

import copy
from unittest.mock import MagicMock, patch

import pytest

from app.services.manual_partition import ManualPartitioner


@pytest.fixture
def partitioner():
    return ManualPartitioner({"line_thickness": 2})


def test_split_room_splits_polygon_and_appends_new_room(partitioner):
    rooms = [{
        "name": "A",
        "id": "ROOM_001",
        "type": "polygon",
        "geometry": [0.0, 0.0, 4.0, 0.0, 4.0, 4.0, 0.0, 4.0, 0.0, 0.0],
        "colorType": 0,
        "graph": [],
        "groundMaterial": None,
    }]

    out = partitioner.split_room(copy.deepcopy(rooms), 0, [-1.0, 2.0], [5.0, 2.0])

    assert len(out) == 2
    ids = sorted([r["id"] for r in out])
    assert ids == ["ROOM_001", "ROOM_002"]
    assert all(len(r["geometry"]) >= 8 for r in out)


def test_build_landmarks_rebuilds_split_rooms(partitioner):
    rooms_data = [
        {"id": "ROOM_001", "name": "A", "geometry": [0, 0, 2, 0, 2, 2, 0, 2, 0, 0]},
        {"id": "ROOM_002", "name": "B", "geometry": [2, 0, 4, 0, 4, 2, 2, 2, 2, 0]},
        {"id": "ROOM_003", "name": "C", "geometry": [2, 2, 4, 2, 4, 4, 2, 4, 2, 2]},
    ]
    new_rooms_data = copy.deepcopy(rooms_data)
    landmarks_data = [
        {"geometry": [1.0, 1.0, 0], "roomId": "ROOM_001"},
        {"geometry": [3.0, 1.0, 0], "roomId": "ROOM_002"},
    ]

    transformer = MagicMock()
    transformer.world_to_pixel.return_value = (10, 10)
    graph_builder = MagicMock()
    graph_builder.find_start_room.return_value = 0
    graph_builder.dfs_sort.return_value = [0, 1, 2]
    landmark_builder = MagicMock()
    landmark_builder._find_center.side_effect = [(2.5, 1.5), (3.0, 3.0)]

    out = partitioner.build_landmarks(
        landmarks_data=landmarks_data,
        rooms_data=rooms_data,
        new_rooms_data=new_rooms_data,
        target_room_idx=1,
        graph={0: [1], 1: [0, 2], 2: [1]},
        contours_list=[MagicMock(), MagicMock(), MagicMock()],
        world_charge_pose=[0.0, 0.0, 0.0],
        transformer=transformer,
        graph_builder=graph_builder,
        landmark_builder=landmark_builder,
    )

    assert len(out) == 3
    assert out[0]["geometry"] == [1.0, 1.0, 0]  # 非拆分房间复用老平台点
    assert out[1]["geometry"] == [2.5, 1.5, 0]  # 被拆分房间重新生成
    assert out[2]["geometry"] == [3.0, 3.0, 0]  # 新增房间生成
    assert out[0]["type"] == "pose"
    assert out[1]["type"] == "pose"
    assert out[2]["type"] == "pose"
    assert landmark_builder._find_center.call_count == 2


def test_process_assembles_output_labels(partitioner):
    rooms = [{
        "name": "A",
        "id": "ROOM_001",
        "type": "polygon",
        "geometry": [0.0, 0.0, 4.0, 0.0, 4.0, 4.0, 0.0, 4.0, 0.0, 0.0],
        "colorType": 0,
        "graph": [],
        "groundMaterial": None,
    }]
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
    division = {"id": "ROOM_001", "A": [-1.0, 2.0], "B": [5.0, 2.0]}
    fake_rooms = [
        {"id": "ROOM_001", "name": "A"},
        {"id": "ROOM_002", "name": "B"},
    ]
    fake_landmarks = [
        {"id": "PLATFORM_LANDMARK_001", "roomId": "ROOM_001", "name": "A", "geometry": [1, 1, 0], "type": "pose"},
        {"id": "PLATFORM_LANDMARK_002", "roomId": "ROOM_002", "name": "B", "geometry": [2, 2, 0], "type": "pose"},
    ]

    with patch.object(partitioner, "split_room", return_value=fake_rooms) as split_mock, \
         patch.object(partitioner, "build_graph_and_colors", return_value=(fake_rooms, [], {0: [1], 1: [0]})) as graph_mock, \
         patch.object(partitioner, "build_landmarks", return_value=fake_landmarks) as landmarks_mock:
        out = partitioner.process(
            map_data=map_data,
            division_croods_dict=division,
            transformer=MagicMock(),
            graph_builder=MagicMock(),
            landmark_builder=MagicMock(),
        )

    assert out["version"] == "online_4.0.2"
    assert out["uuid"] == "u-001"
    assert len(out["data"]) == 4
    assert [d["id"] for d in out["data"]] == [
        "ROOM_001",
        "ROOM_002",
        "PLATFORM_LANDMARK_001",
        "PLATFORM_LANDMARK_002",
    ]
    split_mock.assert_called_once()
    graph_mock.assert_called_once()
    landmarks_mock.assert_called_once()
