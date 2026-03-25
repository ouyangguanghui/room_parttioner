"""Postprocessor 单元测试（pytest 风格）。"""

import numpy as np

from app.pipeline.postprocessor import Postprocessor


def _build_test_map(h=40, w=40):
    m = np.zeros((h, w), dtype=np.uint8)
    m[2:-2, 2:-2] = 255
    return m


def test_obb_to_line_recovers_endpoints():
    # 原线段近似: (8,10) -> (22,10), 宽度约 4
    obb = [[8, 12], [22, 12], [22, 8], [8, 8]]
    result = Postprocessor._obb_to_line(obb)
    assert result is not None
    p1, p2, direction = result
    pts = {(round(p1[0], 1), round(p1[1], 1)), (round(p2[0], 1), round(p2[1], 1))}
    assert pts == {(8.0, 10.0), (22.0, 10.0)}
    assert np.isclose(np.linalg.norm(direction), 1.0)


def test_split_by_threshold_splits_free_space():
    m = _build_test_map()
    threshold = np.zeros_like(m, dtype=np.uint8)
    threshold[2:-2, 20] = 255  # 竖线切割
    room_map = Postprocessor._split_by_threshold(threshold, m)
    room_ids = np.unique(room_map[(m >= 200) & (threshold == 0)])
    room_ids = room_ids[room_ids > 0]
    assert len(room_ids) >= 2


def test_split_by_area_separates_normal_and_small():
    pp = Postprocessor({"resolution": 1.0, "min_room_area": 5.0})
    label_map = np.zeros((12, 12), dtype=np.int32)
    label_map[1:10, 1:10] = 1
    label_map[10:12, 10:12] = 2  # 小区域
    normal_map, small_map = pp._split_by_area(label_map, np.zeros_like(label_map, dtype=np.uint8))
    assert 1 in np.unique(normal_map)
    assert 2 in np.unique(small_map)


def test_merge_fragments_merges_into_normal_room():
    pp = Postprocessor({})
    normal = np.zeros((20, 20), dtype=np.int32)
    small = np.zeros((20, 20), dtype=np.int32)
    normal[4:14, 4:14] = 3
    small[14:16, 8:10] = 7  # 与 normal 接壤
    merged = pp._merge_fragments(normal, small)
    assert np.all(merged[14:16, 8:10] == 3)


def test_process_returns_room_map_and_threshold_info():
    pp = Postprocessor({"min_room_area": 1.0, "resolution": 0.1, "max_extend": 30, "thickness": 1})
    m = _build_test_map(48, 48)
    raw_output = [[[10, 26], [38, 26], [38, 22], [10, 22]]]
    out = pp.process(raw_output, {"map_data": m})
    assert set(out.keys()) == {"room_map", "threshold_list", "thickness_size"}
    assert out["room_map"].shape == m.shape
    assert out["thickness_size"] == 1
    assert len(out["threshold_list"]) == 1
