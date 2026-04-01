"""ContourBeautifier 单测。"""

import numpy as np
import cv2
import pytest

from app.utils.beautifier import ContourBeautifier


@pytest.fixture
def beautifier():
    return ContourBeautifier()


class TestRemoveCollinear:
    def test_removes_middle_point(self):
        """共线中间点应被去除"""
        pts = [[0, 0], [5, 0], [10, 0]]
        result = ContourBeautifier._remove_collinear(pts)
        assert result == [[0, 0], [10, 0]]

    def test_keeps_corner(self):
        """拐角点不应被去除"""
        pts = [[0, 0], [10, 0], [10, 10]]
        result = ContourBeautifier._remove_collinear(pts)
        assert len(result) == 3

    def test_less_than_3_points(self):
        assert ContourBeautifier._remove_collinear([[0, 0]]) == [[0, 0]]
        assert ContourBeautifier._remove_collinear([]) == []

    def test_deduplicates(self):
        """相邻重复点应被去重"""
        pts = [[0, 0], [0, 0], [10, 0]]
        result = ContourBeautifier._remove_collinear(pts)
        assert [0, 0] not in result[1:] or len(result) == 2

    def test_rectangle_untouched(self):
        """矩形四角不应被去除"""
        pts = [[0, 0], [10, 0], [10, 10], [0, 10]]
        result = ContourBeautifier._remove_collinear(pts)
        assert len(result) == 4


class TestExtractSegments:
    def test_basic_line(self, beautifier):
        pts = np.array([[0, 0], [10, 0], [10, 10]])
        segments = beautifier._extract_segments(pts)
        assert len(segments) >= 1

    def test_single_point(self, beautifier):
        pts = np.array([[5, 5]])
        segments = beautifier._extract_segments(pts)
        assert segments == []


class TestMergeNearbyThresholds:
    def test_merge_horizontal(self, beautifier):
        """水平共线且接近的线段应合并"""
        segs = [[[0, 5], [10, 5]], [[12, 5], [20, 5]]]
        result = beautifier._merge_nearby_thresholds(segs, gap=3)
        assert len(result) == 1
        assert result[0][0][1] == 5 and result[0][1][1] == 5

    def test_merge_vertical(self, beautifier):
        """垂直共线且接近的线段应合并"""
        segs = [[[5, 0], [5, 10]], [[5, 12], [5, 20]]]
        result = beautifier._merge_nearby_thresholds(segs, gap=3)
        assert len(result) == 1

    def test_no_merge_far_apart(self, beautifier):
        """间距过大的线段不应合并"""
        segs = [[[0, 5], [10, 5]], [[20, 5], [30, 5]]]
        result = beautifier._merge_nearby_thresholds(segs, gap=3)
        assert len(result) == 2

    def test_no_merge_different_axis(self, beautifier):
        """不共线的线段不应合并"""
        segs = [[[0, 5], [10, 5]], [[5, 0], [5, 10]]]
        result = beautifier._merge_nearby_thresholds(segs, gap=3)
        assert len(result) == 2

    def test_single_segment(self, beautifier):
        segs = [[[0, 0], [10, 0]]]
        assert beautifier._merge_nearby_thresholds(segs) == segs

    def test_empty(self, beautifier):
        assert beautifier._merge_nearby_thresholds([]) == []


class TestTryMergeSegments:
    def test_horizontal_overlap(self, beautifier):
        result = beautifier._try_merge_segments(
            [[0, 5], [10, 5]], [[8, 5], [20, 5]], gap=3
        )
        assert result is not None
        assert result[0][1] == 5

    def test_not_collinear(self, beautifier):
        result = beautifier._try_merge_segments(
            [[0, 5], [10, 5]], [[0, 10], [10, 10]], gap=3
        )
        assert result is None


class TestBeautifySmoke:
    def test_single_rectangle(self, beautifier):
        """矩形轮廓美化后不应崩溃"""
        h, w = 100, 100
        map_img = np.full((h, w), 127, dtype=np.uint8)
        # 画一个白色矩形（自由空间）
        map_img[20:80, 20:80] = 255

        cnt = np.array([[[20, 20]], [[80, 20]], [[80, 80]], [[20, 80]]], dtype=np.int32)
        all_bbox, all_threshold = beautifier.beautify([cnt], map_img)
        assert len(all_bbox) == 1
        assert isinstance(all_threshold, list)

    def test_empty_contours(self, beautifier):
        map_img = np.full((50, 50), 127, dtype=np.uint8)
        all_bbox, all_threshold = beautifier.beautify([], map_img)
        assert all_bbox == []
        assert all_threshold == []
