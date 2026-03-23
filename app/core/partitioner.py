"""
房间划分核心编排模块

对外只有 4 个操作:
    1. auto_partition()    — 自动分区
    2. extend_partition()  — 扩展分区
    3. split_by_line() / split_by_polyline() / assign_polygon() — 手动分割
    4. merge_rooms() / merge_by_point() — 手动合并

邻接图、着色、坐标转换、美化框、标记点等都是上述操作的内部步骤,
不单独对外暴露。
"""

from typing import Dict, Any, List, Tuple, Optional

import numpy as np
import cv2

from app.services.auto_partition import AutoPartitioner
from app.services.extended_partition import ExtendedPartitioner
from app.services.manual_partition import ManualPartitioner
from app.services.manual_merge import ManualMerger


class RoomPartitioner:
    """
    房间划分器 — 统一入口

    每个操作内部自动完成:
        分区/修改 → 提取轮廓 → 重建邻接图 → 着色 → relabel
    """

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.auto = AutoPartitioner(self.config)
        self.extended = ExtendedPartitioner(self.config)
        self.manual = ManualPartitioner(self.config)
        self.merger = ManualMerger(self.config)

        self._label_map: Optional[np.ndarray] = None
        self._grid_map: Optional[np.ndarray] = None
        self._contours: List[np.ndarray] = []

    # ==================== 服务 1: 自动分区 ====================

    def auto_partition(self, grid_map: np.ndarray,
                       extend: bool = True) -> np.ndarray:
        """
        完整自动分区

        内部流程: 推理/连通域 → 后处理 → [扩展分区] → relabel → 提取轮廓
        """
        self._grid_map = grid_map
        gray = self._to_gray(grid_map)

        self._label_map = self.auto.run(gray)

        if extend:
            self._label_map = self.extended.extend(self._label_map, gray)

        self._finalize()
        return self._label_map

    # ==================== 服务 2: 扩展分区 ====================

    def extend_partition(self) -> np.ndarray:
        """
        扩展分区: 门口检测 + 区域生长

        内部流程: 门口拆分 → 区域生长 → relabel → 提取轮廓
        """
        gray = self._to_gray(self._grid_map)
        self._label_map = self.extended.extend(self._label_map, gray)
        self._finalize()
        return self._label_map

    # ==================== 服务 3: 手动分割 ====================

    def split_by_line(self, pt1: Tuple[int, int],
                      pt2: Tuple[int, int]) -> np.ndarray:
        """画线分割 → relabel → 提取轮廓"""
        self._label_map = self.manual.split_by_line(self._label_map, pt1, pt2)
        self._finalize()
        return self._label_map

    def split_by_polyline(self, points: List[Tuple[int, int]]) -> np.ndarray:
        """折线分割 → relabel → 提取轮廓"""
        self._label_map = self.manual.split_by_polyline(self._label_map, points)
        self._finalize()
        return self._label_map

    def assign_polygon(self, polygon: List[Tuple[int, int]],
                       room_id: int = -1) -> np.ndarray:
        """多边形划定 → 提取轮廓"""
        self._label_map = self.manual.assign_polygon(
            self._label_map, polygon, room_id
        )
        self._finalize()
        return self._label_map

    # ==================== 服务 4: 手动合并 ====================

    def merge_rooms(self, room_ids: List[int]) -> np.ndarray:
        """合并指定房间 → relabel → 提取轮廓"""
        self._label_map = self.merger.merge_rooms(self._label_map, room_ids)
        self._finalize()
        return self._label_map

    def merge_by_point(self, pt1: Tuple[int, int],
                       pt2: Tuple[int, int]) -> np.ndarray:
        """点选合并 → relabel → 提取轮廓"""
        self._label_map = self.merger.merge_by_point(self._label_map, pt1, pt2)
        self._finalize()
        return self._label_map

    # ==================== 查询 ====================

    def get_room_info(self, resolution: float = 0.05) -> List[Dict[str, Any]]:
        """获取当前所有房间信息"""
        return AutoPartitioner.get_room_info(self._label_map, resolution)

    @property
    def label_map(self) -> Optional[np.ndarray]:
        return self._label_map

    @property
    def contours(self) -> List[np.ndarray]:
        return self._contours

    # ==================== 内部方法 ====================

    def _finalize(self):
        """每次操作后的统一收尾: relabel + 提取轮廓"""
        self._label_map = ManualMerger.relabel(self._label_map)
        self._extract_contours()

    def _extract_contours(self):
        """从 label_map 提取各房间轮廓"""
        self._contours = []
        if self._label_map is None:
            return
        for lid in range(1, self._label_map.max() + 1):
            mask = (self._label_map == lid).astype(np.uint8)
            if mask.sum() == 0:
                continue
            cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
            if cnts:
                self._contours.append(max(cnts, key=cv2.contourArea))

    @staticmethod
    def _to_gray(img: np.ndarray) -> np.ndarray:
        if img is None:
            raise ValueError("未加载地图")
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
