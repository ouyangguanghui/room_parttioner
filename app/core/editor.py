"""房间编辑器 —— 对接 Lambda handler, 编排完整业务流程

与旧 app.py 的 RoomEditor 类等价, 但内部调用新的模块化服务。
"""

import time
import json
import logging
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
import cv2

from app.core.config import load_config
from app.core.partitioner import RoomPartitioner
from app.utils.s3_loader import S3DataLoader
from app.utils.coordinate import CoordinateTransformer
from app.utils.serializer import LabelSerializer
from app.utils.graph import RoomGraph
from app.utils.contour_expander import ContourExpander

logger = logging.getLogger(__name__)

VERSION = "online_4.0.2"


class RoomEditor:
    """
    房间编辑器: Lambda handler 的业务入口

    支持操作:
    - split:       首次自动分割
    - repartition: 清空重新分割
    - division:    用户画线分割
    - merge:       合并房间
    """

    def __init__(self, bucket: str, key: str):
        self.loader = S3DataLoader(bucket, key)
        self.config = load_config("config/default.yaml")
        self.partitioner = RoomPartitioner(self.config)
        self.graph_builder = RoomGraph(self.config)
        self.expander = ContourExpander(self.config)

        # 运行时数据 (load 后填充)
        self.map_img: Optional[np.ndarray] = None
        self.resolution: float = 0.05
        self.origin: List[float] = [0.0, 0.0]
        self.labels_json: Optional[Dict] = None
        self.robot_model: str = "s10"
        self.uuid: Optional[str] = None
        self.markers_json: Optional[Dict] = None
        self.world_charge_pose: List[float] = [0, 0, 0]

    # ==================== 数据加载 ====================

    def load_data(self) -> bool:
        """从 S3 加载所有数据"""
        try:
            data = self.loader.load()
            self.map_img = data["map_img"]
            self.resolution = data["resolution"]
            self.origin = data["origin"]
            self.labels_json = data["labels_json"]
            self.robot_model = data["robot_model"]
            self.uuid = data["uuid"]
            self.markers_json = data["markers_json"]
            self.world_charge_pose = data["world_charge_pose"]

            # 更新 config
            self.config["resolution"] = self.resolution
            return True
        except Exception as e:
            logger.error(f"加载数据失败: {e}")
            return False

    # ==================== 主入口 ====================

    def room_edit(self, detection: bool, repartition: bool,
                  division_croods_dict: Dict = None,
                  room_merge_list: List = None) -> Tuple[bool, Any]:
        """
        主编辑入口 (与旧 app.py room_edit 签名一致)

        Args:
            detection: 是否执行自动检测 (split/repartition)
            repartition: 是否清空重来
            division_croods_dict: 分割参数 {"id": "ROOM_001", "A": [x,y], "B": [x,y]}
            room_merge_list: 合并列表 ["ROOM_001", "ROOM_002"]

        Returns:
            (success, labels_json_or_error_code)
        """
        if not self.load_data():
            return False, 1

        if self.resolution == 0:
            return False, 2

        if detection:
            return self.room_detect(repartition=repartition)

        if division_croods_dict:
            if not self.labels_json or len(division_croods_dict) != 3:
                return (False, 3) if not self.labels_json else (False, 4)
            return self.room_divide(division_croods_dict)

        if room_merge_list:
            if not self.labels_json or len(room_merge_list) == 1:
                return (False, 3) if not self.labels_json else (False, 4)
            return self.room_merge(room_merge_list)

        return False, 4

    # ==================== 自动检测 ====================

    def room_detect(self, repartition: bool = False) -> Tuple[bool, Any]:
        """自动房间分割"""
        t0 = time.time()
        h, w = self.map_img.shape[:2]
        logger.info(f"robot_model: {self.robot_model}")

        need_detect = not self.labels_json or repartition

        if need_detect:
            if repartition:
                logger.info("重新分区")
            else:
                logger.info("首次分区 (labels 为空)")

            # 初始化 labels
            if self.labels_json is None:
                self.labels_json = {"data": []}
            self.labels_json["data"] = []

            # 自动分区
            gray = cv2.cvtColor(self.map_img, cv2.COLOR_BGR2GRAY)
            self.partitioner.auto_partition(gray, extend=True)
        else:
            logger.info("扩展分区 (已有 labels)")
            # 从已有 labels 恢复轮廓
            transformer = self._make_transformer()
            serializer = LabelSerializer(transformer)
            existing_contours = serializer.deserialize_contours(self.labels_json)

            gray = cv2.cvtColor(self.map_img, cv2.COLOR_BGR2GRAY)
            # 用已有轮廓构建 label_map, 然后做扩展分区
            label_map = np.zeros((h, w), dtype=np.int32)
            for i, cnt in enumerate(existing_contours):
                cv2.drawContours(label_map, [cnt], -1, i + 1, -1)

            self.partitioner._label_map = label_map
            self.partitioner._grid_map = gray
            self.partitioner._extract_contours()
            self.partitioner.extend_partition()

        # 轮廓外扩
        self.partitioner.expand_contours(self.map_img)

        # 序列化输出
        charge_pixel = self._charge_to_pixel()
        labels = self.partitioner.serialize(
            resolution=self.resolution,
            origin=self.origin,
            uuid=self.uuid,
            robot_model=self.robot_model,
            existing_labels=self.labels_json if not need_detect else None,
            marker_polygons=self._get_marker_polygons(),
        )
        # 排序 (k20 从充电桩开始)
        if self.robot_model == "S-K20PRO" and charge_pixel:
            self.partitioner.sort_rooms(charge_pixel)
            labels = self.partitioner.serialize(
                resolution=self.resolution,
                origin=self.origin,
                uuid=self.uuid,
                robot_model=self.robot_model,
                existing_labels=self.labels_json if not need_detect else None,
                marker_polygons=self._get_marker_polygons(),
            )

        self.labels_json = labels
        logger.info(f"分区完成, 耗时 {time.time() - t0:.2f}s")
        return True, self.labels_json

    # ==================== 画线分割 ====================

    def room_divide(self, division_croods_dict: Dict) -> Tuple[bool, Any]:
        """
        用户画线分割房间

        Args:
            division_croods_dict: {"id": "ROOM_001", "A": [wx, wy], "B": [wx, wy]}
        """
        try:
            t0 = time.time()
            h, w = self.map_img.shape[:2]
            transformer = self._make_transformer()

            # 解析参数
            room_id = division_croods_dict['id']
            seg_idx = int(room_id.split('_')[-1]) - 1
            rooms_data = [d for d in self.labels_json['data'] if 'ROOM' in d.get('id', '')]

            if seg_idx >= len(rooms_data):
                return False, 7

            A = division_croods_dict['A']
            B = division_croods_dict['B']

            # 在 geometry 上找交点, 分割多边形
            geometry = rooms_data[seg_idx]['geometry']
            ok, result = self._find_split_points(A, B, geometry)
            if not ok:
                return False, 5

            poly_a, poly_b, intersections = result

            # 构造新 geometry
            geom_a = self._flatten_geometry(poly_a)
            geom_b = self._flatten_geometry(poly_b)

            # 面积检查 (像素)
            cnt_a = transformer.world_to_contour(geom_a)
            cnt_b = transformer.world_to_contour(geom_b)
            area_min = 0.25 / (self.resolution ** 2)
            if cv2.contourArea(cnt_a) < area_min or cv2.contourArea(cnt_b) < area_min:
                return False, 10

            # 连通性检查
            for cnt in [cnt_a, cnt_b]:
                mask = np.zeros((h, w), dtype=np.uint8)
                cv2.drawContours(mask, [cnt], -1, 255, -1)
                kernel = np.ones((5, 5), np.uint8)
                eroded = cv2.erode(mask, kernel, iterations=1)
                n_obj, _, stats, _ = cv2.connectedComponentsWithStats(eroded)
                if n_obj != 2 and all(
                    stats[k, cv2.CC_STAT_AREA] >= 0.10 / (self.resolution ** 2)
                    for k in range(1, n_obj)
                ):
                    return False, 12

            # 更新 labels
            original_name = rooms_data[seg_idx]['name']
            ground = rooms_data[seg_idx].get('groundMaterial')

            rooms_data[seg_idx]['geometry'] = geom_a
            new_id = f"ROOM_{len(rooms_data) + 1:03d}"
            new_name = self._next_room_name(rooms_data)

            rooms_data.append({
                "name": new_name,
                "id": new_id,
                "type": "polygon",
                "geometry": geom_b,
                "colorType": None,
                "graph": None,
                "groundMaterial": ground,
            })

            # 大面积保留原名
            if cv2.contourArea(cnt_a) >= cv2.contourArea(cnt_b):
                rooms_data[seg_idx]['name'] = original_name
                rooms_data[-1]['name'] = new_name
                select_idx = len(rooms_data) - 1
            else:
                rooms_data[seg_idx]['name'] = new_name
                rooms_data[-1]['name'] = original_name
                select_idx = seg_idx

            # 重建图 + 着色
            self.labels_json['data'] = rooms_data
            self._rebuild_graph_and_colors(transformer, h, w, select_idx)

            self.labels_json['version'] = VERSION
            self.labels_json['uuid'] = self.uuid
            logger.info(f"分割完成, 耗时 {time.time() - t0:.2f}s")
            return True, self.labels_json

        except Exception as e:
            logger.error(f"分割失败: {e}", exc_info=True)
            return False, 9

    # ==================== 合并 ====================

    def room_merge(self, room_merge_list: List[str]) -> Tuple[bool, Any]:
        """
        合并房间

        Args:
            room_merge_list: ["ROOM_001", "ROOM_002", ...]
        """
        try:
            t0 = time.time()
            h, w = self.map_img.shape[:2]
            transformer = self._make_transformer()

            # 解析索引
            merge_indices = sorted(
                int(r.split("_")[-1]) - 1 for r in room_merge_list
            )
            rooms_data = [d for d in self.labels_json['data'] if 'ROOM' in d.get('id', '')]
            platform_data = [d for d in self.labels_json['data'] if 'PLATFORM_LANDMARK' in d.get('id', '')]

            # 合并轮廓
            mask = np.zeros((h, w), dtype=np.uint8)
            contours_list = []
            max_area = -1
            max_idx = merge_indices[0]

            for idx in merge_indices:
                cnt = transformer.world_to_contour(rooms_data[idx]['geometry'])
                contours_list.append(cnt)
                cv2.drawContours(mask, [cnt], -1, 255, -1)
                area = cv2.contourArea(cnt)
                if area > max_area:
                    max_area = area
                    max_idx = idx

            # 连通性检查
            n_obj, _ = cv2.connectedComponents(mask)
            if n_obj != 2:
                return False, 8

            # 去除墙壁区域, 提取合并后轮廓
            gray = cv2.cvtColor(self.map_img, cv2.COLOR_BGR2GRAY)
            merge_mask = mask.copy()
            merge_mask[mask != 255] = 127
            merge_mask[(gray == 0) & (mask == 0)] = 127
            merge_mask[(gray == 255) | (mask == 255)] = 255

            _, thresh = cv2.threshold(merge_mask, 235, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            merged_cnt = max(contours, key=cv2.contourArea)

            # 更新 labels
            min_idx = min(merge_indices)
            new_geom = transformer.contour_to_geometry(merged_cnt)
            rooms_data[min_idx]['geometry'] = new_geom

            # 保留最大房间的名称
            if min_idx != max_idx:
                rooms_data[min_idx]['name'] = rooms_data[max_idx].get('name', rooms_data[min_idx]['name'])
                rooms_data[min_idx]['groundMaterial'] = rooms_data[max_idx].get('groundMaterial')

            # 删除被合并的 (从大到小删)
            for idx in sorted(merge_indices, reverse=True):
                if idx != min_idx:
                    del rooms_data[idx]

            # 重编号 + 重建图
            for i, room in enumerate(rooms_data):
                room['id'] = f"ROOM_{i + 1:03d}"

            self.labels_json['data'] = rooms_data
            self._rebuild_graph_and_colors(transformer, h, w, min_idx)

            self.labels_json['version'] = VERSION
            self.labels_json['uuid'] = self.uuid
            logger.info(f"合并完成, 耗时 {time.time() - t0:.2f}s")
            return True, self.labels_json

        except Exception as e:
            logger.error(f"合并失败: {e}", exc_info=True)
            return False, 9

    # ==================== 内部工具 ====================

    def _make_transformer(self) -> CoordinateTransformer:
        h = self.map_img.shape[0]
        return CoordinateTransformer(self.resolution, self.origin, h)

    def _charge_to_pixel(self) -> Optional[Tuple[int, int]]:
        """充电桩世界坐标 → 像素坐标"""
        if self.world_charge_pose == [0, 0, 0]:
            return None
        t = self._make_transformer()
        return t.world_to_pixel(self.world_charge_pose[0],
                                self.world_charge_pose[1])

    def _get_marker_polygons(self) -> Optional[List]:
        """从 markers_json 提取家具标记多边形"""
        if not self.markers_json:
            return None
        polys = []
        for item in self.markers_json.get('data', []):
            if item.get('name') == '家具' and item.get('type') == 'polygon':
                polys.append(item['geometry'])
        return polys if polys else None

    def _rebuild_graph_and_colors(self, transformer, h, w, select_idx):
        """重建邻接图和颜色 (分割/合并后)"""
        rooms_data = [d for d in self.labels_json['data'] if 'ROOM' in d.get('id', '')]
        pixel_contours = [
            transformer.world_to_contour(r['geometry']) for r in rooms_data
        ]

        # 重建图
        graph_builder = RoomGraph(self.config)
        for i, cnt1 in enumerate(pixel_contours):
            neighbors = []
            for j, cnt2 in enumerate(pixel_contours):
                if i != j and graph_builder.check_connectivity(cnt1, cnt2, self.map_img):
                    neighbors.append(j)
            rooms_data[i]['graph'] = neighbors

        # 着色
        graph = {i: rooms_data[i]['graph'] for i in range(len(rooms_data))}
        colors = graph_builder.assign_colors(graph)
        for i in range(len(rooms_data)):
            if rooms_data[i]['colorType'] is None or i == select_idx:
                rooms_data[i]['colorType'] = colors.get(i, 0)

        self.labels_json['data'] = rooms_data

    def _find_split_points(self, A, B, geometry):
        """找分割线与多边形的交点, 拆分多边形"""
        import math

        contour_pts = [(geometry[i], geometry[i + 1])
                       for i in range(0, len(geometry), 2)]

        intersections = []
        intersection_indices = []

        for i in range(len(contour_pts)):
            p1 = contour_pts[i]
            p2 = contour_pts[(i + 1) % len(contour_pts)]
            ip = self._line_intersection(A, B, p1, p2)
            if ip:
                intersections.append(ip)
                intersection_indices.append(i + 1)

        if len(intersections) < 2:
            return False, "交点不足两个"

        if len(intersections) > 2:
            # 选离 A/B 最近的两个
            da = [math.dist(A, p) for p in intersections]
            db = [math.dist(B, p) for p in intersections]
            ia = da.index(min(da))
            ib = db.index(min(db))
            i1, i2 = min(ia, ib), max(ia, ib)
            intersections = [intersections[i1], intersections[i2]]
            intersection_indices = [intersection_indices[i1], intersection_indices[i2]]

        idx1, idx2 = intersection_indices
        poly_a = contour_pts[idx1:idx2] + [intersections[1], intersections[0]]
        poly_b = contour_pts[idx2:] + contour_pts[:idx1] + [intersections[0], intersections[1]]

        return True, (poly_a, poly_b, intersections)

    @staticmethod
    def _line_intersection(A, B, p1, p2):
        """计算线段 AB 与线段 p1p2 的交点"""
        if A == B:
            return None

        ax, ay = A
        bx, by = B
        p1x, p1y = p1
        p2x, p2y = p2

        if ax == bx:  # AB 垂直
            x = ax
            if p1x == p2x:
                return None
            if min(p1x, p2x) <= x <= max(p1x, p2x):
                k = (p2y - p1y) / (p2x - p1x)
                y = k * (x - p1x) + p1y
                return (x, y)
        else:
            k1 = (by - ay) / (bx - ax)
            b1 = ay - k1 * ax

            if p1x == p2x:
                x = p1x
                y = k1 * x + b1
                if min(p1y, p2y) <= y <= max(p1y, p2y):
                    return (x, y)
            else:
                k2 = (p2y - p1y) / (p2x - p1x)
                b2 = p1y - k2 * p1x
                cross_a = k1 * p1x + b1 - p1y
                cross_b = k1 * p2x + b1 - p2y
                if cross_a * cross_b <= 0:
                    if k1 == k2:
                        return None
                    x = (b2 - b1) / (k1 - k2)
                    y = k1 * x + b1
                    return (x, y)
        return None

    @staticmethod
    def _flatten_geometry(poly_pts):
        """多边形点列表 → flat geometry [x0,y0,x1,y1,...,x0,y0]"""
        geom = []
        for pt in poly_pts:
            geom.extend([pt[0], pt[1]])
        geom.extend([poly_pts[0][0], poly_pts[0][1]])
        return geom

    @staticmethod
    def _next_room_name(rooms_data):
        """分配下一个可用房间名 (A~Z)"""
        used = {r.get('name') for r in rooms_data}
        name = chr(ord('A'))
        while name in used:
            name = chr(ord(name) + 1)
        return name
