"""自动分区模块

内置处理器:
  1. 推理器 (Inferencer): 调用 Triton 模型或使用连通域 fallback
  2. 后处理器 (Postprocessor): OBB 解码 → threshold 切割 → 面积过滤 → 碎片合并

预处理在外部完成，传入 meta dict:
  meta = {
      "map_data":   去噪但未补墙的地图 (H, W) uint8
      "input_data": 补墙平滑后的地图 (H, W) uint8
  }

process() 入口签名与 ManualPartitioner / ManualMerger 保持一致:
  process(map_data, transformer, graph_builder, landmark_builder) -> labels_json
"""

import logging
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
import cv2

from app.pipeline.inferencer import Inferencer
from app.pipeline.postprocessor import Postprocessor
from app.services.extended_partition import ExtendedPartitioner
from app.utils.coordinate import CoordinateTransformer
from app.utils.graph import RoomGraph
from app.utils.landmark import LandmarkManager
from app.utils.contour_expander import ContourExpander
from app.utils.beautifier import ContourBeautifier

logger = logging.getLogger(__name__)



class AutoPartitioner:
    """
    自动分区：推理 → 后处理 → 扩展分区 → 轮廓提取 → 图着色 → 序列化 

    当 Triton 不可用时，fallback 到传统连通域方法。
    """

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}

        # 内置处理器
        self.inferencer: Optional[Inferencer] = None
        if self.config.get("triton_url"):
            self.inferencer = Inferencer(self.config)
        self.postprocessor = Postprocessor(self.config)
        self.extended = ExtendedPartitioner(self.config)

        # 张量准备参数
        self.target_size: List[int] = self.config.get("target_size", [512, 512])
        self.min_input_size: int = self.config.get("min_input_size", 416)
        self.normalize: bool = self.config.get("normalize", True)
        self.mean = self.config.get("mean", [0.485, 0.456, 0.406])
        self.std = self.config.get("std", [0.229, 0.224, 0.225])

        self.beautifier_status = False

    def set_beautifier_status(self, status: bool):
        self.beautifier_status = status

    # ==================== 核心分区 ====================

    def partition(self, meta: Dict[str, Any],
                  extend: bool = True) -> Dict[str, Any]:
        """
        核心分区: 推理 → 后处理 → [扩展] → label_map + 轮廓

        Args:
            meta: 预处理结果 {"map_data": ..., "input_data": ...}
            extend: 是否执行扩展分区 (门口检测 + 区域生长)

        Returns:
            {
                "label_map": (H, W) int32,
                "contours": [np.ndarray, ...],
                "threshold_list": [...],
                "thickness_size": int,
            }
        """
        map_data = meta["map_data"]

        if self.inferencer and self.inferencer.is_ready():
            label_map, threshold_list, thickness_size = self._infer_partition(map_data, meta)
        else:
            label_map = self._fallback_partition(meta)
            threshold_list = []
            thickness_size = self.config.get("thickness", 2)

        if extend:
            label_map = self.extended.extend_pixel(label_map, map_data)

        # relabel 使 ID 连续
        label_map = self._relabel(label_map)

        # 提取轮廓
        contours = self._extract_contours(label_map)

        return {
            "label_map": label_map,
            "contours": contours,
            "threshold_list": threshold_list,
            "thickness_size": thickness_size,
        }

    def _infer_partition(self, map_data: np.ndarray,
                         meta: Dict[str, Any]
                         ) -> Tuple[np.ndarray, List, int]:
        """模型推理分区路径"""
        tensor = self._prepare_tensor(map_data, meta)
        raw_output = self.inferencer.run(tensor)
        result = self.postprocessor.process(raw_output, meta)
        return result["room_map"], result["threshold_list"], result["thickness_size"]

    def _fallback_partition(self, meta: Dict[str, Any]) -> np.ndarray:
        """传统连通域分割 (Triton 不可用时的兜底方案)"""
        map_data = meta["map_data"]
        free = (map_data >= 200).astype(np.uint8)
        if not free.any():
            free = (map_data > 0).astype(np.uint8)

        _, labels = cv2.connectedComponents(free, connectivity=8)
        labels = labels.astype(np.int32)

        normal_map, small_map = self.postprocessor._split_by_area(labels, map_data)
        labels = self.postprocessor._merge_fragments(normal_map, small_map)
        return labels

    # ==================== 张量准备 ====================

    def _prepare_tensor(self, map_data: np.ndarray,
                        meta: Dict[str, Any]) -> np.ndarray:
        """
        将预处理后的灰度地图转换为 Triton 模型输入张量

        流程:
            1. 灰度 → BGR (3 通道复制)
            2. Letterbox resize 到 target_size, 保持长宽比, 填充 114 灰色
            3. uint8 → float32 归一化 [0, 1], 可选 ImageNet 均值/方差标准化
            4. HWC → NCHW

        同时将 scale / pad 写入 meta, 供 postprocessor 做坐标逆映射。

        Args:
            map_data: 前处理后的灰度地图 (H, W) uint8
            meta: 会追加 tensor_scale / tensor_pad / tensor_size

        Returns:
            tensor: (1, 3, target_h, target_w) float32
        """
        th, tw = self.target_size[0], self.target_size[1]

        # 灰度 → BGR
        if map_data.ndim == 2:
            img_bgr = cv2.cvtColor(map_data, cv2.COLOR_GRAY2BGR)
        else:
            img_bgr = map_data.copy()

        h, w = img_bgr.shape[:2]

        min_sz = self.min_input_size
        pre_pad = (0, 0)
        if h < min_sz and w < min_sz:
            # 小图先居中 pad 到 min_input_size
            pad_top_pre = (min_sz - h) // 2
            pad_left_pre = (min_sz - w) // 2
            padded = np.full((min_sz, min_sz, 3), 127, dtype=img_bgr.dtype)
            padded[pad_top_pre:pad_top_pre + h, pad_left_pre:pad_left_pre + w] = img_bgr
            img_bgr = padded
            pre_pad = (pad_top_pre, pad_left_pre)

        h, w = img_bgr.shape[:2]

        # Letterbox: 等比缩放, 短边 pad 至 target
        scale = min(tw / w, th / h)
        nw, nh = int(round(w * scale)), int(round(h * scale))
        resized = cv2.resize(img_bgr, (nw, nh), interpolation=cv2.INTER_NEAREST)

        pad_top = (th - nh) // 2
        pad_left = (tw - nw) // 2
        canvas = np.full((th, tw, 3), 127, dtype=np.uint8)
        canvas[pad_top:pad_top + nh, pad_left:pad_left + nw] = resized

        # 归一化
        tensor = canvas.astype(np.float32) / 255.0
        if self.normalize:
            mean = np.array(self.mean, dtype=np.float32)
            std = np.array(self.std, dtype=np.float32)
            tensor = (tensor - mean) / std

        # HWC → NCHW
        tensor = np.ascontiguousarray(
            tensor.transpose(2, 0, 1)[np.newaxis, ...], dtype=np.float32
        )

        # 写入 meta, 供 postprocessor 逆映射
        meta["tensor_scale"] = scale
        meta["tensor_pad"] = (pad_top, pad_left)
        meta["tensor_size"] = (th, tw)
        meta["pre_pad"] = pre_pad

        return tensor

    # ==================== 轮廓扩展 ====================

    def expand_contours(self, contours: List[np.ndarray],
                        map_img: np.ndarray) -> List[np.ndarray]:
        """将房间轮廓向外扩展一圈, 覆盖边界空闲像素"""
        expander = ContourExpander(self.config)
        return expander.expand(contours, map_img)

    # ==================== 图着色 ====================

    def build_graph_and_colors(
        self,
        contours: List[np.ndarray],
        map_img: np.ndarray,
        graph_builder: RoomGraph,
    ) -> Tuple[Dict[int, List[int]], Dict[int, int]]:
        """
        构建邻接图 + 全量着色

        Returns:
            (graph, colors)
        """
        graph = graph_builder.build_graph(contours, map_img)
        colors = graph_builder.assign_colors(graph)
        return graph, colors

    # ==================== 房间排序 ====================

    def sort_contours(
        self,
        contours: List[np.ndarray],
        graph: Dict[int, List[int]],
        graph_builder: RoomGraph,
        charge_pixel: Tuple[int, int] = None,
        max_area_start: bool = False,
    ) -> Tuple[List[np.ndarray], List[int]]:
        """
        按 DFS 顺序排列轮廓

        Returns:
            (排序后的 contours, 排序顺序 order)
        """
        if charge_pixel:
            start = graph_builder.find_start_room(contours, charge_pixel,
                                                   max_area_start=max_area_start)
        else:
            start = graph_builder.find_start_room(contours, (0, 0),
                                                   max_area_start=True)

        order = graph_builder.dfs_sort(graph, start)

        # 兜底: 确保每个房间都在 order 中
        if not order:
            order = list(range(len(contours)))
        else:
            missing = [i for i in range(len(contours)) if i not in order]
            order.extend(missing)

        sorted_contours = [contours[i] for i in order]
        return sorted_contours, order

    # ==================== 序列化 ====================

    def serialize_contours(
        self,
        contours: List[np.ndarray],
        graph: Dict[int, List[int]],
        colors: Dict[int, int],
        order: List[int],
        transformer: CoordinateTransformer,
    ) -> List[Dict[str, Any]]:
        """
        将轮廓序列化为 labels_json 的 data 列表 (ROOM 部分)

        Args:
            contours: 排序后的轮廓列表
            graph: 邻接图 (排序前的索引)
            colors: 颜色映射 (排序前的索引)
            order: 排序映射 (new_idx → old_idx)
            transformer: 坐标变换器

        Returns:
            [{"name": "A", "id": "ROOM_001", "type": "polygon",
              "geometry": [...], "colorType": 0, "graph": [...], ...}, ...]
        """
        rooms_data = []
        # 构建 old_idx → new_idx 映射, 用于重建 graph 索引
        old_to_new = {old: new for new, old in enumerate(order)}

        for new_idx, cnt in enumerate(contours):
            old_idx = order[new_idx]

            # 坐标转换
            geometry = transformer.contour_to_geometry(cnt, clockwise=True)

            # 重映射 graph 邻居索引
            old_neighbors = graph.get(old_idx, [])
            new_neighbors = sorted(
                old_to_new[nb] for nb in old_neighbors if nb in old_to_new
            )

            # 房间名: A, B, C, ...
            name_idx = new_idx % 26
            name_suffix = new_idx // 26
            name = chr(ord("A") + name_idx) + (str(name_suffix) if name_suffix else "")

            rooms_data.append({
                "name": name,
                "id": f"ROOM_{new_idx + 1:03d}",
                "type": "polygon",
                "geometry": geometry,
                "colorType": colors.get(old_idx, 0),
                "graph": new_neighbors,
                "groundMaterial": None,
            })

        return rooms_data

    # ==================== 标记点 ====================

    def build_landmarks(
        self,
        rooms_data: List[Dict[str, Any]],
        landmark_builder: LandmarkManager,
        marker_polygons: List[List[float]] = None,
    ) -> List[Dict[str, Any]]:
        """
        为所有房间生成平台标记点

        Args:
            rooms_data: 序列化后的房间数据
            landmark_builder: 标记点管理器
            marker_polygons: 家具标记多边形 (世界坐标)

        Returns:
            [{"geometry": ..., "id": ..., "roomId": ..., "name": ..., "type": "pose"}, ...]
        """
        rooms_geometry = [d["geometry"] for d in rooms_data]
        room_names = [d["name"] for d in rooms_data]
        room_ids = [d["id"] for d in rooms_data]

        return landmark_builder.generate_landmarks(
            rooms_geometry, room_names, room_ids,
            marker_polygons=marker_polygons,
        )

    # ==================== 美化框 ====================

    def beautify_contours(
        self,
        contours: List[np.ndarray],
        map_img: np.ndarray,
    ) -> Tuple[Optional[List], Optional[List]]:
        """
        s10 美化框 + 门槛线

        Returns:
            (bbox_list, threshold_list) 或 (None, None)
        """
        if not self.beautifier_status:
            return None, None
        beautifier = ContourBeautifier(self.config)
        return beautifier.beautify(contours, map_img)

    # ==================== 完整流程入口 ====================

    def process(
        self,
        map_data: Dict[str, Any],
        meta: Dict[str, Any],
        transformer: CoordinateTransformer,
        graph_builder: RoomGraph,
        landmark_builder: LandmarkManager,
        extend: bool = True,
        repartition: bool = False,
    ) -> Dict[str, Any]:
        """
        完整自动分区流程 (与 ManualPartitioner.process / ManualMerger.process 对齐)

        Args:
            map_data: 地图数据
                "map_img": 原始地图图像 (H, W) 或 (H, W, 3)
                "resolution": 地图分辨率 float
                "origin": 地图原点 [x, y]
                "labels_json": 已有标注 (repartition=False 且有值时做扩展分区)
                "robot_model": 机器人型号 str
                "uuid": UUID str
                "markers_json": 标记信息 json
                "world_charge_pose": 充电桩世界坐标 [x, y, z]
            meta: 预处理结果 {"map_data": ..., "input_data": ...}
            transformer: 坐标变换器
            graph_builder: 邻接图构建器
            landmark_builder: 标记点管理器
            extend: 是否扩展分区
            repartition: 是否重新分区 (True=清空已有 labels 重来)

        Returns:
            Dict[str, Any]: labels_json 格式
        """
        map_img = map_data["map_img"]
        robot_model = map_data.get("robot_model", "s10")
        labels_json = map_data.get("labels_json")

        need_detect = not labels_json or repartition or not labels_json.get("data")

        # step1: 分区
        if need_detect:
            logger.info("重新分区" if repartition else "首次分区")
            partition_result = self.partition(meta, extend=extend)
            contours = partition_result["contours"]
        else:
            logger.info("扩展分区 (已有 labels)")
            contours = self._restore_contours(labels_json, transformer, map_img)
            if extend:
                # 从已有轮廓构建 label_map → 扩展 → 重提取
                label_map = self._contours_to_label_map(contours, map_img.shape[:2])
                label_map = self.extended.extend_pixel(label_map, meta["map_data"])
                label_map = self._relabel(label_map)
                contours = self._extract_contours(label_map)

        # step2: 轮廓外扩
        contours = self.expand_contours(contours, map_img)

        # step3: 邻接图 + 着色
        gray = self._to_gray(map_img)
        graph, colors = self.build_graph_and_colors(contours, gray, graph_builder)

        # step4: 排序
        charge_pixel = self._get_charge_pixel(map_data, transformer)
        contours, order = self.sort_contours(
            contours, graph, graph_builder,
            charge_pixel=charge_pixel,
            max_area_start=(robot_model == "S-K20PRO"),
        )

        # step5: 序列化房间
        rooms_data = self.serialize_contours(
            contours, graph, colors, order, transformer
        )

        # step6: 美化框 (s10)
        if robot_model and "s10" in robot_model.lower():
            self.set_beautifier_status(True)
        _bbox_list, _threshold_list = self.beautify_contours(contours, map_img)

        # step7: 标记点 (K20)
        landmarks_data = []
        if robot_model == "S-K20PRO":
            marker_polygons = self._get_marker_polygons(map_data)
            landmarks_data = self.build_landmarks(
                rooms_data, landmark_builder,
                marker_polygons=marker_polygons,
            )

        # step8: 组装输出
        labels = {
            "version": self.config.get("labels_version", f"v{self.config.get('service_version', '4.0.2')}"),
            "uuid": map_data.get("uuid", ""),
            "data": rooms_data + landmarks_data,
        }

        logger.info(f"分区完成: {len(rooms_data)} 个房间, {len(landmarks_data)} 个标记点")
        return labels

    # ==================== 内部工具 ====================

    @staticmethod
    def _extract_contours(label_map: np.ndarray) -> List[np.ndarray]:
        """从 label_map 提取各房间轮廓"""
        contours = []
        if label_map is None or label_map.max() == 0:
            return contours
        for lid in range(1, label_map.max() + 1):
            mask = (label_map == lid).astype(np.uint8)
            if mask.sum() == 0:
                continue
            cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
            if cnts:
                contours.append(max(cnts, key=cv2.contourArea))
        return contours

    @staticmethod
    def _relabel(label_map: np.ndarray) -> np.ndarray:
        """重编号使 label ID 从 1 开始连续"""
        unique_ids = sorted(set(label_map.flat) - {0})
        new_map = np.zeros_like(label_map)
        for new_id, old_id in enumerate(unique_ids, start=1):
            new_map[label_map == old_id] = new_id
        return new_map

    @staticmethod
    def _contours_to_label_map(contours: List[np.ndarray],
                               shape: Tuple[int, int]) -> np.ndarray:
        """轮廓列表 → label_map"""
        h, w = shape[:2]
        label_map = np.zeros((h, w), dtype=np.int32)
        for i, cnt in enumerate(contours):
            cv2.drawContours(label_map, [cnt], -1, i + 1, -1)
        return label_map

    @staticmethod
    def _to_gray(img: np.ndarray) -> np.ndarray:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img.copy()

    def _restore_contours(
        self,
        labels_json: Dict[str, Any],
        transformer: CoordinateTransformer,
        _map_img: np.ndarray,
    ) -> List[np.ndarray]:
        """从已有 labels_json 恢复轮廓"""
        rooms_data = [d for d in labels_json.get("data", [])
                      if "ROOM" in d.get("id", "")]
        return transformer.rooms_data_to_contours(rooms_data)

    @staticmethod
    def _get_charge_pixel(
        map_data: Dict[str, Any],
        transformer: CoordinateTransformer,
    ) -> Optional[Tuple[int, int]]:
        """充电桩世界坐标 → 像素坐标"""
        pose = map_data.get("world_charge_pose", [0, 0, 0])
        if pose == [0, 0, 0]:
            return None
        return transformer.world_to_pixel(pose[0], pose[1])

    @staticmethod
    def _get_marker_polygons(map_data: Dict[str, Any]) -> Optional[List]:
        """提取家具标记多边形"""
        markers_json = map_data.get("markers_json")
        if not markers_json:
            return None
        polys = []
        for item in markers_json.get("data", []):
            if item.get("name") == "家具" and item.get("type") == "polygon":
                polys.append(item["geometry"])
        return polys if polys else None

    # ==================== 房间统计 (静态工具) ====================

    @staticmethod
    def get_room_info(label_map: np.ndarray,
                      resolution: float = 0.05) -> List[Dict[str, Any]]:
        """获取各房间统计信息"""
        rooms = []
        if label_map is None or label_map.max() == 0:
            return rooms
        for lid in range(1, label_map.max() + 1):
            mask = label_map == lid
            if not mask.any():
                continue
            ys, xs = np.where(mask)
            rooms.append({
                "id": int(lid),
                "area": float(mask.sum() * resolution ** 2),
                "center": (float(xs.mean()), float(ys.mean())),
                "bbox": (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())),
            })
        return rooms
