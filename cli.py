"""RoomPartitioner CLI 入口

本地调试用：读取栅格地图 → 自动分区 → 输出彩色标记图 + 房间信息
"""

import argparse

import cv2
import numpy as np

from app.core.config import load_config
from app.pipeline.preprocessor import Preprocessor
from app.services.auto_partition import AutoPartitioner
from app.utils.coordinate import CoordinateTransformer
from app.utils.graph import RoomGraph
from app.utils.landmark import LandmarkManager


def label_to_color(label_map: np.ndarray) -> np.ndarray:
    """标签图 → 彩色可视化"""
    color_map = np.zeros((*label_map.shape, 3), dtype=np.uint8)
    for lid in range(1, label_map.max() + 1):
        color = np.random.RandomState(lid).randint(50, 255, 3).tolist()
        color_map[label_map == lid] = color
    return color_map


def main():
    parser = argparse.ArgumentParser(description="RoomPartitioner - 房间划分服务")
    parser.add_argument("--input", "-i", required=True, help="输入栅格地图 (png/pgm)")
    parser.add_argument("--output", "-o", default="result.png", help="输出标记图")
    parser.add_argument("--config", "-c", default="config/default.yaml", help="配置文件路径")
    parser.add_argument("--resolution", "-r", type=float, default=None, help="地图分辨率 m/pixel")
    parser.add_argument("--min-area", type=float, default=None, help="最小房间面积 m²")
    parser.add_argument("--no-extend", action="store_true", help="跳过扩展分区")
    args = parser.parse_args()

    overrides = {}
    if args.resolution is not None:
        overrides["resolution"] = args.resolution
    if args.min_area is not None:
        overrides["min_room_area"] = args.min_area
    config = load_config(args.config, overrides)

    # 读取地图
    grid_map = cv2.imread(args.input, cv2.IMREAD_GRAYSCALE)
    if grid_map is None:
        print(f"错误: 无法读取地图 {args.input}")
        return

    resolution = config["resolution"]
    map_img = cv2.cvtColor(grid_map, cv2.COLOR_GRAY2BGR)

    # 预处理
    preprocessor = Preprocessor(config)
    meta = preprocessor.process(grid_map)

    # 自动分区
    partitioner = AutoPartitioner(config)
    transformer = CoordinateTransformer(resolution, [0, 0], grid_map.shape[0])
    graph_builder = RoomGraph(config)
    landmark_builder = LandmarkManager(config)

    map_data = {
        "map_img": map_img,
        "resolution": resolution,
        "origin": [0, 0],
        "labels_json": None,
        "robot_model": "s10",
        "uuid": "cli",
        "markers_json": None,
        "world_charge_pose": [0, 0, 0],
    }

    labels_json = partitioner.process(
        map_data, meta, transformer, graph_builder, landmark_builder,
        extend=not args.no_extend,
    )

    # 输出结果
    rooms_data = [d for d in labels_json.get("data", []) if "ROOM" in d.get("id", "")]
    print(f"划分完成，共 {len(rooms_data)} 个房间")
    for r in rooms_data:
        print(f"  {r['id']} ({r['name']})")

    # 生成可视化
    partition_result = partitioner.partition(meta, extend=not args.no_extend)
    cv2.imwrite(args.output, label_to_color(partition_result["label_map"]))
    print(f"结果已保存: {args.output}")


if __name__ == "__main__":
    main()
