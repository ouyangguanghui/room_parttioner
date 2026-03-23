"""RoomPartitioner CLI 入口"""

import argparse
import numpy as np
import cv2

from app.core.config import load_config
from app.core.partitioner import RoomPartitioner


def _label_to_color(label_map: np.ndarray) -> np.ndarray:
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

    grid_map = cv2.imread(args.input, cv2.IMREAD_GRAYSCALE)
    if grid_map is None:
        print(f"错误: 无法读取地图 {args.input}")
        return

    partitioner = RoomPartitioner(config)
    label_map = partitioner.auto_partition(grid_map, extend=not args.no_extend)
    rooms = partitioner.get_room_info(config["resolution"])

    cv2.imwrite(args.output, _label_to_color(label_map))

    print(f"划分完成，共 {len(rooms)} 个房间")
    for r in rooms:
        print(f"  房间 {r['id']}: 面积={r['area']:.2f}m², 中心={r['center']}")
    print(f"结果已保存: {args.output}")


if __name__ == "__main__":
    main()
