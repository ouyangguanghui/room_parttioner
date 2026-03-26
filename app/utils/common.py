"""
逐步功能验证脚本 —— 用真实地图数据逐个测试每个模块

用法:
    cd /home/ouyang/project/RoomPartitioner
    
    # Step 1: 前处理
    PYTHONPATH=. python tests/verify_step1_preprocess.py
    
查看输出: tests/output/ 目录下的图片
"""

import sys
import numpy as np
import cv2
from pathlib import Path

# 输出目录
OUT_DIR = Path("tests/output")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# 默认测试地图
DEFAULT_MAP = "dataset/origin_data/000064.png"


def load_map(path: str = None) -> np.ndarray:
    """加载地图"""
    path = path or DEFAULT_MAP
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"❌ 无法读取地图: {path}")
        sys.exit(1)
    print(f"✅ 加载地图: {path}")
    print(f"   形状: {img.shape}, 像素值: {np.unique(img)}")
    return img


def save_result(name: str, img: np.ndarray):
    """保存结果图"""
    path = OUT_DIR / f"{name}.png"
    cv2.imwrite(str(path), img)
    print(f"   💾 已保存: {path}")


def label_to_color(label_map: np.ndarray) -> np.ndarray:
    """标签图转彩色可视化"""
    color_map = np.zeros((*label_map.shape, 3), dtype=np.uint8)
    for lid in range(1, label_map.max() + 1):
        color = np.random.RandomState(lid).randint(50, 255, 3).tolist()
        color_map[label_map == lid] = color
    return color_map


def show_summary(label_map: np.ndarray, resolution: float = 0.05):
    """打印房间摘要"""
    from app.services.auto_partition import AutoPartitioner
    rooms = AutoPartitioner.get_room_info(label_map, resolution)
    print(f"   房间数: {len(rooms)}")
    for r in rooms:
        print(f"   📦 房间 {r['id']}: 面积={r['area']:.2f}m², 中心=({r['center'][0]:.0f}, {r['center'][1]:.0f})")
    return rooms