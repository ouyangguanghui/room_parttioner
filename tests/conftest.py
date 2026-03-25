"""公共 fixtures"""

import numpy as np
import pytest

from app.core.config import load_config


@pytest.fixture
def config():
    """基础配置（降低面积阈值方便测试）"""
    return load_config(overrides={"min_room_area": 0.01, "wall_threshold": 128})


@pytest.fixture
def two_room_map():
    """两个房间被横墙隔开的 100x100 地图

    布局:
        255 255 255 ... 255   ← 顶墙
        255   0   0 ...  255  ← 房间A (行1~44)
        ...
        255 255 255 ... 255   ← 横墙 (行45~54)
        255   0   0 ...  255  ← 房间B (行55~98)
        255 255 255 ... 255   ← 底墙

    像素值: 0=空闲, 255=墙壁 (注意 wall_threshold=128, <128 为墙)
    所以这里 0 是空闲，255 是墙壁 → 但 auto_partition fallback 用 image < wall_threshold 判定墙
    实际: 0 < 128 → 被判为墙, 255 >= 128 → 被判为空闲
    """
    # 全黑 (0) = 墙壁
    grid = np.zeros((100, 100), dtype=np.uint8)
    # 白色区域 = 空闲空间
    grid[1:45, 1:99] = 255   # 房间A
    grid[55:99, 1:99] = 255  # 房间B
    # 行0, 行45-54, 行99 以及列0, 列99 保持为0（墙壁）
    return grid


@pytest.fixture
def single_room_map():
    """单个大房间的地图"""
    grid = np.zeros((100, 100), dtype=np.uint8)
    grid[1:99, 1:99] = 255
    return grid


@pytest.fixture
def empty_map():
    """全是墙壁（无房间）"""
    return np.zeros((100, 100), dtype=np.uint8)
