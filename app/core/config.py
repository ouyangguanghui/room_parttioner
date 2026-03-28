"""统一配置加载：yaml + 环境变量覆盖"""

import os
from pathlib import Path
from typing import Dict, Any

import yaml

_CONFIG_DIR = Path(__file__).resolve().parent.parent.parent / "config"


_DEFAULT_CONFIG = {
    # 版本
    "service_version": "4.0.2",
    "model_version_tag": "0.0.1",
    # 前处理
    "target_size": [512, 512],
    "min_input_size": 416,
    "normalize": True,
    "mean": [0.485, 0.456, 0.406],
    "std": [0.229, 0.224, 0.225],
    # 推理
    "triton_url": "",
    "model_name": "room_seg",
    "model_version": "",
    "triton_protocol": "grpc",
    "input_name": "input",
    "output_name": "output",
    # 后处理
    "morph_kernel_size": 5,
    # 通用
    "min_room_area": 1.0,
    "wall_threshold": 128,
    "resolution": 0.05,
    # 扩展分区
    "door_width": 20,
    "grow_iterations": 10,
    # 手动划分
    "line_thickness": 3,
}

# 环境变量映射：ENV_NAME -> (config_key, type)
_ENV_MAP = {
    "SERVICE_VERSION": ("service_version", str),
    "MODEL_VERSION_TAG": ("model_version_tag", str),
    "TRITON_URL": ("triton_url", str),
    "MODEL_NAME": ("model_name", str),
    "MODEL_VERSION": ("model_version", str),
    "TRITON_PROTOCOL": ("triton_protocol", str),
    "INPUT_NAME": ("input_name", str),
    "OUTPUT_NAME": ("output_name", str),
    "MIN_ROOM_AREA": ("min_room_area", float),
    "WALL_THRESHOLD": ("wall_threshold", int),
    "RESOLUTION": ("resolution", float),
    "DOOR_WIDTH": ("door_width", int),
    "GROW_ITERATIONS": ("grow_iterations", int),
    "LINE_THICKNESS": ("line_thickness", int),
    "MORPH_KERNEL_SIZE": ("morph_kernel_size", int),
    "NORMALIZE": ("normalize", lambda x: x.lower() in ("1", "true", "yes")),
    "MIN_INPUT_SIZE": ("min_input_size", int),
}


def load_config(yaml_path: str = None, overrides: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    加载配置，优先级：环境变量 > overrides > yaml > 默认值
    """
    config = dict(_DEFAULT_CONFIG)

    # 1. 从 yaml 加载
    if yaml_path is None:
        yaml_path = str(_CONFIG_DIR / "default.yaml")
    if os.path.exists(yaml_path):
        with open(yaml_path, "r") as f:
            yaml_config = yaml.safe_load(f) or {}
        config.update(yaml_config)

    # 2. 代码覆盖
    if overrides:
        config.update(overrides)

    # 3. 环境变量覆盖（最高优先级）
    for env_name, (key, cast) in _ENV_MAP.items():
        val = os.getenv(env_name)
        if val is not None:
            config[key] = cast(val)

    # 特殊处理 TARGET_SIZE: "512,512" -> [512, 512]
    target_size_env = os.getenv("TARGET_SIZE")
    if target_size_env:
        config["target_size"] = [int(x) for x in target_size_env.split(",")]

    # 自动拼接 labels_version: "online_{service_version}_{model_version_tag}"
    sv = config.get("service_version", "")
    mv = config.get("model_version_tag", "")
    config["labels_version"] = f"v{sv}_{mv}" if mv else f"v{sv}"

    return config
