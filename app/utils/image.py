"""图像编解码工具函数"""

import io

import numpy as np
import cv2
from fastapi.responses import StreamingResponse


def decode_image(data: bytes) -> np.ndarray:
    """将上传的字节数据解码为灰度图"""
    arr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("无法解析图片")
    return img


def label_to_color(label_map: np.ndarray) -> np.ndarray:
    """label_map → 可视化彩色图"""
    color_map = np.zeros((*label_map.shape, 3), dtype=np.uint8)
    for lid in range(1, label_map.max() + 1):
        color = np.random.RandomState(lid).randint(50, 255, 3).tolist()
        color_map[label_map == lid] = color
    return color_map


def encode_png(image: np.ndarray) -> StreamingResponse:
    """ndarray → PNG StreamingResponse"""
    _, buf = cv2.imencode(".png", image)
    return StreamingResponse(io.BytesIO(buf.tobytes()), media_type="image/png")
