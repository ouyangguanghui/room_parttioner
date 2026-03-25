"""Preprocessor 单元测试（pytest 风格）。"""

import numpy as np

from app.pipeline.preprocessor import Preprocessor


def test_map_pixels_maps_known_values():
    pp = Preprocessor({})
    src = np.array([[0, 247, 254]], dtype=np.uint8)
    mapped = pp._map_pixels(src)
    assert mapped.tolist() == [[0, 127, 255]]


def test_remove_interior_noise_removes_small_black_component():
    pp = Preprocessor({"max_noise_area": 10, "dilate_iter": 1})
    img = np.full((20, 20), 255, dtype=np.uint8)
    img[10, 10] = 0  # 自由空间内部黑色噪点

    cleaned, noise_mask, stats = pp.remove_interior_noise(img)

    assert cleaned[10, 10] == 255
    assert noise_mask[10, 10] == 255
    assert stats["removed"] >= 1


def test_process_returns_expected_meta_structure():
    pp = Preprocessor({"max_noise_area": 5, "smooth_iterations": 1})
    img = np.full((32, 32), 254, dtype=np.uint8)
    img[0, :] = 0
    img[:, 0] = 0
    img[15, 15] = 247

    out = pp.process(img)

    assert set(out.keys()) == {"map_data", "input_data"}
    assert out["map_data"].shape == img.shape
    assert out["input_data"].shape == img.shape
