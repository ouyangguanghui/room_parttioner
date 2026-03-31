"""CoordinateTransformer 稳定性回归测试。"""

import numpy as np

from app.utils.coordinate import CoordinateTransformer


def test_pixel_world_roundtrip_is_stable():
    transformer = CoordinateTransformer(
        resolution=0.05,
        origin=[-10.0, -10.0],
        height=400,
    )
    samples = [(0, 0), (1, 1), (17, 93), (199, 245), (399, 399)]
    for px, py in samples:
        wx, wy = transformer.pixel_to_world(px, py)
        back_px, back_py = transformer.world_to_pixel(wx, wy)
        assert (back_px, back_py) == (px, py)


def test_contour_geometry_roundtrip_keeps_pixels():
    transformer = CoordinateTransformer(
        resolution=0.05,
        origin=[0.0, 0.0],
        height=120,
    )
    contour = np.array(
        [[[10, 10]], [[30, 10]], [[30, 30]], [[10, 30]]],
        dtype=np.int32,
    )
    geometry = transformer.contour_to_geometry(contour, clockwise=True)
    reconstructed = transformer.world_to_contour(geometry)
    np.testing.assert_array_equal(reconstructed.reshape(-1, 2), contour.reshape(-1, 2))
