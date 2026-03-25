"""S3DataLoader 单元测试（重点覆盖调试模式本地读写）。"""

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
import pytest

from app.utils.s3_loader import S3DataLoader


def _write_basic_local_dataset(base: Path):
    """写入最小可用本地调试数据集。"""
    img = np.zeros((24, 24, 3), dtype=np.uint8)
    img[4:20, 4:20] = 254
    cv2.imwrite(str(base / "saved_map.png"), img)
    (base / "saved_map.json").write_text(
        json.dumps({"resolution": 0.05, "origin": [0.0, 0.0]}), encoding="utf-8"
    )
    (base / "mapinfo.json").write_text(
        json.dumps({"robot_model": "s10", "uuid": "u-test"}), encoding="utf-8"
    )


def test_debug_mode_load_and_save_local_dir():
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        _write_basic_local_dataset(root)

        with patch.dict(
            os.environ,
            {
                "ROOM_PARTITIONER_DEBUG": "1",
                "ROOM_PARTITIONER_LOCAL_DIR": str(root),
            },
            clear=False,
        ):
            loader = S3DataLoader(bucket="dummy", key="ignored")
            data = loader.load()
            assert "map_img" in data
            assert data["resolution"] == 0.05
            assert data["origin"] == [0.0, 0.0]
            assert data["robot_model"] == "s10"
            assert data["uuid"] == "u-test"

            payload = {"version": "v1", "data": []}
            loader.save_labels(payload)
            out = json.loads((root / "labels.json").read_text(encoding="utf-8"))
            assert out["version"] == "v1"


def test_non_debug_without_boto3_raises_import_error():
    # 强制模拟“非调试 + boto3 缺失”
    with patch("app.utils.s3_loader.boto3", None):
        with patch.dict(os.environ, {"ROOM_PARTITIONER_DEBUG": "0"}, clear=False):
            with pytest.raises(ImportError):
                S3DataLoader(bucket="dummy", key="k")


def test_debug_without_local_dir_raises():
    with patch.dict(os.environ, {"ROOM_PARTITIONER_DEBUG": "1"}, clear=False):
        os.environ.pop("ROOM_PARTITIONER_LOCAL_DIR", None)
        with pytest.raises(FileNotFoundError):
            S3DataLoader(bucket="dummy", key="ignored")

