"""S3 数据加载/存储层 —— 与旧 app.py 完全一致的 S3 路径约定"""

import json
import os
import logging
from pathlib import Path
from typing import Dict, Any

import numpy as np
import cv2
try:
    import boto3
    from botocore.exceptions import ClientError
except Exception:  # 允许开发环境未安装 boto3（本地调试模式可工作）
    boto3 = None
    ClientError = Exception

logger = logging.getLogger(__name__)


class S3DataLoader:
    """
    S3 数据读写, 兼容旧接口的路径约定:

        {key}/saved_map.png     → 栅格地图
        {key}/saved_map.json    → 地图元信息 (resolution, origin)
        {key}/labels.json       → 房间标注数据
        {key}/mapinfo.json      → 设备信息 (robot_model, uuid)
        {key}/markers.json      → 标记信息 (充电桩等, k20)
    """

    def __init__(self, bucket: str, key: str):
        self.bucket = bucket
        self.key = key
        self.debug_mode = self._is_debug_mode()
        if self.debug_mode:
            self.s3 = None
        else:
            if boto3 is None:
                raise ImportError("非调试模式需要 boto3，请先安装 boto3")
            self.s3 = boto3.client("s3")

        # 构造 S3 路径
        self.paths = {
            "map_png": f"{key}/saved_map.png",
            "map_json": f"{key}/saved_map.json",
            "labels_json": f"{key}/labels.json",
            "mapinfo_json": f"{key}/mapinfo.json",
            "markers_json": f"{key}/markers.json",
        }
        self.local_dir = self._resolve_local_dir() if self.debug_mode else None

    @staticmethod
    def _is_debug_mode() -> bool:
        """是否开启本地调试模式。"""
        v = os.getenv("ROOM_PARTITIONER_DEBUG", "").strip().lower()
        return v in {"1", "true", "yes", "on"}

    def _resolve_local_dir(self) -> Path:
        """
        解析本地调试目录（仅支持 ROOM_PARTITIONER_LOCAL_DIR）。
        """
        local_dir = os.getenv("ROOM_PARTITIONER_LOCAL_DIR", "").strip()
        if local_dir:
            p = Path(local_dir).expanduser().resolve()
            if not p.exists():
                raise FileNotFoundError(f"本地调试目录不存在: {p}")
            return p

        raise FileNotFoundError(
            "调试模式已开启，但未找到本地数据目录。请设置 ROOM_PARTITIONER_LOCAL_DIR"
        )


    @staticmethod
    def _read_json_file(path: Path) -> Dict[str, Any]:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)

    def _load_from_local(self) -> Dict[str, Any]:
        """
        从本地目录加载数据（调试模式）:
            <local_dir>/saved_map.png
            <local_dir>/saved_map.json
            <local_dir>/labels.json
            <local_dir>/mapinfo.json
            <local_dir>/markers.json
        """
        assert self.local_dir is not None
        data: Dict[str, Any] = {}

        map_png = self.local_dir / "saved_map.png"
        map_json = self.local_dir / "saved_map.json"
        labels_json = self.local_dir / "labels.json"
        mapinfo_json = self.local_dir / "mapinfo.json"
        markers_json = self.local_dir / "markers.json"

        map_img = cv2.imread(str(map_png), cv2.IMREAD_COLOR)
        if map_img is None:
            raise FileNotFoundError(f"无法读取地图文件: {map_png}")
        data["map_img"] = map_img

        map_meta = self._read_json_file(map_json)
        data["resolution"] = map_meta["resolution"]
        data["origin"] = map_meta["origin"]

        if labels_json.exists():
            try:
                data["labels_json"] = self._read_json_file(labels_json)
            except Exception:
                data["labels_json"] = None
        else:
            data["labels_json"] = None

        if mapinfo_json.exists():
            try:
                mapinfo = self._read_json_file(mapinfo_json)
            except Exception:
                mapinfo = {}
        else:
            mapinfo = {}
        data["robot_model"] = mapinfo.get("robot_model", "s10")
        data["uuid"] = mapinfo.get("uuid", None)

        data["markers_json"] = None
        data["world_charge_pose"] = [0, 0, 0]
        if data["robot_model"] == "S-K20PRO" and markers_json.exists():
            try:
                markers = self._read_json_file(markers_json)
                data["markers_json"] = markers
                for item in markers.get("data", []):
                    if "CHARGE" in item.get("name", ""):
                        data["world_charge_pose"] = item.get("geometry", [0, 0, 0])
                        break
            except Exception as e:
                logger.warning(f"加载本地 markers.json 失败: {e}")

        logger.info(f"[DEBUG] 已从本地加载数据: {self.local_dir}")
        return data

    def _load_from_s3(self) -> Dict[str, Any]:
        """从 S3 加载数据（线上模式）。"""
        assert self.s3 is not None
        data: Dict[str, Any] = {}

        # ---- 地图 PNG ----
        obj = self.s3.get_object(Bucket=self.bucket, Key=self.paths["map_png"])
        buf = np.frombuffer(obj["Body"].read(), np.uint8)
        map_img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        data["map_img"] = map_img

        # ---- 地图 JSON ----
        obj = self.s3.get_object(Bucket=self.bucket, Key=self.paths["map_json"])
        map_json = json.loads(obj["Body"].read().decode("utf-8"))
        data["resolution"] = map_json["resolution"]
        data["origin"] = map_json["origin"]

        # ---- labels.json (可能不存在) ----
        try:
            obj = self.s3.get_object(Bucket=self.bucket, Key=self.paths["labels_json"])
            data["labels_json"] = json.loads(obj["Body"].read().decode("utf-8"))
        except ClientError as e:
            if "labels.json" in str(e):
                data["labels_json"] = None
            else:
                raise
        except Exception:
            data["labels_json"] = None

        # ---- mapinfo.json ----
        try:
            obj = self.s3.get_object(Bucket=self.bucket, Key=self.paths["mapinfo_json"])
            mapinfo = json.loads(obj["Body"].read().decode("utf-8"))
        except Exception:
            mapinfo = {}
        data["robot_model"] = mapinfo.get("robot_model", "s10")
        data["uuid"] = mapinfo.get("uuid", None)

        # ---- markers.json (k20 充电桩) ----
        data["markers_json"] = None
        data["world_charge_pose"] = [0, 0, 0]
        if data["robot_model"] == "S-K20PRO":
            try:
                obj = self.s3.get_object(Bucket=self.bucket, Key=self.paths["markers_json"])
                markers = json.loads(obj["Body"].read().decode("utf-8"))
                data["markers_json"] = markers
                # 解析充电桩位置
                for item in markers.get("data", []):
                    if "CHARGE" in item.get("name", ""):
                        data["world_charge_pose"] = item.get("geometry", [0, 0, 0])
                        break
            except Exception as e:
                logger.warning(f"加载 markers.json 失败: {e}")

        return data

    def load(self) -> Dict[str, Any]:
        """
        加载所有数据。

        调试模式（ROOM_PARTITIONER_DEBUG=1）:
            从本地目录读取
        非调试模式:
            正常从 S3 读取
        return: Dict[str, Any]
            "map_img": 地图图像 (H, W, 3) uint8
            "resolution": 地图分辨率 float
            "origin": 地图原点 [x, y] float
            "labels_json": 房间标注数据 json
            "robot_model": 机器人型号 str
            "uuid": 机器人 UUID str
            "markers_json": 标记信息 json
            "world_charge_pose": 充电桩世界坐标 [x, y, z] float
        """
        if self.debug_mode:
            return self._load_from_local()
        return self._load_from_s3()

    def save_labels(self, labels_json: Dict[str, Any]):
        """保存 labels.json（调试模式写本地，否则写回 S3）。"""
        body = json.dumps(labels_json, ensure_ascii=False, indent=2)
        if self.debug_mode:
            assert self.local_dir is not None
            out = self.local_dir / "labels.json"
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(body, encoding="utf-8")
            logger.info(f"[DEBUG] labels.json 已写回本地: {out}")
            return

        assert self.s3 is not None
        self.s3.put_object(
            Bucket=self.bucket,
            Key=self.paths["labels_json"],
            Body=body.encode("utf-8"),
            ContentType="application/json",
        )
        logger.info(f"labels.json 已写回 s3://{self.bucket}/{self.paths['labels_json']}")
