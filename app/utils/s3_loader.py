"""S3 数据加载/存储层 —— 与旧 app.py 完全一致的 S3 路径约定"""

import json
import logging
from typing import Dict, Any, Optional, Tuple

import numpy as np
import cv2
import boto3
from botocore.exceptions import ClientError

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
        self.s3 = boto3.client('s3')

        # 构造 S3 路径
        self.paths = {
            "map_png": f"{key}/saved_map.png",
            "map_json": f"{key}/saved_map.json",
            "labels_json": f"{key}/labels.json",
            "mapinfo_json": f"{key}/mapinfo.json",
            "markers_json": f"{key}/markers.json",
        }

    def load(self) -> Dict[str, Any]:
        """
        加载所有数据

        Returns:
            {
                "map_img": np.ndarray (H,W,3) BGR,
                "resolution": float,
                "origin": [x, y],
                "labels_json": dict or None,
                "robot_model": str,
                "uuid": str or None,
                "markers_json": dict or None,
                "world_charge_pose": [x, y, theta],
            }

        Raises:
            ClientError: S3 读取失败 (labels.json 除外)
        """
        data = {}

        # ---- 地图 PNG ----
        obj = self.s3.get_object(Bucket=self.bucket, Key=self.paths["map_png"])
        buf = np.frombuffer(obj['Body'].read(), np.uint8)
        map_img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        # 像素值归一化 (与旧代码一致)
        map_img[map_img < 100] = 0
        map_img[((100 <= map_img) & (map_img <= 250))] = 127
        map_img[map_img > 250] = 255
        data["map_img"] = map_img

        # ---- 地图 JSON ----
        obj = self.s3.get_object(Bucket=self.bucket, Key=self.paths["map_json"])
        map_json = json.loads(obj['Body'].read().decode('utf-8'))
        data["resolution"] = map_json['resolution']
        data["origin"] = map_json['origin']

        # ---- labels.json (可能不存在) ----
        try:
            obj = self.s3.get_object(Bucket=self.bucket, Key=self.paths["labels_json"])
            data["labels_json"] = json.loads(obj['Body'].read().decode('utf-8'))
        except ClientError as e:
            if 'labels.json' in str(e):
                data["labels_json"] = None
            else:
                raise
        except Exception:
            data["labels_json"] = None

        # ---- mapinfo.json ----
        try:
            obj = self.s3.get_object(Bucket=self.bucket, Key=self.paths["mapinfo_json"])
            mapinfo = json.loads(obj['Body'].read().decode('utf-8'))
        except Exception:
            mapinfo = {}
        data["robot_model"] = mapinfo.get('robot_model', 's10')
        data["uuid"] = mapinfo.get('uuid', None)

        # ---- markers.json (k20 充电桩) ----
        data["markers_json"] = None
        data["world_charge_pose"] = [0, 0, 0]
        if data["robot_model"] == "S-K20PRO":
            try:
                obj = self.s3.get_object(Bucket=self.bucket,
                                         Key=self.paths["markers_json"])
                markers = json.loads(obj['Body'].read().decode('utf-8'))
                data["markers_json"] = markers
                # 解析充电桩位置
                for item in markers.get('data', []):
                    if "CHARGE" in item.get("name", ""):
                        data["world_charge_pose"] = item.get('geometry', [0, 0, 0])
                        break
            except Exception as e:
                logger.warning(f"加载 markers.json 失败: {e}")

        return data

    def save_labels(self, labels_json: Dict[str, Any]):
        """写回 labels.json 到 S3"""
        body = json.dumps(labels_json, ensure_ascii=False)
        self.s3.put_object(
            Bucket=self.bucket,
            Key=self.paths["labels_json"],
            Body=body.encode('utf-8'),
            ContentType='application/json',
        )
        logger.info(f"labels.json 已写回 s3://{self.bucket}/{self.paths['labels_json']}")
