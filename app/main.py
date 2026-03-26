"""FastAPI HTTP 入口 —— 本地开发 / HTTP 部署用

与 Lambda handler 完全等价的 HTTP 接口，方便本地调试和非 Lambda 部署场景。

启动:
    uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 1
"""

import json
import logging
from typing import Dict, Any, List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from app.core.config import load_config
from app.core.errors import RoomPartitionerError
from app.services.services import RoomService
from app.utils.s3_loader import S3DataLoader

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("room_partitioner")

app = FastAPI(title="RoomPartitioner", version="4.0.2")

# 全局单例：避免每次请求重复初始化
_config = load_config()
_service = RoomService(_config)


# ==================== 请求模型 ====================

class PartitionRequest(BaseModel):
    operation: str
    bucket: str
    key: str
    roomMergeList: Optional[List[str]] = None
    divisionCroodsDict: Optional[Dict[str, Any]] = None


# ==================== 路由 ====================

@app.post("/room_edit")
def room_edit(req: PartitionRequest):
    """房间编辑 —— 与 Lambda handler 完全等价"""

    if req.operation not in ("split", "repartition", "merge", "division"):
        raise HTTPException(status_code=171, detail="no such operation")

    if req.operation == "merge" and not req.roomMergeList:
        raise HTTPException(status_code=171, detail="need room merge list")

    if req.operation == "division" and not req.divisionCroodsDict:
        raise HTTPException(status_code=171, detail="need division croods list")

    try:
        loader = S3DataLoader(req.bucket, req.key)
        map_data = loader.load()

        labels_json = _service.room_edit(
            map_data=map_data,
            operation=req.operation,
            division_croods_dict=req.divisionCroodsDict,
            room_merge_list=req.roomMergeList,
        )

        return {"statusCode": 200, "body": labels_json}

    except RoomPartitionerError as e:
        logger.error(f"业务错误 (code={e.code}): {e}")
        return {"statusCode": 190, "body": str(e.code)}
    except Exception as e:
        logger.error(f"未知错误: {e}", exc_info=True)
        return {"statusCode": 190, "body": "0"}


@app.get("/health")
def health():
    return {"status": "ok"}
