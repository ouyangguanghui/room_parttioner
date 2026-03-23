"""API 路由 —— 仅暴露 4 个业务服务 + 健康检查"""

import io
import numpy as np
import cv2
from fastapi import APIRouter, UploadFile, File, Query
from fastapi.responses import StreamingResponse, JSONResponse

from app.schemas.requests import (
    LineRequest, PolylineRequest, PolygonRequest,
    MergeRequest, MergePointRequest,
)

router = APIRouter()

_partitioner = None
_config = None


def init(partitioner, config):
    global _partitioner, _config
    _partitioner = partitioner
    _config = config


def _decode_image(data: bytes) -> np.ndarray:
    arr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("无法解析图片")
    return img


def _label_to_color(label_map: np.ndarray) -> np.ndarray:
    color_map = np.zeros((*label_map.shape, 3), dtype=np.uint8)
    for lid in range(1, label_map.max() + 1):
        color = np.random.RandomState(lid).randint(50, 255, 3).tolist()
        color_map[label_map == lid] = color
    return color_map


def _encode_png(image: np.ndarray) -> StreamingResponse:
    _, buf = cv2.imencode(".png", image)
    return StreamingResponse(io.BytesIO(buf.tobytes()), media_type="image/png")


def _check_state():
    if _partitioner.label_map is None:
        return JSONResponse(status_code=400, content={"error": "请先执行自动分区"})
    return None


def _get_resolution(res: float = None) -> float:
    return res or _config["resolution"]


# ==================== 健康检查 ====================

@router.get("/health")
def health():
    return {"status": "ok"}


# ==================== 服务 1: 自动分区 ====================

@router.post("/auto_partition")
async def auto_partition(
    file: UploadFile = File(..., description="栅格地图"),
    res: float = Query(default=None, description="分辨率 m/pixel"),
    extend: bool = Query(default=True, description="是否执行扩展分区"),
):
    """自动分区: 上传地图 → 完整分区流程 → 返回房间 JSON"""
    img = _decode_image(await file.read())
    _partitioner.auto_partition(img, extend=extend)
    rooms = _partitioner.get_room_info(_get_resolution(res))
    return {"room_count": len(rooms), "rooms": rooms}


@router.post("/auto_partition/image")
async def auto_partition_image(
    file: UploadFile = File(...),
    extend: bool = Query(default=True),
):
    """自动分区: 返回可视化图片"""
    img = _decode_image(await file.read())
    label_map = _partitioner.auto_partition(img, extend=extend)
    return _encode_png(_label_to_color(label_map))


# ==================== 服务 2: 扩展分区 ====================

@router.post("/extend_partition")
async def extend_partition(res: float = Query(default=None)):
    """扩展分区: 对当前结果执行门口检测 + 区域生长"""
    err = _check_state()
    if err:
        return err
    _partitioner.extend_partition()
    rooms = _partitioner.get_room_info(_get_resolution(res))
    return {"room_count": len(rooms), "rooms": rooms}


# ==================== 服务 3: 手动分割 ====================

@router.post("/manual/split_line")
async def split_by_line(req: LineRequest, res: float = Query(default=None)):
    """画线分割房间"""
    err = _check_state()
    if err:
        return err
    _partitioner.split_by_line(tuple(req.pt1), tuple(req.pt2))
    rooms = _partitioner.get_room_info(_get_resolution(res))
    return {"room_count": len(rooms), "rooms": rooms}


@router.post("/manual/split_polyline")
async def split_by_polyline(req: PolylineRequest, res: float = Query(default=None)):
    """折线分割房间"""
    err = _check_state()
    if err:
        return err
    points = [tuple(p) for p in req.points]
    _partitioner.split_by_polyline(points)
    rooms = _partitioner.get_room_info(_get_resolution(res))
    return {"room_count": len(rooms), "rooms": rooms}


@router.post("/manual/assign_polygon")
async def assign_polygon(req: PolygonRequest, res: float = Query(default=None)):
    """多边形划定新房间"""
    err = _check_state()
    if err:
        return err
    polygon = [tuple(p) for p in req.polygon]
    _partitioner.assign_polygon(polygon, req.room_id)
    rooms = _partitioner.get_room_info(_get_resolution(res))
    return {"room_count": len(rooms), "rooms": rooms}


# ==================== 服务 4: 手动合并 ====================

@router.post("/manual/merge")
async def merge_rooms(req: MergeRequest, res: float = Query(default=None)):
    """合并指定房间"""
    err = _check_state()
    if err:
        return err
    _partitioner.merge_rooms(req.room_ids)
    rooms = _partitioner.get_room_info(_get_resolution(res))
    return {"room_count": len(rooms), "rooms": rooms}


@router.post("/manual/merge_by_point")
async def merge_by_point(req: MergePointRequest, res: float = Query(default=None)):
    """点选合并两个房间"""
    err = _check_state()
    if err:
        return err
    _partitioner.merge_by_point(tuple(req.pt1), tuple(req.pt2))
    rooms = _partitioner.get_room_info(_get_resolution(res))
    return {"room_count": len(rooms), "rooms": rooms}


# ==================== 状态查询 (辅助调试) ====================

@router.get("/current/image")
async def current_image():
    """获取当前分区可视化图"""
    err = _check_state()
    if err:
        return err
    return _encode_png(_label_to_color(_partitioner.label_map))


@router.get("/current/info")
async def current_info(res: float = Query(default=None)):
    """获取当前分区房间信息"""
    err = _check_state()
    if err:
        return err
    rooms = _partitioner.get_room_info(_get_resolution(res))
    return {"room_count": len(rooms), "rooms": rooms}
