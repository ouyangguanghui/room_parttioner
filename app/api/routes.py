"""API 路由 —— 仅暴露 4 个业务服务 + 健康检查"""

import numpy as np
from fastapi import APIRouter, UploadFile, File, Query, Request
from fastapi.responses import JSONResponse

from app.schemas.requests import (
    LineRequest, PolylineRequest, PolygonRequest,
    MergeRequest, MergePointRequest,
)
from app.utils.image import decode_image, label_to_color, encode_png

router = APIRouter()


def _check_state(request: Request):
    partitioner = getattr(request.app.state, "partitioner", None)
    if partitioner is None or partitioner.label_map is None:
        return JSONResponse(status_code=400, content={"error": "请先执行自动分区"})
    return None


def _get_resolution(request: Request, res: float = None) -> float:
    config = getattr(request.app.state, "config", None) or {}
    return res if res is not None else config.get("resolution", 0.05)


# ==================== 健康检查 ====================

@router.get("/health")
def health():
    return {"status": "ok"}


# ==================== 服务 1: 自动分区 ====================

@router.post("/auto_partition")
async def auto_partition(
    request: Request,
    file: UploadFile = File(..., description="栅格地图"),
    res: float = Query(default=None, description="分辨率 m/pixel"),
    extend: bool = Query(default=True, description="是否执行扩展分区"),
):
    """自动分区: 上传地图 → 完整分区流程 → 返回房间 JSON"""
    partitioner = request.app.state.partitioner
    img = decode_image(await file.read())
    partitioner.auto_partition(img, extend=extend)
    rooms = partitioner.get_room_info(_get_resolution(request, res))
    return {"room_count": len(rooms), "rooms": rooms}


@router.post("/auto_partition/image")
async def auto_partition_image(
    request: Request,
    file: UploadFile = File(...),
    extend: bool = Query(default=True),
):
    """自动分区: 返回可视化图片"""
    partitioner = request.app.state.partitioner
    img = decode_image(await file.read())
    label_map = partitioner.auto_partition(img, extend=extend)
    return encode_png(label_to_color(label_map))


# ==================== 服务 2: 扩展分区 ====================

@router.post("/extend_partition")
async def extend_partition(request: Request, res: float = Query(default=None)):
    """扩展分区: 对当前结果执行门口检测 + 区域生长"""
    err = _check_state(request)
    if err:
        return err
    partitioner = request.app.state.partitioner
    partitioner.extend_partition()
    rooms = partitioner.get_room_info(_get_resolution(request, res))
    return {"room_count": len(rooms), "rooms": rooms}


# ==================== 服务 3: 手动分割 ====================

@router.post("/manual/split_line")
async def split_by_line(
    request: Request, req: LineRequest, res: float = Query(default=None)
):
    """画线分割房间"""
    err = _check_state(request)
    if err:
        return err
    partitioner = request.app.state.partitioner
    partitioner.split_by_line(tuple(req.pt1), tuple(req.pt2))
    rooms = partitioner.get_room_info(_get_resolution(request, res))
    return {"room_count": len(rooms), "rooms": rooms}


@router.post("/manual/split_polyline")
async def split_by_polyline(
    request: Request, req: PolylineRequest, res: float = Query(default=None)
):
    """折线分割房间"""
    err = _check_state(request)
    if err:
        return err
    partitioner = request.app.state.partitioner
    points = [tuple(p) for p in req.points]
    partitioner.split_by_polyline(points)
    rooms = partitioner.get_room_info(_get_resolution(request, res))
    return {"room_count": len(rooms), "rooms": rooms}


@router.post("/manual/assign_polygon")
async def assign_polygon(
    request: Request, req: PolygonRequest, res: float = Query(default=None)
):
    """多边形划定新房间"""
    err = _check_state(request)
    if err:
        return err
    partitioner = request.app.state.partitioner
    polygon = [tuple(p) for p in req.polygon]
    partitioner.assign_polygon(polygon, req.room_id)
    rooms = partitioner.get_room_info(_get_resolution(request, res))
    return {"room_count": len(rooms), "rooms": rooms}


# ==================== 服务 4: 手动合并 ====================

@router.post("/manual/merge")
async def merge_rooms(
    request: Request, req: MergeRequest, res: float = Query(default=None)
):
    """合并指定房间"""
    err = _check_state(request)
    if err:
        return err
    partitioner = request.app.state.partitioner
    partitioner.merge_rooms(req.room_ids)
    rooms = partitioner.get_room_info(_get_resolution(request, res))
    return {"room_count": len(rooms), "rooms": rooms}


@router.post("/manual/merge_by_point")
async def merge_by_point(
    request: Request, req: MergePointRequest, res: float = Query(default=None)
):
    """点选合并两个房间"""
    err = _check_state(request)
    if err:
        return err
    partitioner = request.app.state.partitioner
    partitioner.merge_by_point(tuple(req.pt1), tuple(req.pt2))
    rooms = partitioner.get_room_info(_get_resolution(request, res))
    return {"room_count": len(rooms), "rooms": rooms}


# ==================== 状态查询 (辅助调试) ====================

@router.get("/current/image")
async def current_image(request: Request):
    """获取当前分区可视化图"""
    err = _check_state(request)
    if err:
        return err
    partitioner = request.app.state.partitioner
    return encode_png(label_to_color(partitioner.label_map))


@router.get("/current/info")
async def current_info(request: Request, res: float = Query(default=None)):
    """获取当前分区房间信息"""
    err = _check_state(request)
    if err:
        return err
    partitioner = request.app.state.partitioner
    rooms = partitioner.get_room_info(_get_resolution(request, res))
    return {"room_count": len(rooms), "rooms": rooms}
