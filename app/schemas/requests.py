"""Pydantic 请求/响应模型"""

from typing import List
from pydantic import BaseModel


class LineRequest(BaseModel):
    pt1: List[int]  # [x, y]
    pt2: List[int]  # [x, y]


class PolylineRequest(BaseModel):
    points: List[List[int]]  # [[x,y], ...]


class PolygonRequest(BaseModel):
    polygon: List[List[int]]
    room_id: int = -1


class MergeRequest(BaseModel):
    room_ids: List[int]


class MergePointRequest(BaseModel):
    pt1: List[int]
    pt2: List[int]
