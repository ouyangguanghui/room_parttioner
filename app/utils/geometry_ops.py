"""几何纯函数工具（世界坐标/房间元数据级，Shapely-only）。"""

import math
from typing import Any, Dict, List, Sequence, Tuple

from shapely.geometry import LineString, Polygon  # type: ignore
from shapely.ops import split as shp_split  # type: ignore



def get_room_index_by_id(rooms_data: List[Dict[str, Any]], room_id: str) -> int:
    """根据房间 ID 获取房间索引。"""
    for i, room in enumerate(rooms_data):
        if room["id"] == room_id:
            return i
    return -1


def find_room_index_by_id(rooms_data: List[Dict[str, Any]], room_id: str) -> int:
    """兼容旧命名：等价于 get_room_index_by_id。"""
    return get_room_index_by_id(rooms_data, room_id)

def split_labels_data(labels: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """分割房间数据和标记点数据。"""
    rooms_data = [d for d in labels['data'] if 'ROOM' in d.get('id', '')]
    landmarks_data = [d for d in labels['data'] if 'PLATFORM_LANDMARK' in d.get('id', '')]
    return rooms_data, landmarks_data


def next_room_id(rooms_data: List[Dict[str, Any]]) -> str:
    """生成下一个 ROOM_xxx ID（按现有最大序号 + 1）。"""
    max_idx = 0
    for room in rooms_data:
        rid = room.get("id", "")
        if not isinstance(rid, str) or not rid.startswith("ROOM_"):
            continue
        try:
            max_idx = max(max_idx, int(rid.split("_")[-1]))
        except ValueError:
            continue
    if max_idx >= 999:
        raise ValueError("房间 ID 已达到上限 ROOM_999，无法继续分配")
    return f"ROOM_{max_idx + 1:03d}"


def next_room_name(rooms_data: List[Dict[str, Any]]) -> str:
    """分配下一个可用房间名：A~Z，超出后 A1~Z1、A2~Z2..."""
    used_names = {r.get("name") for r in rooms_data if isinstance(r.get("name"), str)}
    letters = [chr(ord("A") + i) for i in range(26)]

    suffix = 0
    while True:
        for letter in letters:
            candidate = letter if suffix == 0 else f"{letter}{suffix}"
            if candidate not in used_names:
                return candidate
        suffix += 1


def flatten_geometry(poly_pts: List[Tuple[float, float]]) -> List[float]:
    """多边形点列表 -> flat geometry [x0,y0,x1,y1,...,x0,y0]。"""
    geom: List[float] = []
    for pt in poly_pts:
        geom.extend([pt[0], pt[1]])
    geom.extend([poly_pts[0][0], poly_pts[0][1]])
    return geom


def _dedupe_closed_points(geometry: Sequence[float]) -> List[Tuple[float, float]]:
    """flat geometry -> 点列表，并去掉重复闭合点。"""
    pts = [(float(geometry[i]), float(geometry[i + 1])) for i in range(0, len(geometry), 2)]
    if len(pts) >= 2 and pts[0] == pts[-1]:
        pts = pts[:-1]
    return pts


def _extract_points_from_geom(geom) -> List[Tuple[float, float]]:
    """从 shapely 几何对象中提取点坐标。"""
    if geom is None:
        return []
    gt = geom.geom_type
    if gt == "Point":
        return [(float(geom.x), float(geom.y))]
    if gt == "MultiPoint":
        return [(float(g.x), float(g.y)) for g in geom.geoms]
    if gt == "LineString":
        coords = list(geom.coords)
        if not coords:
            return []
        return [(float(coords[0][0]), float(coords[0][1])), (float(coords[-1][0]), float(coords[-1][1]))]
    if gt == "GeometryCollection":
        pts: List[Tuple[float, float]] = []
        for g in geom.geoms:
            pts.extend(_extract_points_from_geom(g))
        return pts
    return []


def _build_extended_cut_line(
    A: Tuple[float, float],
    B: Tuple[float, float],
    contour_pts: Sequence[Tuple[float, float]],
):
    """构造穿过 A-B 方向的长切分线（模拟原版“无限直线”语义）。"""
    ax, ay = A
    bx, by = B
    vx, vy = (bx - ax), (by - ay)
    norm = math.hypot(vx, vy)
    if norm < 1e-8:
        return None
    vx, vy = vx / norm, vy / norm
    xs = [p[0] for p in contour_pts]
    ys = [p[1] for p in contour_pts]
    diag = math.hypot(max(xs) - min(xs), max(ys) - min(ys))
    extent = max(diag * 4.0, 10.0)
    s = (ax - vx * extent, ay - vy * extent)
    e = (bx + vx * extent, by + vy * extent)
    return LineString([s, e])


def _find_split_points_shapely(
    A: Tuple[float, float],
    B: Tuple[float, float],
    geometry: List[float],
):
    """使用 shapely 进行稳健切分。"""
    contour_pts = _dedupe_closed_points(geometry)
    if len(contour_pts) < 3:
        return False, "invalid polygon"

    try:
        poly = Polygon(contour_pts)
        if not poly.is_valid:
            poly = poly.buffer(0)
        if poly.is_empty:
            return False, "empty polygon"

        cut_line = _build_extended_cut_line(A, B, contour_pts)
        if cut_line is None:
            return False, "invalid cut line"

        # 交点
        ip_geom = poly.boundary.intersection(cut_line)
        intersections = _extract_points_from_geom(ip_geom)
        if len(intersections) < 2:
            return False, "交点不足两个"

        # 与原版保持一致：多交点时取离 A/B 最近的两点
        ia = min(range(len(intersections)), key=lambda i: math.dist(A, intersections[i]))
        ib = min(range(len(intersections)), key=lambda i: math.dist(B, intersections[i]))
        if ia == ib and len(intersections) > 1:
            dists = sorted(
                ((math.dist(B, p), i) for i, p in enumerate(intersections) if i != ia),
                key=lambda x: x[0],
            )
            ib = dists[0][1]
        picked = [intersections[ia], intersections[ib]]

        # 切分
        parts = shp_split(poly, cut_line)
        polys = [g for g in parts.geoms if g.geom_type == "Polygon" and not g.is_empty]
        if len(polys) < 2:
            return False, "split failed"
        polys.sort(key=lambda p: p.area, reverse=True)
        p1, p2 = polys[0], polys[1]

        # 去掉闭合尾点
        poly_a = [(float(x), float(y)) for x, y in list(p1.exterior.coords)[:-1]]
        poly_b = [(float(x), float(y)) for x, y in list(p2.exterior.coords)[:-1]]

        return True, (poly_a, poly_b, picked)
    except Exception:
        return False, "shapely exception"



def find_split_points(
    A: Tuple[float, float],
    B: Tuple[float, float],
    geometry: List[float],
):
    """
    在 world geometry 上找分割线交点并拆分。

    Returns:
        (ok, (poly_a, poly_b, intersections)) 或 (False, message)
    """
    return _find_split_points_shapely(A, B, geometry)