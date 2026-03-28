#!/usr/bin/env python3
"""网页版交互验证工具 —— 用真实 map_data 验证手动划分 & 手动合并

用法:
    python tools/verify_web.py                  # 默认端口 8080
    python tools/verify_web.py --port 9000      # 指定端口
    python tools/verify_web.py --data-dir gt_map_data  # 指定数据目录

浏览器打开 http://localhost:8080

依赖: fastapi, uvicorn, numpy, opencv-python, shapely (均在 requirements.txt 中)
"""

import argparse
import base64
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

# 项目根目录
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.core.config import load_config
from app.core.errors import RoomPartitionerError
from app.services.services import RoomService
from app.utils.coordinate import CoordinateTransformer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("verify_web")

# ==================== 全局状态 ====================

config = load_config()
DATA_ROOT: Path = ROOT / "gt_map_data"
room_service = RoomService(config)


# ==================== 数据加载 ====================

def load_case(case_dir: Path) -> dict:
    """加载单个 case 的 map_data"""
    map_img = cv2.imread(str(case_dir / "saved_map.png"), cv2.IMREAD_COLOR)
    if map_img is None:
        raise FileNotFoundError(f"无法读取: {case_dir / 'saved_map.png'}")

    with open(case_dir / "saved_map.json") as f:
        map_json = json.load(f)

    labels_json = None
    labels_path = case_dir / "labels.json"
    if labels_path.exists():
        with open(labels_path) as f:
            labels_json = json.load(f)

    mapinfo = {}
    mapinfo_path = case_dir / "mapinfo.json"
    if mapinfo_path.exists():
        with open(mapinfo_path) as f:
            mapinfo = json.load(f)

    markers_json = None
    markers_path = case_dir / "markers.json"
    if markers_path.exists():
        with open(markers_path) as f:
            markers_json = json.load(f)

    world_charge_pose = [0, 0, 0]
    if markers_json:
        for item in markers_json.get("data", []):
            if "CHARGE" in item.get("name", ""):
                world_charge_pose = item.get("geometry", [0, 0, 0])
                break

    return {
        "map_img": map_img,
        "resolution": map_json["resolution"],
        "origin": map_json["origin"],
        "labels_json": labels_json,
        "robot_model": mapinfo.get("robot_model", "s10"),
        "uuid": mapinfo.get("uuid", "unknown"),
        "markers_json": markers_json,
        "world_charge_pose": world_charge_pose,
    }


def rooms_to_pixel_coords(rooms_data: List[Dict], transformer: CoordinateTransformer) -> List[Dict]:
    """将房间 geometry (世界坐标) 转为像素坐标，供前端 Canvas 绘制"""
    result = []
    for room in rooms_data:
        geom = room.get("geometry", [])
        if len(geom) < 6:
            continue
        contour = transformer.world_to_contour(geom)
        # contour: (N,1,2) int32 → [[x,y], ...]
        pixels = contour.reshape(-1, 2).tolist()

        # 质心
        M = cv2.moments(contour)
        if M["m00"] > 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
        else:
            cx, cy = pixels[0] if pixels else [0, 0]

        result.append({
            "id": room["id"],
            "name": room.get("name", ""),
            "pixels": pixels,
            "centroid": [cx, cy],
            "colorType": room.get("colorType"),
        })
    return result


def map_img_to_base64(map_img: np.ndarray) -> str:
    """BGR 图片 → base64 PNG"""
    _, buf = cv2.imencode(".png", map_img)
    return base64.b64encode(buf).decode("utf-8")


# ==================== FastAPI App ====================

app = FastAPI(title="RoomPartitioner Verify Tool")


class DivisionRequest(BaseModel):
    case_id: str
    division_dict: Dict[str, Any]  # {"id": "ROOM_001", "A": [wx,wy], "B": [wx,wy]}


class MergeRequest(BaseModel):
    case_id: str
    room_ids: List[str]


class AutoPartitionRequest(BaseModel):
    case_id: str
    triton_url: str = ""


class RepartitionRequest(BaseModel):
    case_id: str
    triton_url: str = ""


@app.get("/", response_class=HTMLResponse)
def index():
    return HTML_TEMPLATE


@app.get("/api/cases")
def list_cases():
    if not DATA_ROOT.exists():
        return {"cases": []}
    cases = sorted(
        [d.name for d in DATA_ROOT.iterdir() if d.is_dir()],
        key=lambda x: int(x) if x.isdigit() else x,
    )
    return {"cases": cases}


@app.get("/api/cases/{case_id}")
def get_case(case_id: str):
    case_dir = DATA_ROOT / case_id
    if not case_dir.exists():
        raise HTTPException(status_code=404, detail=f"case {case_id} 不存在")

    map_data = load_case(case_dir)
    labels_json = map_data["labels_json"]

    if not labels_json or not labels_json.get("data"):
        return {
            "map_b64": map_img_to_base64(map_data["map_img"]),
            "width": map_data["map_img"].shape[1],
            "height": map_data["map_img"].shape[0],
            "resolution": map_data["resolution"],
            "origin": map_data["origin"][:2],
            "rooms": [],
        }

    rooms_data = [d for d in labels_json["data"] if "ROOM" in d.get("id", "")]
    transformer = CoordinateTransformer(
        map_data["resolution"], map_data["origin"],
        map_data["map_img"].shape[0],
    )
    rooms_px = rooms_to_pixel_coords(rooms_data, transformer)

    return {
        "map_b64": map_img_to_base64(map_data["map_img"]),
        "width": map_data["map_img"].shape[1],
        "height": map_data["map_img"].shape[0],
        "resolution": map_data["resolution"],
        "origin": map_data["origin"][:2],
        "rooms": rooms_px,
    }


def _execute_operation(case_id: str, operation: str,
                       triton_url: str = "",
                       division_dict: Optional[Dict] = None,
                       room_ids: Optional[List[str]] = None) -> Dict:
    """通过 RoomService.room_edit() 统一执行操作并返回新的房间数据"""
    case_dir = DATA_ROOT / case_id
    map_data = load_case(case_dir)

    # triton_url 覆盖时创建临时 service
    svc = room_service
    if triton_url:
        cfg = dict(config)
        cfg["triton_url"] = triton_url
        svc = RoomService(cfg)

    result = svc.room_edit(
        map_data,
        operation=operation,
        division_croods_dict=division_dict,
        room_merge_list=room_ids,
    )

    # 写回 labels.json
    labels_path = case_dir / "labels.json"
    with open(labels_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    logger.info(f"{operation} labels.json updated: {labels_path}")

    # 转换新结果的房间为像素坐标
    transformer = CoordinateTransformer(
        map_data["resolution"], map_data["origin"],
        map_data["map_img"].shape[0],
    )
    new_rooms_data = [d for d in result.get("data", []) if "ROOM" in d.get("id", "")]
    rooms_px = rooms_to_pixel_coords(new_rooms_data, transformer)

    return {"rooms": rooms_px}


@app.post("/api/division")
def do_division(req: DivisionRequest):
    try:
        result = _execute_operation(req.case_id, "division",
                                    division_dict=req.division_dict)
        return {"ok": True, "rooms": result["rooms"]}
    except RoomPartitionerError as e:
        return {"ok": False, "error": f"[{e.code}] {e}"}
    except Exception as e:
        logger.error(f"division 异常: {e}", exc_info=True)
        return {"ok": False, "error": str(e)}


@app.post("/api/merge")
def do_merge(req: MergeRequest):
    try:
        result = _execute_operation(req.case_id, "merge",
                                    room_ids=req.room_ids)
        return {"ok": True, "rooms": result["rooms"]}
    except RoomPartitionerError as e:
        return {"ok": False, "error": f"[{e.code}] {e}"}
    except Exception as e:
        logger.error(f"merge 异常: {e}", exc_info=True)
        return {"ok": False, "error": str(e)}


@app.post("/api/auto_partition")
def do_auto_partition(req: AutoPartitionRequest):
    try:
        result = _execute_operation(req.case_id, "split",
                                    triton_url=req.triton_url)
        return {"ok": True, "rooms": result["rooms"]}
    except RoomPartitionerError as e:
        return {"ok": False, "error": f"[{e.code}] {e}"}
    except Exception as e:
        logger.error(f"auto_partition 异常: {e}", exc_info=True)
        return {"ok": False, "error": str(e)}


@app.post("/api/repartition")
def do_repartition(req: RepartitionRequest):
    try:
        result = _execute_operation(req.case_id, "repartition",
                                    triton_url=req.triton_url)
        return {"ok": True, "rooms": result["rooms"]}
    except RoomPartitionerError as e:
        return {"ok": False, "error": f"[{e.code}] {e}"}
    except Exception as e:
        logger.error(f"repartition 异常: {e}", exc_info=True)
        return {"ok": False, "error": str(e)}


# ==================== HTML 前端 ====================

HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="UTF-8">
<title>RoomPartitioner Verify Tool</title>
<style>
* { margin: 0; padding: 0; box-sizing: border-box; }
body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; background: #1a1a2e; color: #eee; }

.header { background: #16213e; padding: 12px 24px; display: flex; align-items: center; gap: 16px; flex-wrap: wrap; }
.header h1 { font-size: 18px; background: #e94560; padding: 4px 12px; border-radius: 4px; color: white; }
.header select, .header button { padding: 6px 12px; border-radius: 4px; border: 1px solid #444; background: #0f3460; color: #eee; cursor: pointer; font-size: 14px; }
.header button:hover { background: #e94560; }
.header .mode-btn.active { background: #e94560; border-color: #e94560; }
.zoom-info { color: #aaa; font-size: 12px; margin-left: auto; }

.main { display: flex; height: calc(100vh - 50px); }

.canvas-panel { flex: 1; position: relative; overflow: hidden; background: #0f0f23; }
.canvas-panel canvas { position: absolute; top: 0; left: 0; }

.sidebar { width: 320px; background: #16213e; padding: 16px; overflow-y: auto; border-left: 1px solid #333; }
.sidebar h3 { color: #e94560; margin-bottom: 8px; font-size: 14px; }

.room-list { list-style: none; margin-bottom: 16px; }
.room-list li { padding: 6px 10px; margin: 3px 0; border-radius: 4px; cursor: pointer; font-size: 13px; display: flex; align-items: center; gap: 8px; }
.room-list li:hover { background: rgba(255,255,255,0.1); }
.room-list li.selected { background: rgba(233,69,96,0.3); border: 1px solid #e94560; }
.color-dot { width: 12px; height: 12px; border-radius: 50%; display: inline-block; flex-shrink: 0; }

.info-box { background: #0f3460; padding: 10px; border-radius: 6px; margin-bottom: 12px; font-size: 13px; line-height: 1.6; }
.info-box .label { color: #aaa; }

.log { background: #0a0a1a; padding: 10px; border-radius: 6px; font-size: 12px; max-height: 200px; overflow-y: auto; font-family: monospace; line-height: 1.5; }
.log .err { color: #e94560; }
.log .ok { color: #4ecca3; }

.btn-execute { width: 100%; padding: 10px; background: #e94560; color: white; border: none; border-radius: 6px; font-size: 14px; cursor: pointer; margin-top: 8px; }
.btn-execute:hover { background: #c73854; }
.btn-execute:disabled { background: #555; cursor: not-allowed; }

.btn-reset { width: 100%; padding: 8px; background: #0f3460; color: #eee; border: 1px solid #444; border-radius: 6px; font-size: 13px; cursor: pointer; margin-top: 6px; }

.coord-tip { position: absolute; bottom: 8px; left: 8px; background: rgba(0,0,0,0.7); padding: 4px 8px; border-radius: 4px; font-size: 11px; color: #ccc; pointer-events: none; font-family: monospace; }
</style>
</head>
<body>

<div class="header">
    <h1>Verify</h1>
    <label>Case:</label>
    <select id="caseSelect"><option>Loading...</option></select>
    <button class="mode-btn active" id="btnDivision" onclick="setMode('division')">Division</button>
    <button class="mode-btn" id="btnMerge" onclick="setMode('merge')">Merge</button>
    <button onclick="fitToView()">Fit</button>
    <span style="color:#aaa;font-size:12px;margin-left:8px;">Triton:</span>
    <input id="tritonUrl" type="text" placeholder="ip:port" value=""
           style="width:160px;padding:4px 8px;border-radius:4px;border:1px solid #444;background:#0f3460;color:#eee;font-size:13px;">
    <button onclick="executeAutoPartition()" style="background:#4ecca3;color:#000;font-weight:bold;padding:6px 12px;border-radius:4px;border:none;cursor:pointer;">Auto Partition</button>
    <button onclick="executeRepartition()" style="background:#f0a500;color:#000;font-weight:bold;padding:6px 12px;border-radius:4px;border:none;cursor:pointer;">Repartition</button>
    <span class="zoom-info" id="zoomInfo">100% | Scroll=Zoom, MiddleDrag=Pan</span>
</div>

<div class="main">
    <div class="canvas-panel" id="canvasPanel">
        <canvas id="mapCanvas"></canvas>
        <div class="coord-tip" id="coordTip">-</div>
    </div>
    <div class="sidebar">
        <div class="info-box">
            <div><span class="label">Case:</span> <span id="infoCase">-</span></div>
            <div><span class="label">Size:</span> <span id="infoSize">-</span></div>
            <div><span class="label">Rooms:</span> <span id="infoRooms">-</span></div>
            <div><span class="label">Mode:</span> <span id="infoMode">division</span></div>
        </div>

        <h3>Rooms</h3>
        <ul class="room-list" id="roomList"></ul>

        <div id="divisionControls">
            <h3>Division</h3>
            <div class="info-box" id="divisionInfo">
                Click two points on Canvas for split line (A, B)
            </div>
            <label style="font-size:13px;">Target Room:</label>
            <select id="targetRoom" style="width:100%;padding:6px;margin:6px 0;background:#0f3460;color:#eee;border:1px solid #444;border-radius:4px;"></select>
            <button class="btn-execute" id="btnDoDivision" onclick="executeDivision()" disabled>Execute Division</button>
        </div>

        <div id="mergeControls" style="display:none;">
            <h3>Merge</h3>
            <div class="info-box" id="mergeInfo">
                Click rooms in the list to select (min 2)
            </div>
            <button class="btn-execute" id="btnDoMerge" onclick="executeMerge()" disabled>Execute Merge</button>
        </div>

        <button class="btn-reset" onclick="resetState()">Reset</button>

        <h3 style="margin-top:16px;">Log</h3>
        <div class="log" id="logBox"></div>
    </div>
</div>

<script>
// ==================== Constants ====================
// 五色地图: colorType 0-4
const ROOM_COLORS = [
    [84, 166, 217],   // 0: 蓝
    [102, 199, 102],  // 1: 绿
    [230, 179, 51],   // 2: 橙
    [217, 84, 84],    // 3: 红
    [179, 102, 217],  // 4: 紫
];

// ==================== State ====================
let currentMode = 'division';
let caseData = null;
let currentCaseId = null;
let mapImage = null;

// Zoom & Pan state
let viewScale = 1.0;    // current zoom level
let viewOffX = 0;        // canvas-panel-space offset of image origin
let viewOffY = 0;
let isPanning = false;
let panStartX = 0, panStartY = 0;
let panStartOffX = 0, panStartOffY = 0;

// Division
let splitPoints = [];    // [{px, py, wx, wy}, ...]

// Merge
let selectedRoomIds = new Set();
let beforeRooms = null;

// ==================== Canvas ====================
const canvas = document.getElementById('mapCanvas');
const ctx = canvas.getContext('2d');
const panel = document.getElementById('canvasPanel');

function pixelToWorld(px, py) {
    if (!caseData) return [0, 0];
    const wx = px * caseData.resolution + caseData.origin[0] + 0.025;
    const wy = (caseData.height - py - 1) * caseData.resolution + caseData.origin[1] - 0.025;
    return [Math.round(wx * 1000) / 1000, Math.round(wy * 1000) / 1000];
}

/** Convert screen (mouse) coords relative to panel → original image pixel coords */
function screenToImagePx(clientX, clientY) {
    const rect = panel.getBoundingClientRect();
    const sx = clientX - rect.left;
    const sy = clientY - rect.top;
    const imgX = (sx - viewOffX) / viewScale;
    const imgY = (sy - viewOffY) / viewScale;
    return [Math.round(imgX), Math.round(imgY)];
}

/** Fit image centered in panel with padding */
function fitToView() {
    if (!caseData) return;
    const pw = panel.clientWidth;
    const ph = panel.clientHeight;
    const pad = 20;
    viewScale = Math.min((pw - pad * 2) / caseData.width, (ph - pad * 2) / caseData.height);
    viewOffX = (pw - caseData.width * viewScale) / 2;
    viewOffY = (ph - caseData.height * viewScale) / 2;
    updateZoomInfo();
    drawMap();
}

function updateZoomInfo() {
    document.getElementById('zoomInfo').textContent =
        `${Math.round(viewScale * 100)}% | Scroll=Zoom, MiddleDrag=Pan`;
}

function drawMap() {
    if (!mapImage || !caseData) return;

    // Size canvas to fill panel
    canvas.width = panel.clientWidth;
    canvas.height = panel.clientHeight;

    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.save();

    // Apply view transform: translate then scale
    ctx.translate(viewOffX, viewOffY);
    ctx.scale(viewScale, viewScale);

    // Draw base map image at original size
    ctx.drawImage(mapImage, 0, 0);

    // Draw rooms (all coords are in original image pixel space)
    drawRooms(caseData.rooms);

    // Draw split line overlay
    drawSplitLine();

    ctx.restore();
}

function drawRooms(rooms) {
    // Compute inverse scale so that labels/strokes stay readable regardless of zoom
    const invS = 1 / viewScale;

    rooms.forEach((room, i) => {
        const ct = (room.colorType != null && room.colorType >= 0) ? room.colorType % ROOM_COLORS.length : 0;
        const color = ROOM_COLORS[ct];
        const pixels = room.pixels;
        if (pixels.length < 3) return;

        // Filled polygon
        ctx.beginPath();
        ctx.moveTo(pixels[0][0], pixels[0][1]);
        for (let j = 1; j < pixels.length; j++) {
            ctx.lineTo(pixels[j][0], pixels[j][1]);
        }
        ctx.closePath();
        ctx.fillStyle = `rgba(${color[0]},${color[1]},${color[2]},0.35)`;
        ctx.fill();

        // Border
        const isSelected = selectedRoomIds.has(room.id);
        ctx.strokeStyle = isSelected ? '#e94560' : `rgb(${color[0]},${color[1]},${color[2]})`;
        ctx.lineWidth = (isSelected ? 3 : 1.5) * invS;
        ctx.stroke();

        // Label at centroid
        const [cx, cy] = room.centroid;
        const fontSize = Math.max(9, Math.min(13, 11 * invS));
        ctx.font = `bold ${fontSize}px sans-serif`;
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';

        const lines = [room.id, `(${room.name})`];
        const lineH = fontSize * 1.3;
        const maxW = Math.max(...lines.map(l => ctx.measureText(l).width));
        const boxW = maxW + 8 * invS;
        const boxH = lines.length * lineH + 4 * invS;

        ctx.fillStyle = `rgba(${color[0]},${color[1]},${color[2]},0.85)`;
        roundRect(ctx, cx - boxW / 2, cy - boxH / 2, boxW, boxH, 3 * invS);
        ctx.fill();

        ctx.fillStyle = '#fff';
        lines.forEach((line, li) => {
            ctx.fillText(line, cx, cy + (li - (lines.length - 1) / 2) * lineH);
        });
    });
}

function roundRect(ctx, x, y, w, h, r) {
    ctx.beginPath();
    ctx.moveTo(x + r, y);
    ctx.lineTo(x + w - r, y);
    ctx.quadraticCurveTo(x + w, y, x + w, y + r);
    ctx.lineTo(x + w, y + h - r);
    ctx.quadraticCurveTo(x + w, y + h, x + w - r, y + h);
    ctx.lineTo(x + r, y + h);
    ctx.quadraticCurveTo(x, y + h, x, y + h - r);
    ctx.lineTo(x, y + r);
    ctx.quadraticCurveTo(x, y, x + r, y);
    ctx.closePath();
}

function drawSplitLine() {
    if (splitPoints.length === 0) return;
    const invS = 1 / viewScale;
    drawCross(splitPoints[0].px, splitPoints[0].py, '#e94560', invS);
    if (splitPoints.length === 2) {
        drawCross(splitPoints[1].px, splitPoints[1].py, '#e94560', invS);
        ctx.beginPath();
        ctx.moveTo(splitPoints[0].px, splitPoints[0].py);
        ctx.lineTo(splitPoints[1].px, splitPoints[1].py);
        ctx.strokeStyle = '#e94560';
        ctx.lineWidth = 2 * invS;
        ctx.setLineDash([6 * invS, 4 * invS]);
        ctx.stroke();
        ctx.setLineDash([]);
    }
}

function drawCross(x, y, color, invS) {
    const s = 8 * invS;
    ctx.strokeStyle = color;
    ctx.lineWidth = 2 * invS;
    ctx.beginPath();
    ctx.moveTo(x - s, y - s); ctx.lineTo(x + s, y + s);
    ctx.moveTo(x + s, y - s); ctx.lineTo(x - s, y + s);
    ctx.stroke();
    ctx.beginPath();
    ctx.arc(x, y, s + 2 * invS, 0, Math.PI * 2);
    ctx.stroke();
}

// ==================== Zoom (scroll wheel) ====================

panel.addEventListener('wheel', (e) => {
    e.preventDefault();
    if (!caseData) return;
    const rect = panel.getBoundingClientRect();
    const mx = e.clientX - rect.left;
    const my = e.clientY - rect.top;

    const zoomFactor = e.deltaY < 0 ? 1.15 : 1 / 1.15;
    const newScale = Math.min(20, Math.max(0.1, viewScale * zoomFactor));

    // Zoom towards mouse position
    viewOffX = mx - (mx - viewOffX) * (newScale / viewScale);
    viewOffY = my - (my - viewOffY) * (newScale / viewScale);
    viewScale = newScale;

    updateZoomInfo();
    drawMap();
}, {passive: false});

// ==================== Pan (middle-button drag OR right-button drag) ====================

panel.addEventListener('mousedown', (e) => {
    // Middle button (1) or right button (2) for panning
    if (e.button === 1 || e.button === 2) {
        e.preventDefault();
        isPanning = true;
        panStartX = e.clientX;
        panStartY = e.clientY;
        panStartOffX = viewOffX;
        panStartOffY = viewOffY;
        canvas.style.cursor = 'grabbing';
    }
});

window.addEventListener('mousemove', (e) => {
    if (isPanning) {
        viewOffX = panStartOffX + (e.clientX - panStartX);
        viewOffY = panStartOffY + (e.clientY - panStartY);
        drawMap();
    }
    // Update coordinate tooltip
    if (caseData) {
        const [imgX, imgY] = screenToImagePx(e.clientX, e.clientY);
        if (imgX >= 0 && imgX < caseData.width && imgY >= 0 && imgY < caseData.height) {
            const [wx, wy] = pixelToWorld(imgX, imgY);
            document.getElementById('coordTip').textContent =
                `pixel(${imgX}, ${imgY})  world(${wx}, ${wy})`;
        } else {
            document.getElementById('coordTip').textContent = '-';
        }
    }
});

window.addEventListener('mouseup', (e) => {
    if (isPanning) {
        isPanning = false;
        canvas.style.cursor = currentMode === 'division' ? 'crosshair' : 'default';
    }
});

// Disable context menu on canvas panel so right-click pan works
panel.addEventListener('contextmenu', (e) => e.preventDefault());

// ==================== Click → image pixel coords ====================

canvas.addEventListener('click', (e) => {
    if (!caseData || isPanning) return;
    if (e.button !== 0) return;  // left click only
    const [px, py] = screenToImagePx(e.clientX, e.clientY);

    // Bounds check
    if (px < 0 || px >= caseData.width || py < 0 || py >= caseData.height) return;

    if (currentMode === 'division') {
        handleDivisionClick(px, py);
    }
});

function handleDivisionClick(px, py) {
    if (splitPoints.length >= 2) return;
    const [wx, wy] = pixelToWorld(px, py);
    splitPoints.push({px, py, wx, wy});
    const label = splitPoints.length === 1 ? 'A' : 'B';
    addLog(`${label} = pixel(${px},${py}) → world(${wx},${wy})`);
    drawMap();

    if (splitPoints.length === 2) {
        document.getElementById('btnDoDivision').disabled = false;
        document.getElementById('divisionInfo').textContent =
            `A=(${splitPoints[0].wx}, ${splitPoints[0].wy})  B=(${splitPoints[1].wx}, ${splitPoints[1].wy})`;
    } else {
        document.getElementById('divisionInfo').textContent = 'Point A selected, click point B...';
    }
}

// ==================== Mode switch ====================

function setMode(mode) {
    currentMode = mode;
    document.getElementById('btnDivision').classList.toggle('active', mode === 'division');
    document.getElementById('btnMerge').classList.toggle('active', mode === 'merge');
    document.getElementById('divisionControls').style.display = mode === 'division' ? 'block' : 'none';
    document.getElementById('mergeControls').style.display = mode === 'merge' ? 'block' : 'none';
    document.getElementById('infoMode').textContent = mode;
    canvas.style.cursor = mode === 'division' ? 'crosshair' : 'default';
    resetState();
}

function resetState() {
    splitPoints = [];
    selectedRoomIds.clear();
    beforeRooms = null;
    document.getElementById('btnDoDivision').disabled = true;
    document.getElementById('btnDoMerge').disabled = true;
    document.getElementById('divisionInfo').textContent = 'Click two points on Canvas for split line (A, B)';
    document.getElementById('mergeInfo').textContent = 'Click rooms in the list to select (min 2)';
    updateRoomList();
    drawMap();
}

// ==================== Room list ====================

function updateRoomList() {
    const ul = document.getElementById('roomList');
    const sel = document.getElementById('targetRoom');
    ul.innerHTML = '';
    sel.innerHTML = '';

    if (!caseData || !caseData.rooms) return;

    caseData.rooms.forEach((room, i) => {
        const ct = (room.colorType != null && room.colorType >= 0) ? room.colorType % ROOM_COLORS.length : 0;
        const color = ROOM_COLORS[ct];

        const li = document.createElement('li');
        li.innerHTML = `<span class="color-dot" style="background:rgb(${color.join(',')})"></span>${room.id} (${room.name})`;
        li.dataset.roomId = room.id;
        if (selectedRoomIds.has(room.id)) li.classList.add('selected');
        li.onclick = () => toggleRoomSelect(room.id);
        ul.appendChild(li);

        const opt = document.createElement('option');
        opt.value = room.id;
        opt.textContent = `${room.id} (${room.name})`;
        sel.appendChild(opt);
    });
}

function toggleRoomSelect(roomId) {
    if (currentMode !== 'merge') return;
    if (selectedRoomIds.has(roomId)) {
        selectedRoomIds.delete(roomId);
    } else {
        selectedRoomIds.add(roomId);
    }
    document.getElementById('btnDoMerge').disabled = selectedRoomIds.size < 2;
    document.getElementById('mergeInfo').textContent =
        selectedRoomIds.size > 0
            ? `Selected: ${[...selectedRoomIds].join(', ')}`
            : 'Click rooms in the list to select (min 2)';
    updateRoomList();
    drawMap();
}

// ==================== Execute operations ====================

async function executeDivision() {
    if (splitPoints.length < 2 || !caseData) return;
    const targetRoom = document.getElementById('targetRoom').value;
    if (!targetRoom) { addLog('Please select a target room', true); return; }

    const payload = {
        case_id: currentCaseId,
        division_dict: {
            id: targetRoom,
            A: [splitPoints[0].wx, splitPoints[0].wy],
            B: [splitPoints[1].wx, splitPoints[1].wy],
        }
    };

    addLog(`Division: ${targetRoom}, A=[${splitPoints[0].wx},${splitPoints[0].wy}], B=[${splitPoints[1].wx},${splitPoints[1].wy}]`);
    beforeRooms = caseData.rooms;

    try {
        const resp = await fetch('/api/division', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify(payload),
        });
        const data = await resp.json();
        if (data.ok) {
            addLog(`OK: ${beforeRooms.length} → ${data.rooms.length} rooms`, false, true);
            caseData.rooms = data.rooms;
            splitPoints = [];
            updateRoomList();
            drawMap();
            document.getElementById('infoRooms').textContent = data.rooms.length;
        } else {
            addLog(`FAIL: ${data.error}`, true);
        }
    } catch (e) {
        addLog(`Error: ${e.message}`, true);
    }
    document.getElementById('btnDoDivision').disabled = true;
}

async function executeMerge() {
    if (selectedRoomIds.size < 2 || !caseData) return;
    const roomIds = [...selectedRoomIds];

    addLog(`Merge: ${roomIds.join(', ')}`);
    beforeRooms = caseData.rooms;

    try {
        const resp = await fetch('/api/merge', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({ case_id: currentCaseId, room_ids: roomIds }),
        });
        const data = await resp.json();
        if (data.ok) {
            addLog(`OK: ${beforeRooms.length} → ${data.rooms.length} rooms`, false, true);
            caseData.rooms = data.rooms;
            selectedRoomIds.clear();
            updateRoomList();
            drawMap();
            document.getElementById('infoRooms').textContent = data.rooms.length;
        } else {
            addLog(`FAIL: ${data.error}`, true);
        }
    } catch (e) {
        addLog(`Error: ${e.message}`, true);
    }
    document.getElementById('btnDoMerge').disabled = true;
}

async function executeAutoPartition() {
    if (!caseData || !currentCaseId) { addLog('No case loaded', true); return; }
    const tritonUrl = document.getElementById('tritonUrl').value.trim();
    addLog(`Auto Partition: case=${currentCaseId}, triton=${tritonUrl || '(fallback)'}`);

    try {
        const resp = await fetch('/api/auto_partition', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({ case_id: currentCaseId, triton_url: tritonUrl }),
        });
        const data = await resp.json();
        if (data.ok) {
            addLog(`Auto Partition OK: ${data.rooms.length} rooms`, false, true);
            caseData.rooms = data.rooms;
            updateRoomList();
            drawMap();
            document.getElementById('infoRooms').textContent = data.rooms.length;
        } else {
            addLog(`Auto Partition FAIL: ${data.error}`, true);
        }
    } catch (e) {
        addLog(`Auto Partition Error: ${e.message}`, true);
    }
}

async function executeRepartition() {
    if (!caseData || !currentCaseId) { addLog('No case loaded', true); return; }
    const tritonUrl = document.getElementById('tritonUrl').value.trim();
    addLog(`Repartition: case=${currentCaseId}, triton=${tritonUrl || '(fallback)'}`);

    try {
        const resp = await fetch('/api/repartition', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({ case_id: currentCaseId, triton_url: tritonUrl }),
        });
        const data = await resp.json();
        if (data.ok) {
            addLog(`Repartition OK: ${data.rooms.length} rooms`, false, true);
            caseData.rooms = data.rooms;
            updateRoomList();
            drawMap();
            document.getElementById('infoRooms').textContent = data.rooms.length;
        } else {
            addLog(`Repartition FAIL: ${data.error}`, true);
        }
    } catch (e) {
        addLog(`Repartition Error: ${e.message}`, true);
    }
}

// ==================== Case loading ====================

async function loadCases() {
    const resp = await fetch('/api/cases');
    const data = await resp.json();
    const sel = document.getElementById('caseSelect');
    sel.innerHTML = '';
    data.cases.forEach(c => {
        const opt = document.createElement('option');
        opt.value = c;
        opt.textContent = `Case ${c}`;
        sel.appendChild(opt);
    });
    if (data.cases.length > 0) {
        loadCase(data.cases[0]);
    }
}

async function loadCase(caseId) {
    currentCaseId = caseId;
    addLog(`Loading case ${caseId}...`);
    document.getElementById('infoCase').textContent = caseId;

    try {
        const resp = await fetch(`/api/cases/${caseId}`);
        caseData = await resp.json();

        mapImage = new Image();
        mapImage.onload = () => {
            fitToView();
            addLog(`Loaded: ${caseData.width}x${caseData.height}, ${caseData.rooms.length} rooms`, false, true);
        };
        mapImage.src = 'data:image/png;base64,' + caseData.map_b64;

        document.getElementById('infoSize').textContent = `${caseData.width}x${caseData.height}`;
        document.getElementById('infoRooms').textContent = caseData.rooms.length;

        resetState();
    } catch (e) {
        addLog(`Load failed: ${e.message}`, true);
    }
}

document.getElementById('caseSelect').addEventListener('change', (e) => {
    loadCase(e.target.value);
});

// Handle window resize
window.addEventListener('resize', () => { if (caseData) fitToView(); });

// ==================== Log ====================

function addLog(msg, isErr = false, isOk = false) {
    const box = document.getElementById('logBox');
    const cls = isErr ? 'err' : (isOk ? 'ok' : '');
    const time = new Date().toLocaleTimeString();
    box.innerHTML += `<div class="${cls}">[${time}] ${msg}</div>`;
    box.scrollTop = box.scrollHeight;
}

// ==================== Init ====================
loadCases();
</script>
</body>
</html>
"""


# ==================== 启动入口 ====================

def main():
    parser = argparse.ArgumentParser(description="网页版验证工具")
    parser.add_argument("--port", type=int, default=8080, help="端口号")
    parser.add_argument("--data-dir", type=str, default="gt_map_data", help="数据根目录")
    parser.add_argument("--triton-url", type=str, default="", help="Triton gRPC 地址 (如 192.168.2.55:8001)")
    args = parser.parse_args()

    global DATA_ROOT, config
    DATA_ROOT = ROOT / args.data_dir
    if args.triton_url:
        config["triton_url"] = args.triton_url

    logger.info(f"数据目录: {DATA_ROOT}")
    logger.info(f"启动地址: http://localhost:{args.port}")

    uvicorn.run(app, host="0.0.0.0", port=args.port, log_level="info")


if __name__ == "__main__":
    main()
