#!/usr/bin/env python3
"""真实推理验证脚本 —— 加载 gt_map_data → 预处理 → Triton 推理 → 后处理 → 可视化

用法:
    python tools/verify_infer.py --case 1 --triton-url 192.168.2.55:8001
    python tools/verify_infer.py --case 1 --triton-url 192.168.2.55:8001 --save result.png
    python tools/verify_infer.py --case 1 --triton-url 192.168.2.55:8001 --all-cases

依赖: numpy, opencv-python, tritonclient[grpc]
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, Any

import cv2
import numpy as np

# 项目根目录
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.core.config import load_config
from app.pipeline.preprocessor import Preprocessor
from app.pipeline.inferencer import Inferencer
from app.pipeline.postprocessor import Postprocessor
from app.services.auto_partition import AutoPartitioner

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("verify_infer")


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


# ==================== 可视化 ====================

def label_to_color(label_map: np.ndarray) -> np.ndarray:
    """标签图 → 彩色可视化"""
    color_map = np.zeros((*label_map.shape, 3), dtype=np.uint8)
    for lid in range(1, label_map.max() + 1):
        color = np.random.RandomState(lid * 7).randint(60, 230, 3).tolist()
        color_map[label_map == lid] = color
    return color_map


def draw_obb_overlay(img: np.ndarray, obb_list: list) -> np.ndarray:
    """在图上绘制 OBB 检测框"""
    vis = img.copy() if img.ndim == 3 else cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    for i, obb in enumerate(obb_list):
        pts = np.array(obb, dtype=np.int32)
        cv2.polylines(vis, [pts], True, (0, 0, 255), 2)
        # 标号
        cx = int(np.mean(pts[:, 0]))
        cy = int(np.mean(pts[:, 1]))
        cv2.putText(vis, str(i), (cx - 5, cy + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    return vis


def draw_threshold_overlay(img: np.ndarray, threshold_list: list) -> np.ndarray:
    """在图上绘制 threshold 分割线"""
    vis = img.copy() if img.ndim == 3 else cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    for seg in threshold_list:
        p1 = (int(seg[0][0]), int(seg[0][1]))
        p2 = (int(seg[1][0]), int(seg[1][1]))
        cv2.line(vis, p1, p2, (0, 0, 255), 2)
    return vis


# ==================== 主流程 ====================

def run_inference(case_dir: Path, config: Dict[str, Any],
                  save_path: str = None) -> Dict[str, Any]:
    """对单个 case 执行完整推理流程"""
    logger.info(f"{'='*50}")
    logger.info(f"Case: {case_dir.name}")
    logger.info(f"{'='*50}")

    # 1. 加载数据
    map_data = load_case(case_dir)
    map_img = map_data["map_img"]
    gray = cv2.cvtColor(map_img, cv2.COLOR_BGR2GRAY) if map_img.ndim == 3 else map_img

    logger.info(f"地图尺寸: {gray.shape[1]}x{gray.shape[0]}, "
                f"resolution={map_data['resolution']}")

    # 2. 预处理
    preprocessor = Preprocessor(config)
    meta = preprocessor.process(gray)
    logger.info(f"预处理完成: map_data shape={meta['map_data'].shape}, "
                f"input_data shape={meta['input_data'].shape}")

    # 3. 准备张量
    partitioner = AutoPartitioner(config)
    tensor = partitioner._prepare_tensor(meta["input_data"], meta)
    logger.info(f"张量: shape={tensor.shape}, dtype={tensor.dtype}, "
                f"range=[{tensor.min():.3f}, {tensor.max():.3f}]")
    logger.info(f"  tensor_scale={meta.get('tensor_scale')}, "
                f"tensor_pad={meta.get('tensor_pad')}")

    # 4. Triton 推理
    inferencer = Inferencer(config)
    if not inferencer.is_ready():
        logger.error("Triton 模型未就绪!")
        return {"error": "model not ready"}

    logger.info("开始推理...")
    raw_output = inferencer.run_raw(tensor)
    logger.info(f"原始输出: shape={raw_output.shape}, dtype={raw_output.dtype}")

    # 5. 解码 OBB
    obb_list = inferencer.decode(raw_output)
    logger.info(f"检测到 {len(obb_list)} 个 OBB")

    if len(obb_list) > 0:
        for i, obb in enumerate(obb_list):
            pts = np.array(obb)
            cx, cy = pts.mean(axis=0)
            logger.info(f"  OBB[{i}]: center=({cx:.1f},{cy:.1f}), "
                        f"vertices={[[round(v[0],1),round(v[1],1)] for v in obb]}")

    # 6. 后处理
    postprocessor = Postprocessor(config)
    result = postprocessor.process(obb_list, meta)
    room_map = result["room_map"]
    threshold_list = result["threshold_list"]

    num_rooms = room_map.max()
    logger.info(f"后处理完成: {num_rooms} 个房间")

    # 7. 可视化输出
    if save_path or True:  # 始终生成可视化
        h, w = meta["map_data"].shape[:2]

        # 创建 2x2 对比图
        # (1) 原图 (2) OBB 检测 (3) threshold 线 (4) 房间分区
        orig_bgr = cv2.cvtColor(meta["input_data"], cv2.COLOR_GRAY2BGR)

        # OBB 叠加（需要逆映射回原图坐标）
        obb_vis = draw_obb_overlay(orig_bgr, obb_list)

        # threshold 线叠加
        thresh_vis = draw_threshold_overlay(orig_bgr, threshold_list)

        # 房间分区着色
        room_color = label_to_color(room_map)
        # 叠加到原图
        room_vis = orig_bgr.copy()
        mask = room_map > 0
        room_vis[mask] = cv2.addWeighted(
            room_vis[mask], 0.5, room_color[mask], 0.5, 0
        )

        # 拼接 2x2
        top = np.hstack([orig_bgr, obb_vis])
        bottom = np.hstack([thresh_vis, room_vis])
        canvas = np.vstack([top, bottom])

        # 添加标题
        font = cv2.FONT_HERSHEY_SIMPLEX
        titles = [
            (10, 25, f"Input ({w}x{h})"),
            (w + 10, 25, f"OBB ({len(obb_list)} detections)"),
            (10, h + 25, f"Threshold ({len(threshold_list)} lines)"),
            (w + 10, h + 25, f"Rooms ({num_rooms} rooms)"),
        ]
        for x, y, text in titles:
            cv2.putText(canvas, text, (x, y), font, 0.6, (0, 255, 0), 2)

        out_path = save_path or f"tests/output/infer_case_{case_dir.name}.png"
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(out_path, canvas)
        logger.info(f"可视化保存: {out_path}")

    return {
        "case": case_dir.name,
        "num_obbs": len(obb_list),
        "num_rooms": num_rooms,
        "map_size": (w, h),
    }


def main():
    parser = argparse.ArgumentParser(description="Triton 真实推理验证")
    parser.add_argument("--case", type=str, default="1", help="case 编号")
    parser.add_argument("--data-dir", type=str, default="gt_map_data")
    parser.add_argument("--triton-url", type=str, default="192.168.2.55:8001",
                        help="Triton gRPC 地址")
    parser.add_argument("--protocol", type=str, default="grpc",
                        choices=["grpc", "http"])
    parser.add_argument("--conf", type=float, default=0.5,
                        help="置信度阈值")
    parser.add_argument("--save", type=str, default=None,
                        help="保存可视化到指定路径")
    parser.add_argument("--all-cases", action="store_true",
                        help="遍历所有 case")
    args = parser.parse_args()

    # 加载配置并覆盖 Triton 参数
    config = load_config()
    config["triton_url"] = args.triton_url
    config["triton_protocol"] = args.protocol
    config["conf_threshold"] = args.conf

    data_root = ROOT / args.data_dir

    if args.all_cases:
        cases = sorted(
            [d.name for d in data_root.iterdir() if d.is_dir()],
            key=lambda x: int(x) if x.isdigit() else x,
        )
        results = []
        for case_id in cases:
            try:
                r = run_inference(data_root / case_id, config)
                results.append(r)
            except Exception as e:
                logger.error(f"Case {case_id} 失败: {e}")
                results.append({"case": case_id, "error": str(e)})

        # 汇总
        logger.info(f"\n{'='*50}")
        logger.info(f"汇总: {len(results)} cases")
        for r in results:
            if "error" in r:
                logger.info(f"  Case {r['case']}: ERROR - {r['error']}")
            else:
                logger.info(f"  Case {r['case']}: {r['num_obbs']} OBBs → "
                            f"{r['num_rooms']} rooms")
    else:
        case_dir = data_root / args.case
        if not case_dir.exists():
            logger.error(f"Case {args.case} 不存在: {case_dir}")
            sys.exit(1)
        run_inference(case_dir, config, args.save)


if __name__ == "__main__":
    main()
