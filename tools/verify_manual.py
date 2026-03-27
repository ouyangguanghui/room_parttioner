#!/usr/bin/env python3
"""交互式验证脚本 —— 用真实 map_data 验证手动划分 & 手动合并

用法:
    python tools/verify_manual.py                  # 默认加载 gt_map_data/1
    python tools/verify_manual.py --case 11        # 指定 case

交互流程:
    1. 启动后显示带房间标注的地图，每个房间用不同颜色填充并标注 ROOM_ID + name
    2. 终端提示选择操作: [d]ivision / [m]erge / [n]ext case / [q]uit
    3. division: 在图上依次点两个点 (A, B) 画分割线，选择要分割的房间
    4. merge: 在终端输入要合并的房间 ID 列表
    5. 执行后显示结果对比 (before / after)

依赖: matplotlib, numpy, opencv-python, shapely
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.backend_bases import MouseButton
import numpy as np

# 项目根目录
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.core.config import load_config
from app.core.errors import RoomPartitionerError
from app.pipeline.preprocessor import Preprocessor
from app.services.manual_partition import ManualPartitioner
from app.services.manual_merge import ManualMerger
from app.utils.coordinate import CoordinateTransformer
from app.utils.graph import RoomGraph
from app.utils.landmark import LandmarkManager


# ==================== 数据加载 ====================

def load_case(case_dir: Path) -> dict:
    """加载单个 case 的 map_data (与 S3DataLoader.load() 格式一致)"""
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

# 预定义颜色 (R, G, B) 0-1
ROOM_COLORS = [
    (0.85, 0.33, 0.33),  # 红
    (0.33, 0.65, 0.85),  # 蓝
    (0.40, 0.78, 0.40),  # 绿
    (0.90, 0.70, 0.20),  # 橙
    (0.70, 0.40, 0.85),  # 紫
    (0.20, 0.80, 0.75),  # 青
    (0.90, 0.50, 0.65),  # 粉
    (0.55, 0.55, 0.55),  # 灰
    (0.95, 0.85, 0.30),  # 黄
    (0.35, 0.50, 0.75),  # 深蓝
]


def draw_rooms_on_map(ax, map_img, rooms_data, transformer, title=""):
    """在 ax 上绘制地图 + 房间填充 + 标注"""
    # 底图 (灰度)
    if map_img.ndim == 3:
        gray = cv2.cvtColor(map_img, cv2.COLOR_BGR2GRAY)
    else:
        gray = map_img
    h, w = gray.shape

    # 创建 RGBA 底图
    bg = np.zeros((h, w, 4), dtype=np.float32)
    norm = gray.astype(np.float32) / 255.0
    bg[..., 0] = norm
    bg[..., 1] = norm
    bg[..., 2] = norm
    bg[..., 3] = 1.0

    # 绘制房间填充
    overlay = bg.copy()
    for i, room in enumerate(rooms_data):
        geom = room.get("geometry", [])
        if len(geom) < 6:
            continue
        contour = transformer.world_to_contour(geom)
        color = ROOM_COLORS[i % len(ROOM_COLORS)]
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.drawContours(mask, [contour], -1, 1, -1)
        for c in range(3):
            overlay[..., c] = np.where(mask > 0, color[c] * 0.5 + overlay[..., c] * 0.5, overlay[..., c])

    ax.imshow(overlay)

    # 标注房间 ID + name
    legend_patches = []
    for i, room in enumerate(rooms_data):
        geom = room.get("geometry", [])
        if len(geom) < 6:
            continue
        contour = transformer.world_to_contour(geom)
        # 质心
        M = cv2.moments(contour)
        if M["m00"] > 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
        else:
            cx, cy = contour[0][0]

        color = ROOM_COLORS[i % len(ROOM_COLORS)]
        label = f"{room['id']}\n({room['name']})"
        ax.text(cx, cy, label, ha="center", va="center",
                fontsize=7, fontweight="bold",
                color="white", bbox=dict(boxstyle="round,pad=0.2",
                                         facecolor=color, alpha=0.8))
        legend_patches.append(
            mpatches.Patch(color=color, label=f"{room['id']} {room['name']}")
        )

    ax.set_title(title, fontsize=11)
    ax.axis("off")
    return legend_patches


# ==================== 交互: 画分割线 ====================

def pick_split_line(fig, ax, map_img, rooms_data, transformer):
    """交互选择分割线: 点击两个点 A, B"""
    print("\n[Division] 请在图上依次点击两个点 (A → B) 作为分割线")
    print("  提示: 分割线应穿过要分割的房间")

    points = []

    def on_click(event):
        if event.inaxes != ax or event.button != MouseButton.LEFT:
            return
        px, py = int(event.xdata), int(event.ydata)
        # 转换为世界坐标
        wx, wy = transformer.pixel_to_world(px, py)
        points.append((wx, wy, px, py))
        ax.plot(px, py, "rx", markersize=12, markeredgewidth=2)

        if len(points) == 1:
            print(f"  A = pixel({px},{py}) → world({wx:.3f},{wy:.3f})")
            print("  请点击第二个点 B...")
        elif len(points) == 2:
            print(f"  B = pixel({px},{py}) → world({wx:.3f},{wy:.3f})")
            # 画分割线
            ax.plot([points[0][2], points[1][2]],
                    [points[0][3], points[1][3]],
                    "r-", linewidth=2)
            fig.canvas.draw_idle()
            fig.canvas.mpl_disconnect(cid)

        fig.canvas.draw_idle()

    cid = fig.canvas.mpl_connect("button_press_event", on_click)
    print("  等待点击...(请在图上点击两个点)")

    # 阻塞等待两个点
    while len(points) < 2:
        plt.pause(0.1)

    # 选择要分割的房间
    print("\n  可分割的房间:")
    for i, room in enumerate(rooms_data):
        print(f"    [{i}] {room['id']} ({room['name']})")

    room_id = input("  输入要分割的房间 ID (如 ROOM_001): ").strip()

    return {
        "id": room_id,
        "A": [points[0][0], points[0][1]],
        "B": [points[1][0], points[1][1]],
    }


# ==================== 交互: 选合并房间 ====================

def pick_merge_rooms(rooms_data):
    """终端选择要合并的房间 ID 列表"""
    print("\n[Merge] 当前房间列表:")
    for i, room in enumerate(rooms_data):
        neighbors = room.get("graph", [])
        neighbor_ids = [rooms_data[n]["id"] for n in neighbors if n < len(rooms_data)]
        print(f"  {room['id']} ({room['name']}) — 邻居: {neighbor_ids}")

    raw = input("\n  输入要合并的房间 ID，逗号分隔 (如 ROOM_001,ROOM_002): ").strip()
    return [r.strip() for r in raw.split(",") if r.strip()]


# ==================== 执行并对比 ====================

def run_division(map_data, division_dict, config):
    """执行手动划分并返回结果"""
    partitioner = ManualPartitioner(config)
    transformer = CoordinateTransformer(
        map_data["resolution"], map_data["origin"],
        map_data["map_img"].shape[0],
    )
    graph_builder = RoomGraph(config)
    landmark_builder = LandmarkManager(config)

    return partitioner.process(
        map_data, division_dict,
        transformer, graph_builder, landmark_builder,
    )


def run_merge(map_data, merge_list, config):
    """执行手动合并并返回结果"""
    merger = ManualMerger(config)
    transformer = CoordinateTransformer(
        map_data["resolution"], map_data["origin"],
        map_data["map_img"].shape[0],
    )
    graph_builder = RoomGraph(config)
    landmark_builder = LandmarkManager(config)

    return merger.process(
        map_data, merge_list,
        transformer, graph_builder, landmark_builder,
    )


def show_comparison(map_data, old_labels, new_labels, config, operation):
    """before / after 对比显示"""
    transformer = CoordinateTransformer(
        map_data["resolution"], map_data["origin"],
        map_data["map_img"].shape[0],
    )

    old_rooms = [d for d in old_labels.get("data", []) if "ROOM" in d.get("id", "")]
    new_rooms = [d for d in new_labels.get("data", []) if "ROOM" in d.get("id", "")]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    draw_rooms_on_map(ax1, map_data["map_img"], old_rooms, transformer,
                      f"Before ({len(old_rooms)} rooms)")
    patches = draw_rooms_on_map(ax2, map_data["map_img"], new_rooms, transformer,
                                f"After {operation} ({len(new_rooms)} rooms)")

    fig.suptitle(f"{operation} 结果对比", fontsize=14, fontweight="bold")
    fig.tight_layout()
    plt.show(block=False)

    print(f"\n  结果: {len(old_rooms)} → {len(new_rooms)} 个房间")
    if operation == "division":
        print(f"  新增房间: {new_rooms[-1]['id']} ({new_rooms[-1]['name']})")


# ==================== 主循环 ====================

def main():
    parser = argparse.ArgumentParser(description="交互式验证手动划分/合并")
    parser.add_argument("--case", type=str, default="1", help="case 编号 (gt_map_data 子目录名)")
    parser.add_argument("--data-dir", type=str, default="gt_map_data", help="数据根目录")
    args = parser.parse_args()

    config = load_config()
    data_root = ROOT / args.data_dir

    # 可用 case 列表
    cases = sorted([d.name for d in data_root.iterdir() if d.is_dir()],
                   key=lambda x: int(x) if x.isdigit() else x)
    print(f"可用 case: {cases}")

    current_case = args.case

    while True:
        case_dir = data_root / current_case
        if not case_dir.exists():
            print(f"case {current_case} 不存在")
            current_case = input("输入 case 编号: ").strip()
            continue

        print(f"\n{'='*50}")
        print(f"加载 case: {current_case}")
        print(f"{'='*50}")

        map_data = load_case(case_dir)
        labels_json = map_data["labels_json"]

        if not labels_json or not labels_json.get("data"):
            print("  该 case 无 labels_json，跳过")
            current_case = input("输入下一个 case 编号 (q 退出): ").strip()
            if current_case == "q":
                break
            continue

        rooms_data = [d for d in labels_json["data"] if "ROOM" in d.get("id", "")]
        print(f"  房间数: {len(rooms_data)}")
        for r in rooms_data:
            print(f"    {r['id']} ({r['name']})")

        # 显示地图
        transformer = CoordinateTransformer(
            map_data["resolution"], map_data["origin"],
            map_data["map_img"].shape[0],
        )

        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        draw_rooms_on_map(ax, map_data["map_img"], rooms_data, transformer,
                          f"Case {current_case} — {len(rooms_data)} rooms")
        fig.tight_layout()
        plt.show(block=False)
        plt.pause(0.5)

        # 选择操作
        while True:
            choice = input("\n操作: [d]ivision / [m]erge / [n]ext case / [q]uit → ").strip().lower()

            if choice == "q":
                plt.close("all")
                return

            if choice == "n":
                plt.close("all")
                current_case = input("输入 case 编号: ").strip()
                break

            if choice == "d":
                try:
                    division_dict = pick_split_line(fig, ax, map_data["map_img"],
                                                    rooms_data, transformer)
                    print(f"\n  分割参数: {json.dumps(division_dict, ensure_ascii=False)}")
                    confirm = input("  确认执行? [y/n]: ").strip().lower()
                    if confirm != "y":
                        continue

                    result = run_division(map_data, division_dict, config)
                    show_comparison(map_data, labels_json, result, config, "division")

                    # 更新当前 labels 以支持连续操作
                    map_data["labels_json"] = result
                    labels_json = result
                    rooms_data = [d for d in result["data"] if "ROOM" in d.get("id", "")]

                except RoomPartitionerError as e:
                    print(f"  错误 (code={e.code}): {e}")
                except Exception as e:
                    print(f"  异常: {e}")

            elif choice == "m":
                try:
                    merge_list = pick_merge_rooms(rooms_data)
                    if len(merge_list) < 2:
                        print("  至少选择 2 个房间")
                        continue

                    print(f"\n  合并列表: {merge_list}")
                    confirm = input("  确认执行? [y/n]: ").strip().lower()
                    if confirm != "y":
                        continue

                    result = run_merge(map_data, merge_list, config)
                    show_comparison(map_data, labels_json, result, config, "merge")

                    # 更新当前 labels
                    map_data["labels_json"] = result
                    labels_json = result
                    rooms_data = [d for d in result["data"] if "ROOM" in d.get("id", "")]

                except RoomPartitionerError as e:
                    print(f"  错误 (code={e.code}): {e}")
                except Exception as e:
                    print(f"  异常: {e}")

            else:
                print("  无效选择，请输入 d/m/n/q")


if __name__ == "__main__":
    main()
