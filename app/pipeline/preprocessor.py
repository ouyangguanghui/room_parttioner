"""
图像前处理模块
去除噪声、补全墙壁、平滑处理

像素值说明：
- 0: 墙壁
- 247: 未知区域
- 254: 自由空间

处理流程：
1. 去除自由空间内部黑色噪点（小面积孤立墙壁）
2. 去除小面积未知区域噪点，按边界邻域统计填充为墙壁或自由空间
3. 补全墙壁：在未知区域与自由空间直接接触处补墙
4. 平滑墙壁：形态学闭运算 + 去毛刺 + 高斯平滑
"""

import logging
from typing import Dict, Any, Tuple, Optional
from pathlib import Path

import numpy as np
import cv2

logger = logging.getLogger(__name__)

# 像素值常量
WALL = 0
UNKNOWN = 127
FREE = 255

class Preprocessor:
    """栅格地图前处理"""

    def __init__(self, config: Dict[str, Any] = None):
        config = config or {}
        self.max_noise_area: int = config.get("max_noise_area", 30)
        self.dilate_iter: int = config.get("dilate_iter", 2)
        self.smooth_iterations: int = config.get("smooth_iterations", 2)
        self.max_burr_len: int = config.get("max_burr_len", 4)
        self.target_size: int = config.get("target_size", [512, 512])
        self._save_viz: bool = config.get("save_viz", False)
        self._viz_dir: Optional[Path] = Path(config["viz_dir"]) if config.get("viz_dir") else None
        

        # 预生成常用 kernel
        self._k3 = np.ones((3, 3), np.uint8)
        self._k5 = np.ones((5, 5), np.uint8)

    # ========== 核心处理流程 ==========
    def process(self, img: np.ndarray) -> Dict[str, Any]:
        """
        完整前处理流程

        Args:
            img: 原始栅格地图 (H, W) uint8

        Returns:
            字典:
                "cleaned_img":  去噪后的地图 (H, W) uint8
                "cleaned_img2": 去噪+去未知噪点后的地图 (H, W) uint8
                "input_img":    补墙平滑后的地图 (H, W) uint8，用于自由空间判断和模型张量准备
        """
        viz_data = {}
        viz_status = True if self._viz_dir else False

        # Step 0: 像素值标准化
        img = self._map_pixels(img)

        # Step 1: 去除自由空间内部黑色噪点
        cleaned, noise_mask, stats1 = self.remove_interior_noise(img)
        logger.info("内部噪点: 移除 %d / 共 %d 个黑色连通域",
                    stats1["removed"], stats1["total_black_components"])
        if viz_status:
            viz_data["step1"] = (img, cleaned, noise_mask)

        # Step 2: 去除未知区域噪点
        cleaned2, unknown_mask, stats2 = self.remove_unknown_noise(cleaned)
        logger.info("未知噪点: 移除 %d / 共 %d 个未知连通域",
                    stats2["removed"], stats2["total_unknown_components"])
        if viz_status:
            viz_data["step2"] = (cleaned, cleaned2, unknown_mask)

        # Step 3: 补墙 + 平滑
        result, added_wall, stats3 = self.fill_and_smooth(cleaned2)
        logger.info("墙壁: 原始 %d px, 补充 %d px, 最终 %d px (净变化 %+d)",
                    stats3["original_wall_pixels"], stats3["added_boundary_pixels"],
                    stats3["final_wall_pixels"], stats3["net_change"])
        if viz_status:
            viz_data["step3"] = (cleaned2, result, added_wall)

        if viz_status and viz_data:
            self._save_visualization(viz_data)

        return {
            "cleaned_img" : cleaned,
            "cleaned_img2" : cleaned2,
            "input_img": result,
        }

    # ========== Step 1: 去除内部噪点 ==========

    def remove_interior_noise(self, img: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """去除自由空间内部的黑色噪点（被自由空间包围的小黑块）"""
        cleaned = img.copy()

        free_dilated = cv2.dilate(
            (img == FREE).astype(np.uint8), self._k5, iterations=self.dilate_iter
        )

        black_mask = (img == WALL).astype(np.uint8)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(black_mask, connectivity=8)

        removed = 0
        removed_areas = []

        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            if area > self.max_noise_area:
                continue

            component = labels == i
            if np.all(free_dilated[component]):
                cleaned[component] = FREE
                removed += 1
                removed_areas.append(int(area))

        noise_mask = (img != cleaned).astype(np.uint8) * 255
        return cleaned, noise_mask, {
            "total_black_components": int(num_labels - 1),
            "removed": removed,
            "kept": int(num_labels - 1 - removed),
            "removed_areas": removed_areas,
        }

    # ========== Step 2: 去除未知区域噪点 ==========

    def remove_unknown_noise(self, img: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """去除小面积未知区域噪点，按边界环邻域统计决定填充值"""
        cleaned = img.copy()
        wall = (img == WALL).astype(np.uint8)
        free = (img == FREE).astype(np.uint8)

        unknown_mask = (img == UNKNOWN).astype(np.uint8)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(unknown_mask, connectivity=8)

        removed = 0
        removed_areas = []

        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            if area > self.max_noise_area:
                continue

            component = (labels == i).astype(np.uint8)
            # 边界环：膨胀 - 自身
            ring = cv2.dilate(component, self._k3, iterations=1)
            ring = (ring > 0) & (component == 0)

            wall_count = int(wall[ring].sum())
            free_count = int(free[ring].sum())

            cleaned[labels == i] = WALL if wall_count > free_count else FREE
            removed += 1
            removed_areas.append(int(area))

        noise_mask = (img != cleaned).astype(np.uint8) * 255
        return cleaned, noise_mask, {
            "total_unknown_components": int(num_labels - 1),
            "removed": removed,
            "kept": int(num_labels - 1 - removed),
            "removed_areas": removed_areas,
        }

    # ========== Step 3: 补墙 + 平滑 ==========

    def fill_and_smooth(self, img: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """补墙 + 平滑完整流程"""
        # 补墙
        filled = img.copy()
        unknown = (img == UNKNOWN).astype(np.uint8)
        free = (img == FREE).astype(np.uint8)
        boundary = (cv2.dilate(unknown, self._k3, iterations=1) > 0) & (free > 0)
        filled[boundary] = WALL
        added_boundary = int(boundary.sum())

        # 平滑
        result = self._smooth_wall(filled)

        orig_wall = int((img == WALL).sum())
        final_wall = int((result == WALL).sum())
        added_wall = ((result == WALL) & (img != WALL)).astype(np.uint8)

        return result, added_wall, {
            "original_wall_pixels": orig_wall,
            "added_boundary_pixels": added_boundary,
            "final_wall_pixels": final_wall,
            "net_change": final_wall - orig_wall,
        }

    def _smooth_wall(self, img: np.ndarray) -> np.ndarray:
        """墙壁平滑：方向性闭运算 + 去毛刺 + 高斯"""
        wall = (img == WALL).astype(np.uint8)

        kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 1))
        kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 3))

        smoothed = wall.copy()
        for _ in range(self.smooth_iterations):
            smoothed = cv2.morphologyEx(smoothed, cv2.MORPH_CLOSE, kernel_h)
            smoothed = cv2.morphologyEx(smoothed, cv2.MORPH_CLOSE, kernel_v)

        # 去毛刺：端点剪枝
        smoothed = self._remove_burrs(smoothed)

        # 高斯 + 阈值
        blurred = cv2.GaussianBlur(smoothed.astype(np.float32), (3, 3), sigmaX=0.8)
        smoothed = (blurred > 0.5).astype(np.uint8)

        # 合成结果
        result = img.copy()
        new_wall = (smoothed > 0) & (wall == 0)
        result[new_wall] = WALL

        # 被移除的墙壁：按邻域决定恢复为 FREE 还是 UNKNOWN
        removed = (smoothed == 0) & (wall > 0)
        if removed.any():
            free_nearby = cv2.dilate(
                (img == FREE).astype(np.uint8), self._k3, iterations=1
            ) > 0
            result[removed] = np.where(free_nearby[removed], FREE, UNKNOWN).astype(np.uint8)

        return result

    def _remove_burrs(self, wall_mask: np.ndarray) -> np.ndarray:
        """去除墙壁短毛刺（端点剪枝法）"""
        result = wall_mask.copy()
        neighbor_kernel = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=np.uint8)

        for _ in range(self.max_burr_len):
            neighbors = cv2.filter2D(result, -1, neighbor_kernel)
            endpoints = (result > 0) & (neighbors <= 1)
            if not endpoints.any():
                break
            result[endpoints] = 0

        return result

    # ========= Step 4: 像素映射（可选） ==========
    def _map_pixels(self, img: np.ndarray) -> np.ndarray:
        """将像素值映射到新的范围（如 WALL=0, UNKNOWN=127, FREE=255）"""
        mapped = np.zeros_like(img, dtype=np.uint8)
        PIXEL_MAP = {
            0 : 0,       # WALL
            247 : 127,   # UNKNOWN
            254 : 255,   # FREE
        }
        for k, v in PIXEL_MAP.items():
            mapped[img == k] = v
        return mapped

    # ========== 可视化 ==========
    def _save_visualization(self, viz_data: Dict) -> None:
        """保存可视化结果"""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        self._viz_dir.mkdir(parents=True, exist_ok=True)

        fig, axes = plt.subplots(3, 4, figsize=(24, 18))

        for row, (step_name, (before, after, mask)) in enumerate(viz_data.items()):
            ax = axes[row]

            ax[0].imshow(before, cmap="gray", vmin=0, vmax=255)
            ax[0].set_title(f"{step_name} - Input")

            # 高亮标记
            highlight = cv2.cvtColor(before, cv2.COLOR_GRAY2BGR)
            highlight[mask > 0] = [0, 0, 255]
            ax[1].imshow(cv2.cvtColor(highlight, cv2.COLOR_BGR2RGB))
            ax[1].set_title(f"{step_name} - Highlighted")

            ax[2].imshow(after, cmap="gray", vmin=0, vmax=255)
            ax[2].set_title(f"{step_name} - Output")

            diff = cv2.absdiff(before, after)
            ax[3].imshow(diff, cmap="hot")
            ax[3].set_title(f"{step_name} - Diff")

            for a in ax:
                a.axis("off")

        plt.tight_layout()
        out_path = self._viz_dir / "preprocess_viz.png"
        fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info("可视化已保存: %s", out_path)

    def set_viz_dir(self, viz_dir: Optional[str]) -> None:
        """设置可视化结果保存目录"""
        self._viz_dir = Path(viz_dir) if viz_dir else None
