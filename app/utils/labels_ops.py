"""轮廓扩展模块 —— 将房间轮廓向外扩展一圈，覆盖边界空闲像素"""

from typing import Dict, Any, List, Tuple

import numpy as np
import cv2
import logging

logger = logging.getLogger(f"{__name__} [ContourExpander]")


class ContourExpander:
    """
    轮廓外扩：基于 mask 填充 + 边线平移的像素级外扩。

    流程：
      1. 将轮廓填充为实心 mask
      2. 遍历每条边，仅对靠近白色(255)边界的边做外扩
      3. 沿外法线平移线段 1px，将不侵入其他房间的像素合入 mask
      4. 从合并后的 mask 提取最终轮廓

    线段方向分类：
      - 水平（夹角 < 1°）→ 垂直方向外扩
      - 垂直（夹角 > 89°）→ 水平方向外扩
      - 斜线（1°~89°）→ 夹角 > 45° 按垂直处理，≤ 45° 按水平处理
    """

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}

    @staticmethod
    def _to_gray(map_img: np.ndarray) -> np.ndarray:
        if map_img.ndim == 3:
            return cv2.cvtColor(map_img, cv2.COLOR_BGR2GRAY)
        return map_img.copy()

    # ==================== 核心外扩 ====================

    def contour_expand(self, contour: np.ndarray, map_img: np.ndarray,
                       distance: int = 1) -> np.ndarray:
        """
        基于 mask 的轮廓外扩。

        对每条边：若原始线段上存在白色(255)像素（即靠近边界），
        则沿外法线方向平移该线段，将平移后不侵入其他房间的像素合入 mask。
        最后从 mask 提取外扩后的轮廓。

        Args:
            contour: 单个房间轮廓 (N,1,2) 或 (N,2)
            map_img: 原始地图 (BGR 或灰度)
            distance: 外扩像素数，默认 1

        Returns:
            外扩后的轮廓 (N,1,2) int32
        """
        gray = self._to_gray(map_img)
        h, w = gray.shape
        pts = contour.reshape(-1, 2).astype(np.float64)
        n = len(pts)
        if n < 3:
            return contour.reshape(-1, 1, 2).astype(np.int32)

        # 判断轮廓绕向
        cnt_check = pts.reshape(-1, 1, 2).astype(np.float32)
        signed_area = cv2.contourArea(cnt_check, oriented=True)
        sign = 1.0 if signed_area > 0 else -1.0

        # 填充轮廓为实心 mask
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.drawContours(mask, [contour.reshape(-1, 1, 2).astype(np.int32)], -1, 255, -1)

        # 复用单个 buffer 避免每条边分配大数组
        line_buf = np.zeros((h, w), dtype=np.uint8)
        not_room = gray != 255  # 预计算：非房间区域

        for i in range(n):
            j = (i + 1) % n
            p1 = pts[i].astype(np.int32)
            p2 = pts[j].astype(np.int32)

            # 在 buffer 上画原始边，检查是否靠近白色边界
            line_buf.fill(0)
            cv2.line(line_buf, tuple(p1), tuple(p2), 255, 1)
            if not np.any(gray[line_buf == 255] == 255):
                continue

            # 计算外扩偏移
            dx = int(p2[0] - p1[0])
            dy = int(p2[1] - p1[1])
            seg_type, angle = self._classify_segment(dx, dy)
            offset_x, offset_y = self._calc_offset(seg_type, angle, dx, dy,
                                                    sign, distance)
            if offset_x == 0 and offset_y == 0:
                continue

            # 画平移后的线段，将不侵入其他房间的像素合入 mask
            line_buf.fill(0)
            new_p1 = (p1[0] + offset_x, p1[1] + offset_y)
            new_p2 = (p2[0] + offset_x, p2[1] + offset_y)
            cv2.line(line_buf, new_p1, new_p2, 255, 1)
            # 平移线段上非房间像素 → 合入 mask
            mask[(line_buf == 255) & not_room] = 255

        # 提取轮廓
        contours_found, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                             cv2.CHAIN_APPROX_SIMPLE)
        if contours_found:
            result = max(contours_found, key=cv2.contourArea)
        else:
            result = contour.reshape(-1, 1, 2).astype(np.int32)

        return result.reshape(-1, 1, 2).astype(np.int32)

    # ==================== 偏移计算 ====================

    @staticmethod
    def _calc_offset(seg_type: str, angle: float, dx: int, dy: int,
                     sign: float, distance: int) -> Tuple[int, int]:
        """
        根据线段方向计算外扩偏移。

        外法线用右转规则 (dx,dy)→(dy,-dx)，sign 控制绕向。

        Returns:
            (offset_x, offset_y)
        """
        if seg_type == "horizontal":
            return 0, int(sign * (-1.0 if dx > 0 else 1.0)) * distance

        elif seg_type == "vertical":
            return int(sign * (1.0 if dy > 0 else -1.0)) * distance, 0

        elif seg_type == "diagonal":
            if angle > 45:
                return int(sign * (1.0 if dy > 0 else -1.0)) * distance, 0
            else:
                return 0, int(sign * (-1.0 if dx > 0 else 1.0)) * distance

        return 0, 0

    # ==================== 线段分类 ====================

    @staticmethod
    def _classify_segment(dx: float, dy: float) -> Tuple[str, float]:
        """
        判断线段方向：horizontal / vertical / diagonal。

        Returns:
            (seg_type, angle_deg) angle_deg 为与水平轴的夹角 (0~90°)
        """
        length = np.sqrt(dx * dx + dy * dy)
        if length < 1e-6:
            return "horizontal", 0.0

        angle_deg = abs(np.degrees(np.arctan2(abs(dy), abs(dx))))

        if angle_deg < 1.0:
            return "horizontal", angle_deg
        elif angle_deg > 89.0:
            return "vertical", angle_deg
        return "diagonal", angle_deg

    # ==================== 批量接口 ====================

    def expand(self, contours: List[np.ndarray],
               map_img: np.ndarray) -> List[np.ndarray]:
        """对所有房间轮廓执行外扩"""
        return [self.contour_expand(cnt, map_img) for cnt in contours]


def expand_contours(contours: List[np.ndarray],
                    map_img: np.ndarray) -> List[np.ndarray]:
    return ContourExpander().expand(contours, map_img)


def expand_one(contour: np.ndarray, map_img: np.ndarray) -> np.ndarray:
    return ContourExpander().contour_expand(contour, map_img)


# 构造一个测试用例
if __name__ == "__main__":
    contour = np.array([[100, 100], [400, 100], [400, 400], [300, 400], [300, 500], [200, 500], [200, 400], [100, 100]])
    contour1 = np.array([[400, 100], [600, 100], [600, 200], [400, 200], [400, 100]])
    contour2 = np.array([[100, 500], [600, 500], [600, 800], [100, 800], [100, 500]])
    map_img = np.zeros((600, 900), dtype=np.uint8)
    cv2.drawContours(map_img, [contour], -1, 255, -1)
    cv2.drawContours(map_img, [contour1], -1, 255, -1)
    cv2.drawContours(map_img, [contour2], -1, 255, -1)
    draw_img = cv2.cvtColor(map_img, cv2.COLOR_GRAY2RGB) if map_img.ndim == 2 else map_img.copy()
    cv2.drawContours(draw_img, [contour], -1, (0, 0, 255), 1)
    cv2.imwrite("./dataset/debug/expand/0_map.png", draw_img)
    expander = ContourExpander()
    expanded_contour = expander.contour_expand(contour, map_img)
    print(expanded_contour)
    cv2.drawContours(draw_img, [expanded_contour], -1, (0, 255, 0), 1)
    cv2.imwrite("./dataset/debug/expand/1_map.png", draw_img)
