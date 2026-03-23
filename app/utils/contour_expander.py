"""轮廓扩展模块 —— 将房间轮廓向外扩展一圈，覆盖边界空闲像素"""

from typing import Dict, Any, List

import numpy as np
import cv2


class ContourExpander:
    """
    轮廓外扩：将房间边界上紧邻的空闲像素并入房间

    原理: 对每个房间轮廓取 1px 外圈，如果外圈像素是空闲空间
    (非墙壁)，则并入该房间。
    """

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.neighbors = [(1, 0), (-1, 0), (0, 1), (0, -1)]

    def expand(self, contours: List[np.ndarray],
               map_img: np.ndarray) -> List[np.ndarray]:
        """
        对所有房间轮廓执行外扩

        Args:
            contours: 房间轮廓列表 [(N,1,2), ...]
            map_img: 原始地图 (BGR 或灰度)

        Returns:
            外扩后的轮廓列表
        """
        if map_img.ndim == 3:
            gray = cv2.cvtColor(map_img, cv2.COLOR_BGR2GRAY)
        else:
            gray = map_img.copy()

        h, w = gray.shape
        mask = np.zeros((h, w), dtype=np.uint8)
        result = []

        for cnt in contours:
            mask.fill(0)
            # 填充内部
            cv2.drawContours(mask, [cnt], -1, 255, -1)
            # 标记边界
            cv2.drawContours(mask, [cnt], -1, 200, 1)

            # 边界像素向外扩展
            rows, cols = np.where(mask == 200)
            for y, x in zip(rows, cols):
                if gray[y, x] == 255:  # 边界像素是空闲空间
                    for dx, dy in self.neighbors:
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < w and 0 <= ny < h:
                            if gray[ny, nx] != 255 and mask[ny, nx] != 255:
                                mask[ny, nx] = 255

            mask[mask == 200] = 255

            # 提取新轮廓
            new_contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                               cv2.CHAIN_APPROX_SIMPLE)
            if len(new_contours) == 1:
                result.append(new_contours[0])
            else:
                result.append(cnt)

        return result
