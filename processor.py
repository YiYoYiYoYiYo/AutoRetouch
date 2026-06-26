"""图像处理引擎：全局调整 + 局部 mask 调整"""

import cv2
import numpy as np
from PIL import Image

from config import cfg
from vlm.base import EditSuggestion, GlobalParams, LocalAdjustment
from segmentation.segmenter import ImageSegmenter


class ImageProcessor:
    """执行修图建议的图像处理引擎"""

    def __init__(self, segmenter: ImageSegmenter | None = None):
        self._segmenter = segmenter or ImageSegmenter()

    def process(self, image: Image.Image, suggestion: EditSuggestion) -> Image.Image:
        """根据修图建议处理图片

        Args:
            image: 原始图片
            suggestion: VLM 输出的修图建议

        Returns:
            处理后的图片
        """
        # 转为 numpy float32 (0-1 范围)
        img = np.array(image.convert("RGB")).astype(np.float32) / 255.0

        # 1. 全局调整
        img = self._apply_global(img, suggestion.global_params)

        # 2. 局部调整
        for adj in suggestion.local_adjustments:
            mask = self._segmenter.segment(
                image, adj.description, adj.x, adj.y,
            )
            img = self._apply_local(img, adj, mask)

        # 裁剪到合法范围并转回 uint8
        img = np.clip(img * 255, 0, 255).astype(np.uint8)
        return Image.fromarray(img)

    # ── 全局调整 ─────────────────────────────────────

    def _apply_global(self, img: np.ndarray, gp: GlobalParams) -> np.ndarray:
        """应用全局调整"""
        # 曝光：EV 值 → 亮度乘数
        if gp.exposure_ev != 0:
            img = img * (2.0 ** gp.exposure_ev)

        # 白平衡：色温 K → RGB 增益
        if gp.white_balance_k != 5500:
            img = self._apply_white_balance(img, gp.white_balance_k)

        # 对比度
        if gp.contrast != 0:
            factor = 1.0 + gp.contrast / 100.0
            img = (img - 0.5) * factor + 0.5

        # 高光
        if gp.highlights != 0:
            mask = img > 0.5
            shift = gp.highlights / 200.0
            img[mask] = np.clip(img[mask] + shift, 0, 1)

        # 阴影
        if gp.shadows != 0:
            mask = img < 0.5
            shift = gp.shadows / 200.0
            img[mask] = np.clip(img[mask] + shift, 0, 1)

        # 饱和度
        if gp.saturation != 0:
            img = self._apply_saturation(img, gp.saturation)

        return np.clip(img, 0, 1)

    def _apply_white_balance(self, img: np.ndarray, temp_k: int) -> np.ndarray:
        """色温调整：基于 Planckian locus 近似"""
        # 将色温映射到 RGB 增益
        # 参考: http://www.tannerhelland.com/4435/convert-temperature-rgb-algorithm-code/
        temp = temp_k / 100.0

        # Red channel
        if temp <= 66:
            r = 255.0
        else:
            r = 329.698727446 * ((temp - 60) ** -0.1332047592)
            r = max(0, min(255, r))

        # Green channel
        if temp <= 66:
            g = 99.4708025861 * np.log(temp) - 161.1195681661
        else:
            g = 288.1221695283 * ((temp - 60) ** -0.0755148492)
        g = max(0, min(255, g))

        # Blue channel
        if temp >= 66:
            b = 255.0
        elif temp <= 19:
            b = 0.0
        else:
            b = 138.5177312231 * np.log(temp - 10) - 305.0447927307
            b = max(0, min(255, b))

        # 归一化增益
        r_gain = r / 255.0
        g_gain = g / 255.0
        b_gain = b / 255.0

        # 中性化（相对 5500K 的增益）
        neutral = np.array([1.0, 1.0, 1.0])  # 5500K 近似中性
        gain = np.array([r_gain, g_gain, b_gain]) / neutral

        return img * gain

    def _apply_saturation(self, img: np.ndarray, amount: int) -> np.ndarray:
        """饱和度调整"""
        # 转到 HSV
        hsv = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2HSV).astype(np.float32)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * (1.0 + amount / 100.0), 0, 255)
        result = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
        return result.astype(np.float32) / 255.0

    # ── 局部调整 ─────────────────────────────────────

    def _apply_local(
        self, img: np.ndarray, adj: LocalAdjustment, mask: np.ndarray,
    ) -> np.ndarray:
        """应用局部调整"""
        h, w = img.shape[:2]

        # 调整 mask 尺寸
        if mask.shape[:2] != (h, w):
            mask = cv2.resize(mask, (w, h))

        # 归一化 mask 到 0-1
        mask_f = mask.astype(np.float32) / 255.0

        # 创建调整层
        adjustment = np.zeros_like(img)

        if adj.adjustment_type in ("brighten", "darken"):
            ev = adj.exposure_ev if adj.adjustment_type == "brighten" else -abs(adj.exposure_ev)
            adjustment = np.full_like(img, 2.0 ** ev - 1.0)
        elif adj.adjustment_type == "warm":
            adjustment[:, :, 0] = 0.05  # 增加红色
            adjustment[:, :, 2] = -0.05  # 减少蓝色
        elif adj.adjustment_type == "cool":
            adjustment[:, :, 0] = -0.05
            adjustment[:, :, 2] = 0.05

        if adj.temperature_shift != 0:
            shift = adj.temperature_shift / 2000.0
            adjustment[:, :, 0] += shift
            adjustment[:, :, 2] -= shift

        # 应用 mask
        for c in range(3):
            img[:, :, c] += adjustment[:, :, c] * mask_f

        return np.clip(img, 0, 1)
