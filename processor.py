"""图像处理引擎：全局调整 + 局部 mask 调整

采用接近 Lightroom 的调色逻辑：
- 曝光：gamma 校正（非线性乘法），保护高光
- 对比度：S 曲线（sigmoid），保持动态范围
- 高光/阴影：区域化 tone mapping，不互相影响
- 白平衡：相对增益校正
"""

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
        """根据修图建议处理图片"""
        img = np.array(image.convert("RGB")).astype(np.float32) / 255.0

        # 1. 全局调整（基于原图一次性计算）
        img = self._apply_global(img, suggestion.global_params)

        # 2. 局部调整
        for adj in suggestion.local_adjustments:
            mask = self._segmenter.segment(
                image, adj.description, adj.x, adj.y,
            )
            img = self._apply_local(img, adj, mask)

        img = np.clip(img * 255, 0, 255).astype(np.uint8)
        return Image.fromarray(img)

    # ── 全局调整 ─────────────────────────────────────

    def _apply_global(self, img: np.ndarray, gp: GlobalParams) -> np.ndarray:
        """应用全局调整"""
        # 白平衡（最先做，不改变亮度）
        if gp.white_balance_k != 5500:
            img = self._apply_white_balance(img, gp.white_balance_k)

        # 曝光：gamma 校正（比线性乘法更自然）
        if gp.exposure_ev != 0:
            gamma = 1.0 / (2.0 ** gp.exposure_ev)
            img = np.power(np.clip(img, 0, 1), gamma)

        # 对比度：S 曲线（sigmoid 方式，不溢出）
        if gp.contrast != 0:
            strength = gp.contrast / 200.0  # 归一化到 -0.5 ~ +0.5
            img = self._sigmoid_contrast(img, strength)

        # 高光：仅影响亮部（加权混合）
        if gp.highlights != 0:
            img = self._tone_region(img, gp.highlights / 100.0, mode="highlights")

        # 阴影：仅影响暗部
        if gp.shadows != 0:
            img = self._tone_region(img, gp.shadows / 100.0, mode="shadows")

        # 饱和度
        if gp.saturation != 0:
            img = self._apply_saturation(img, gp.saturation)

        return np.clip(img, 0, 1)

    @staticmethod
    def _sigmoid_contrast(img: np.ndarray, strength: float) -> np.ndarray:
        """S 曲线对比度调整，strength ∈ (-0.5, +0.5)"""
        if strength == 0:
            return img
        # 中点 0.5 的 sigmoid 变换
        midpoint = 0.5
        # 调节 steepness
        k = strength * 10.0  # 映射到 -5 ~ +5
        # 避免除零
        result = 1.0 / (1.0 + np.exp(-k * (img - midpoint)))
        # 归一化到 0-1（sigmoid 在 0 和 1 处渐近）
        lo = 1.0 / (1.0 + np.exp(-k * (0.0 - midpoint)))
        hi = 1.0 / (1.0 + np.exp(-k * (1.0 - midpoint)))
        result = (result - lo) / (hi - lo)
        return np.clip(result, 0, 1)

    @staticmethod
    def _tone_region(img: np.ndarray, amount: float, mode: str) -> np.ndarray:
        """区域化 tone mapping：只影响高光或阴影区域

        使用 luminance 作为权重，避免区域边界跳变。
        """
        # 亮度权重（0-1）
        lum = 0.2126 * img[:, :, 0] + 0.7152 * img[:, :, 1] + 0.0722 * img[:, :, 2]

        if mode == "highlights":
            # 亮部权重：luminance 越高，权重越大（smoothstep）
            weight = np.clip((lum - 0.5) * 2.0, 0, 1) ** 2
            # 调整方向：正值提亮，负值压暗
            shift = amount * 0.3  # 控制幅度
        else:  # shadows
            # 暗部权重：luminance 越低，权重越大
            weight = np.clip(1.0 - lum * 2.0, 0, 1) ** 2
            shift = amount * 0.3

        # 用 weight 做加权混合
        for c in range(3):
            img[:, :, c] += shift * weight

        return np.clip(img, 0, 1)

    @staticmethod
    def _apply_white_balance(img: np.ndarray, temp_k: int) -> np.ndarray:
        """色温调整：相对 6500K（标准日光）的增益校正"""
        temp = temp_k / 100.0

        # 计算目标色温的 RGB 值（Tanner Helland 算法）
        def _temp_to_rgb(t: float) -> tuple[float, float, float]:
            # Red
            r = 255.0 if t <= 66 else 329.698727446 * ((t - 60) ** -0.1332047592)
            # Green
            g = (99.4708025861 * np.log(t) - 161.1195681661) if t <= 66 else 288.1221695283 * ((t - 60) ** -0.0755148492)
            # Blue
            if t >= 66:
                b = 255.0
            elif t <= 19:
                b = 0.0
            else:
                b = 138.5177312231 * np.log(t - 10) - 305.0447927307
            return (max(0, min(255, r)) / 255, max(0, min(255, g)) / 255, max(0, min(255, b)) / 255)

        # 参考色温 6500K（标准日光白平衡点）
        ref_r, ref_g, ref_b = _temp_to_rgb(65.0)
        tgt_r, tgt_g, tgt_b = _temp_to_rgb(temp)

        # 相对增益
        gain = np.array([tgt_r / ref_r, tgt_g / ref_g, tgt_b / ref_b], dtype=np.float32)

        # 平衡总亮度（避免色温改变整体亮度）
        gain /= gain.mean()

        return img * gain

    @staticmethod
    def _apply_saturation(img: np.ndarray, amount: int) -> np.ndarray:
        """饱和度调整（在 HSV 空间）"""
        hsv = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2HSV).astype(np.float32)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * (1.0 + amount / 100.0), 0, 255)
        result = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
        return result.astype(np.float32) / 255.0

    # ── 局部调整 ─────────────────────────────────────

    @staticmethod
    def _apply_local(
        img: np.ndarray, adj: LocalAdjustment, mask: np.ndarray,
    ) -> np.ndarray:
        """应用局部调整（加权混合，不直接加法）"""
        h, w = img.shape[:2]

        if mask.shape[:2] != (h, w):
            mask = cv2.resize(mask, (w, h))

        mask_f = mask.astype(np.float32) / 255.0

        # 计算调整后的版本
        adjusted = img.copy()

        t = adj.adjustment_type.lower()

        if t in ("brighten", "darken", "shadows", "highlights"):
            ev = abs(adj.exposure_ev) if adj.exposure_ev != 0 else 0.3
            if t == "darken":
                ev = -ev
            elif t == "shadows":
                ev = abs(ev)  # shadows 总是提亮
            elif t == "highlights":
                ev = -abs(ev)  # highlights 总是压暗
            gamma = 1.0 / (2.0 ** ev) if ev != 0 else 1.0
            adjusted = np.power(np.clip(adjusted, 0, 1), gamma)
        elif t in ("warm", "warmth"):
            shift = max(abs(adj.temperature_shift) / 2000.0, 0.15)
            adjusted[:, :, 0] = np.clip(adjusted[:, :, 0] + shift, 0, 1)
            adjusted[:, :, 2] = np.clip(adjusted[:, :, 2] - shift * 0.7, 0, 1)
        elif t in ("cool", "cooling"):
            shift = max(abs(adj.temperature_shift) / 2000.0, 0.15)
            adjusted[:, :, 0] = np.clip(adjusted[:, :, 0] - shift * 0.7, 0, 1)
            adjusted[:, :, 2] = np.clip(adjusted[:, :, 2] + shift, 0, 1)

        if adj.temperature_shift != 0:
            shift = adj.temperature_shift / 2000.0
            adjusted[:, :, 0] = np.clip(adjusted[:, :, 0] + shift, 0, 1)
            adjusted[:, :, 2] = np.clip(adjusted[:, :, 2] - shift, 0, 1)

        # 用 mask 做加权混合（而非加法）
        for c in range(3):
            img[:, :, c] = img[:, :, c] * (1 - mask_f) + adjusted[:, :, c] * mask_f

        return np.clip(img, 0, 1)
