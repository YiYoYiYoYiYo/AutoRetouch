"""图像分割：GroundingDINO + SAM2，降级为 GrabCut"""

import logging

import cv2
import numpy as np
from PIL import Image

from config import cfg

logger = logging.getLogger(__name__)

# 尝试导入重量级模型（可选）
try:
    from transformers import pipeline as hf_pipeline
    _HAS_GDINO = True
except ImportError:
    _HAS_GDINO = False

try:
    from segment_anything_2 import sam_model_registry, SamPredictor
    _HAS_SAM2 = True
except ImportError:
    _HAS_SAM2 = False


class ImageSegmenter:
    """根据文字描述生成图像 mask"""

    def __init__(self):
        self._gdino = None
        self._sam = None
        self._mode = self._detect_mode()
        logger.info("分割模块初始化，模式: %s", self._mode)

    def _detect_mode(self) -> str:
        if _HAS_GDINO and _HAS_SAM2:
            return "gdino_sam2"
        if _HAS_GDINO:
            return "gdino_grabcut"
        return "grabcut"

    def segment(
        self,
        image: Image.Image,
        description: str,
        x: float = 0.5,
        y: float = 0.5,
        radius: float | None = None,
    ) -> np.ndarray:
        """根据描述生成 mask

        Args:
            image: PIL Image
            description: 区域描述文字（如"前景草丛"）
            x, y: 相对坐标 0-1（fallback 时用作中心点）
            radius: 相对半径（None 则使用配置默认值）

        Returns:
            mask: uint8 numpy 数组，0-255，与 image 同尺寸
        """
        if radius is None:
            radius = cfg.processing.local_adjustment_radius

        w, h = image.size

        if self._mode == "gdino_sam2":
            return self._segment_gdino_sam2(image, description)
        if self._mode == "gdino_grabcut":
            return self._segment_gdino_grabcut(image, description, x, y, radius)
        return self._segment_grabcut(image, x, y, radius)

    def _segment_grabcut(
        self, image: Image.Image, x: float, y: float, radius: float
    ) -> np.ndarray:
        """GrabCut 降级方案：以坐标为中心生成 mask"""
        img = np.array(image.convert("RGB"))
        h, w = img.shape[:2]

        # 中心点和矩形区域
        cx, cy = int(x * w), int(y * h)
        rw, rh = int(radius * w), int(radius * h)
        rect = (
            max(0, cx - rw),
            max(0, cy - rh),
            min(w, cx + rw),
            min(h, cy + rh),
        )

        mask = np.zeros((h, w), np.uint8)
        bgd_model = np.zeros((1, 65), np.float64)
        fgd_model = np.zeros((1, 65), np.float64)

        try:
            cv2.grabCut(img, mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)
            # 转为二值 mask
            result = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0)
        except cv2.error:
            # GrabCut 失败时使用椭圆 fallback
            result = np.zeros((h, w), np.uint8)
            cv2.ellipse(result, (cx, cy), (rw, rh), 0, 0, 360, 255, -1)

        # 羽化
        feather = cfg.processing.mask_feather
        if feather > 0:
            ksize = feather * 2 + 1
            result = cv2.GaussianBlur(
                result.astype(np.float32), (ksize, ksize), 0
            ).astype(np.uint8)

        return result

    def _segment_gdino_grabcut(
        self, image: Image.Image, description: str,
        x: float, y: float, radius: float,
    ) -> np.ndarray:
        """GroundingDINO 检测 + GrabCut 分割"""
        if self._gdino is None:
            self._gdino = hf_pipeline(
                "zero-object-detection",
                model="IDEA-Research/grounding-dino-tiny",
                device=-1,  # CPU
            )

        results = self._gdino(image, candidate_labels=[description])
        if not results:
            logger.warning("GroundingDINO 未检测到 '%s'，降级为 GrabCut", description)
            return self._segment_grabcut(image, x, y, radius)

        # 取最高置信度的检测结果
        best = max(results, key=lambda r: r["score"])
        box = best["box"]  # [x1, y1, x2, y2] 绝对坐标
        w, h = image.size

        # 用检测到的 box 做 GrabCut
        img = np.array(image.convert("RGB"))
        mask = np.zeros((h, w), np.uint8)
        bgd_model = np.zeros((1, 65), np.float64)
        fgd_model = np.zeros((1, 65), np.float64)
        rect = (int(box[0]), int(box[1]), int(box[2]), int(box[3]))

        try:
            cv2.grabCut(img, mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)
            result = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0)
        except cv2.error:
            result = np.zeros((h, w), np.uint8)
            cv2.rectangle(result, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), 255, -1)

        feather = cfg.processing.mask_feather
        if feather > 0:
            ksize = feather * 2 + 1
            result = cv2.GaussianBlur(
                result.astype(np.float32), (ksize, ksize), 0
            ).astype(np.uint8)

        return result

    def _segment_gdino_sam2(
        self, image: Image.Image, description: str,
    ) -> np.ndarray:
        """GroundingDINO + SAM2 精确分割（最高质量）"""
        raise NotImplementedError("SAM2 分割需要额外模型文件，暂未实现")
