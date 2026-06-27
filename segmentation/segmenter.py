"""图像分割：SAM2（默认） / GroundingDINO+GrabCut / GrabCut（降级）

模式选择：
1. SAM2（transformers 内置 Sam2Model）——默认，用 VLM 给的坐标点生成高质量语义 mask
2. GDINO + GrabCut——可选，用 GroundingDINO 按文字描述检测 box，再 GrabCut
3. GrabCut——降级方案，纯坐标 + 固定半径

依赖未安装或推理失败时，自动降级到 GrabCut，segment() 永不抛异常。
"""

import logging

import cv2
import numpy as np
from PIL import Image

from config import cfg

logger = logging.getLogger(__name__)

# ── 可选依赖：torch + Sam2 同一 try 块，任一缺失即降级 ──
try:
    import torch
    from transformers import Sam2Model, Sam2Processor
    _HAS_SAM2 = True
except ImportError:
    _HAS_SAM2 = False

# GroundingDINO 仅需 transformers（可能单独可用）
try:
    from transformers import pipeline as hf_pipeline
    _HAS_GDINO = True
except ImportError:
    _HAS_GDINO = False


class ImageSegmenter:
    """根据坐标/文字描述生成图像 mask"""

    def __init__(self):
        self._sam = None        # Sam2Model，懒加载
        self._sam_proc = None   # Sam2Processor，懒加载
        self._gdino = None      # GroundingDINO pipeline，懒加载
        self._mode = self._detect_mode()
        logger.info("分割模块初始化，模式: %s (SAM2可用=%s, GDINO可用=%s)",
                    self._mode, _HAS_SAM2, _HAS_GDINO)

    def _detect_mode(self) -> str:
        seg = cfg.segmentation
        if seg.enable_sam2 and _HAS_SAM2:
            return "sam2"
        if seg.use_gdino and _HAS_GDINO:
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
        """生成 mask，永不抛异常，返回与 image 同尺寸的 uint8 (0-255)

        Args:
            image: PIL Image
            description: 区域描述文字（GDINO 开启时用于检测）
            x, y: 相对坐标 0-1（SAM2 点提示 / GrabCut 中心）
            radius: 相对半径（None 用配置默认，仅 GrabCut 用）
        """
        if radius is None:
            radius = cfg.processing.local_adjustment_radius

        w, h = image.size

        # SAM2 路径：先尝试点提示；use_gdino 开启时优先用 box 提示
        if self._mode == "sam2":
            try:
                if cfg.segmentation.use_gdino and _HAS_GDINO:
                    box = self._gdino_box(image, description)
                    if box is not None:
                        logger.info("[segment] '%s' → GDINO 检测到 box %s → SAM2(box)", description, box)
                        return self._segment_sam2(image, input_boxes=box)
                    logger.info("[segment] '%s' → GDINO 未检出 → SAM2(point xy=%.2f,%.2f)", description, x, y)
                else:
                    logger.info("[segment] '%s' → SAM2(point xy=%.2f,%.2f)", description, x, y)
                return self._segment_sam2(image, point=(x, y))
            except Exception as e:
                logger.warning("[segment] '%s' → SAM2 失败，降级 GrabCut: %s", description, e)

        if self._mode == "gdino_grabcut" or (self._mode == "sam2" and cfg.segmentation.use_gdino):
            try:
                logger.info("[segment] '%s' → GDINO+GrabCut(xy=%.2f,%.2f)", description, x, y)
                return self._segment_gdino_grabcut(image, description, x, y, radius)
            except Exception as e:
                logger.warning("[segment] '%s' → GDINO+GrabCut 失败，降级纯 GrabCut: %s", description, e)

        logger.info("[segment] '%s' → GrabCut(xy=%.2f,%.2f, r=%.2f)", description, x, y, radius)
        return self._segment_grabcut(image, x, y, radius)

    # ── SAM2 ─────────────────────────────────────────

    def _ensure_sam(self):
        """懒加载 SAM2 模型（首图才加载，避免 app 启动慢）"""
        if self._sam is not None:
            return
        seg = cfg.segmentation
        logger.info("首次加载 SAM2 模型: %s (device=%s)，请稍候...", seg.sam2_model, seg.device)
        self._sam = Sam2Model.from_pretrained(seg.sam2_model).to(seg.device).eval()
        self._sam_proc = Sam2Processor.from_pretrained(seg.sam2_model)
        logger.info("SAM2 模型加载完成")

    def _segment_sam2(
        self,
        image: Image.Image,
        point: tuple[float, float] | None = None,
        input_boxes: tuple[int, int, int, int] | None = None,
    ) -> np.ndarray:
        """用 SAM2 生成 mask，点或 box 提示二选一"""
        self._ensure_sam()
        w, h = image.size
        pil_img = image.convert("RGB")

        # 构造输入：点提示用 input_points/input_labels，box 提示用 input_boxes
        kwargs = {}
        if input_boxes is not None:
            x1, y1, x2, y2 = input_boxes
            # 维度: [batch, n_boxes, 4]
            kwargs["input_boxes"] = [[[list(input_boxes)]]]
        else:
            px, py = point or (0.5, 0.5)
            # 维度: [batch, point_batch, n_points, 2]，坐标为绝对像素
            kwargs["input_points"] = [[[[px * w, py * h]]]]
            kwargs["input_labels"] = [[[1]]]  # 1 = 前景点

        inputs = self._sam_proc(images=pil_img, return_tensors="pt", **kwargs)
        inputs = {k: v.to(cfg.segmentation.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self._sam(**inputs)

        # post_process_masks 还原到原图尺寸: [batch, n_candidates, H, W]
        # 取 batch=0 并 squeeze 掉 batch 维 → [n_candidates, H, W]
        masks = self._sam_proc.post_process_masks(
            outputs.pred_masks.cpu(),
            inputs["original_sizes"].cpu(),
        )[0].squeeze(0)  # [1, 3, H, W] → [3, H, W]
        ious = outputs.iou_scores.cpu().numpy().flatten()  # [3]

        return self._pick_best_mask(masks, ious, w, h)

    @staticmethod
    def _pick_best_mask(masks, ious: np.ndarray, w: int, h: int) -> np.ndarray:
        """按 IoU 取最佳 mask，二值化到 uint8 0/255 并羽化

        masks: [n_candidates, H, W] float tensor
        ious: [n_candidates] float array
        """
        if masks.ndim == 3:
            n = masks.shape[0]
        else:
            n = 1
            masks = masks.unsqueeze(0)
        if n == 0 or len(ious) == 0:
            raise RuntimeError("SAM2 未输出 mask")
        best = int(np.argmax(ious[:n]))
        m = masks[best]
        mask = m.numpy() if hasattr(m, "numpy") else np.asarray(m)
        mask = (mask > 0.5).astype(np.uint8) * 255
        if mask.shape[:2] != (h, w):
            mask = cv2.resize(mask, (w, h))
        pct = np.count_nonzero(mask) / mask.size * 100
        logger.info("[segment] SAM2 mask: best_iou=%.3f, 覆盖 %.1f%% 区域", ious[best], pct)
        return ImageSegmenter._feather(mask)

    def _gdino_box(self, image: Image.Image, description: str):
        """用 GroundingDINO 检测，返回最高置信度 box 或 None"""
        if not description:
            return None
        if self._gdino is None:
            self._gdino = hf_pipeline(
                "zero-shot-object-detection",
                model=cfg.segmentation.gdino_model,
                device=-1,  # CPU
            )
        results = self._gdino(image, candidate_labels=[description])
        if not results:
            return None
        best = max(results, key=lambda r: r["score"])
        box = best["box"]  # {x,y,width,height} 或 [x1,y1,x2,y2]
        logger.info("GDINO 检测到 '%s' (score=%.2f)", description, best["score"])
        if isinstance(box, dict):
            return (int(box["x"]), int(box["y"]),
                    int(box["x"] + box["width"]), int(box["y"] + box["height"]))
        return tuple(int(v) for v in box[:4])

    # ── GrabCut 系列（降级方案）──────────────────────

    def _segment_grabcut(
        self, image: Image.Image, x: float, y: float, radius: float,
    ) -> np.ndarray:
        """GrabCut 降级：以坐标为中心生成 mask"""
        img = np.array(image.convert("RGB"))
        h, w = img.shape[:2]

        cx, cy = int(x * w), int(y * h)
        rw, rh = int(radius * w), int(radius * h)
        rect = (max(0, cx - rw), max(0, cy - rh), min(w, cx + rw), min(h, cy + rh))

        mask = np.zeros((h, w), np.uint8)
        bgd_model = np.zeros((1, 65), np.float64)
        fgd_model = np.zeros((1, 65), np.float64)

        try:
            cv2.grabCut(img, mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)
            result = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
        except cv2.error:
            result = np.zeros((h, w), np.uint8)
            cv2.ellipse(result, (cx, cy), (rw, rh), 0, 0, 360, 255, -1)

        return self._feather(result)

    def _segment_gdino_grabcut(
        self, image: Image.Image, description: str,
        x: float, y: float, radius: float,
    ) -> np.ndarray:
        """GroundingDINO 检测 + GrabCut 分割"""
        box = self._gdino_box(image, description)
        if box is None:
            logger.warning("GDINO 未检测到 '%s'，降级纯 GrabCut", description)
            return self._segment_grabcut(image, x, y, radius)

        img = np.array(image.convert("RGB"))
        h, w = img.shape[:2]
        mask = np.zeros((h, w), np.uint8)
        bgd_model = np.zeros((1, 65), np.float64)
        fgd_model = np.zeros((1, 65), np.float64)
        rect = box

        try:
            cv2.grabCut(img, mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)
            result = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
        except cv2.error:
            result = np.zeros((h, w), np.uint8)
            cv2.rectangle(result, (rect[0], rect[1]), (rect[2], rect[3]), 255, -1)

        return self._feather(result)

    @staticmethod
    def _feather(mask: np.ndarray) -> np.ndarray:
        """高斯羽化，复用于所有分割路径"""
        feather = cfg.processing.mask_feather
        if feather <= 0:
            return mask
        ksize = feather * 2 + 1
        return cv2.GaussianBlur(
            mask.astype(np.float32), (ksize, ksize), 0
        ).astype(np.uint8)
