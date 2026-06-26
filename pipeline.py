"""批量处理管线：VLM 分析 → 分割 → 处理 → 导出"""

import logging
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image

from config import cfg
from processor import ImageProcessor
from segmentation.segmenter import ImageSegmenter
from vlm.base import EditSuggestion
from vlm.bridge import VLMBridge

logger = logging.getLogger(__name__)


@dataclass
class ProcessResult:
    """单张图片处理结果"""
    source_path: str
    original: Image.Image
    processed: Image.Image
    suggestion: EditSuggestion
    success: bool = True
    error: str = ""


@dataclass
class BatchResult:
    """批量处理结果"""
    results: list[ProcessResult] = field(default_factory=list)
    total: int = 0
    success_count: int = 0
    fail_count: int = 0


class BatchPipeline:
    """批量修图管线"""

    def __init__(self):
        self._bridge = VLMBridge()
        self._segmenter = ImageSegmenter()
        self._processor = ImageProcessor(self._segmenter)

    def analyze_batch(
        self,
        images: list[tuple[str, Image.Image]],
        context: str = "",
        backend: str | None = None,
        on_progress: Callable | None = None,
    ) -> list[EditSuggestion]:
        """批量 VLM 分析（带并发控制）

        Args:
            images: [(文件名, PIL Image), ...]
            context: 场景描述
            backend: 指定后端
            on_progress: 进度回调 (current, total, filename)

        Returns:
            list[EditSuggestion]
        """
        suggestions = []
        total = len(images)

        def _analyze_one(idx: int, name: str, img: Image.Image):
            try:
                result = self._bridge.analyze(img, context, backend)
                if on_progress:
                    on_progress(idx + 1, total, name)
                return idx, result
            except Exception as e:
                logger.error("分析 %s 失败: %s", name, e)
                return idx, EditSuggestion(analysis=f"分析失败: {e}", backend="error")

        with ThreadPoolExecutor(max_workers=cfg.max_concurrent) as pool:
            futures = [
                pool.submit(_analyze_one, i, name, img)
                for i, (name, img) in enumerate(images)
            ]
            results = [None] * total
            for future in as_completed(futures):
                idx, result = future.result()
                results[idx] = result

        return results

    def process_batch(
        self,
        images: list[tuple[str, Image.Image]],
        suggestions: list[EditSuggestion],
        on_progress: Callable | None = None,
    ) -> BatchResult:
        """批量处理图片

        Args:
            images: [(文件名, PIL Image), ...]
            suggestions: 对应的修图建议
            on_progress: 进度回调 (current, total, filename)

        Returns:
            BatchResult
        """
        batch = BatchResult(total=len(images))

        for i, ((name, original), suggestion) in enumerate(zip(images, suggestions)):
            try:
                processed = self._processor.process(original, suggestion)
                batch.results.append(ProcessResult(
                    source_path=name,
                    original=original,
                    processed=processed,
                    suggestion=suggestion,
                ))
                batch.success_count += 1
            except Exception as e:
                logger.error("处理 %s 失败: %s", name, e)
                batch.results.append(ProcessResult(
                    source_path=name,
                    original=original,
                    processed=original,
                    suggestion=suggestion,
                    success=False,
                    error=str(e),
                ))
                batch.fail_count += 1

            if on_progress:
                on_progress(i + 1, len(images), name)

        return batch

    def export(
        self,
        result: ProcessResult,
        output_dir: Path,
        fmt: str | None = None,
    ) -> Path:
        """导出单张处理后的图片

        Args:
            result: 处理结果
            output_dir: 输出目录
            fmt: 输出格式 (jpeg/heic)，None 使用配置

        Returns:
            输出文件路径
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        fmt = fmt or cfg.processing.output_format
        stem = Path(result.source_path).stem
        output_path = output_dir / f"{stem}_edited.{fmt}"

        if fmt == "heic":
            try:
                import pillow_heif
                result.processed.save(str(output_path), format="HEIF")
            except ImportError:
                logger.warning("pillow-heif 未安装，回退到 JPEG")
                output_path = output_dir / f"{stem}_edited.jpeg"
                result.processed.save(
                    str(output_path),
                    format="JPEG",
                    quality=cfg.processing.jpeg_quality,
                )
        else:
            result.processed.save(
                str(output_path),
                format="JPEG",
                quality=cfg.processing.jpeg_quality,
            )

        return output_path

    def export_batch(
        self,
        batch: BatchResult,
        output_dir: Path,
        fmt: str | None = None,
        on_progress: Callable | None = None,
    ) -> list[Path]:
        """批量导出"""
        paths = []
        for i, result in enumerate(batch.results):
            if result.success:
                path = self.export(result, output_dir, fmt)
                paths.append(path)
            if on_progress:
                on_progress(i + 1, len(batch.results), result.source_path)
        return paths


def load_images(paths: list[str | Path]) -> list[tuple[str, Image.Image]]:
    """加载图片文件列表

    Args:
        paths: 图片文件路径列表

    Returns:
        [(文件名, PIL Image), ...]
    """
    result = []
    for p in paths:
        p = Path(p)
        try:
            if p.suffix.lower() in (".cr2", ".nef", ".arw", ".dng", ".raf", ".orf"):
                # RAW 文件先转 JPEG
                import rawpy
                with rawpy.imread(str(p)) as raw:
                    rgb = raw.postprocess()
                img = Image.fromarray(rgb)
            else:
                img = Image.open(p).convert("RGB")
            result.append((str(p), img))
        except Exception as e:
            logger.error("加载 %s 失败: %s", p, e)
    return result
