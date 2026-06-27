"""VLM 抽象基类和数据模型"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image


@dataclass
class GlobalParams:
    """全局修图参数"""
    exposure_ev: float = 0.0       # -3.0 ~ +3.0
    white_balance_k: int = 5500    # 2000 ~ 10000
    contrast: int = 0              # -100 ~ +100
    highlights: int = 0            # -100 ~ +100
    shadows: int = 0               # -100 ~ +100
    saturation: int = 0            # -100 ~ +100

    def clamp(self, spec) -> "GlobalParams":
        """将参数限制在规范范围内，并强制类型（int 字段转 int、exposure 转 float）

        VLM 可能返回浮点（如 contrast=10.5），若不强转会导致下游
        `f"{gp.contrast:+d}"` 等格式化崩溃，且与 dataclass 声明不符。
        """
        return GlobalParams(
            exposure_ev=float(max(spec.exposure_ev_min, min(spec.exposure_ev_max, self.exposure_ev))),
            white_balance_k=int(max(spec.wb_k_min, min(spec.wb_k_max, self.white_balance_k))),
            contrast=int(max(spec.adjust_min, min(spec.adjust_max, self.contrast))),
            highlights=int(max(spec.adjust_min, min(spec.adjust_max, self.highlights))),
            shadows=int(max(spec.adjust_min, min(spec.adjust_max, self.shadows))),
            saturation=int(max(spec.adjust_min, min(spec.adjust_max, self.saturation))),
        )


@dataclass
class LocalAdjustment:
    """局部调整参数"""
    description: str = ""          # 区域描述，如"前景草丛"
    x: float = 0.5                 # 相对坐标 0-1
    y: float = 0.5                 # 相对坐标 0-1
    adjustment_type: str = "brighten"  # brighten/darken/warm/warmth/cool/cooling/shadows/highlights
    exposure_ev: float = 0.0       # 局部曝光调整
    temperature_shift: int = 0     # 局部色温偏移（仅 warm/cool 类生效）
    reason: str = ""               # 调整理由

    def clamp(self, spec) -> "LocalAdjustment":
        """规范化局部调整参数：夹取范围 + 类型小写化

        对称于 GlobalParams.clamp。bridge 对每个 local 项调用，
        保证 VLM 返回的越界值（如 x=1.5）不会污染下游处理。
        """
        return LocalAdjustment(
            description=self.description,
            x=max(0.0, min(1.0, float(self.x))),
            y=max(0.0, min(1.0, float(self.y))),
            adjustment_type=self.adjustment_type.lower(),
            exposure_ev=float(max(spec.exposure_ev_min, min(spec.exposure_ev_max, self.exposure_ev))),
            temperature_shift=int(max(spec.adjust_min, min(spec.adjust_max, self.temperature_shift))),
            reason=self.reason,
        )


@dataclass
class EditSuggestion:
    """VLM 输出的完整修图建议"""
    analysis: str = ""             # 照片分析
    style: str = ""                # 推荐风格
    global_params: GlobalParams = field(default_factory=GlobalParams)
    local_adjustments: list[LocalAdjustment] = field(default_factory=list)
    raw_response: str = ""         # 原始响应
    backend: str = ""              # 使用的后端


class VLMProvider(ABC):
    """VLM 提供者抽象基类"""

    @abstractmethod
    def analyze(self, image: Image.Image, context: str = "") -> EditSuggestion:
        """分析图片并返回修图建议

        Args:
            image: PIL Image 对象
            context: 用户提供的场景描述

        Returns:
            EditSuggestion 修图建议
        """

    @property
    @abstractmethod
    def name(self) -> str:
        """提供者名称"""
