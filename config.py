"""全局配置"""

import os
from dataclasses import dataclass, field

from dotenv import load_dotenv

load_dotenv()


@dataclass
class GLMConfig:
    """GLM-4.1V-Thinking-Flash 配置"""
    api_key: str = field(default_factory=lambda: os.environ.get("GLM_API_KEY", ""))
    base_url: str = field(default_factory=lambda: os.environ.get("GLM_BASE_URL", "https://open.bigmodel.cn/api/paas/v4/chat/completions"))
    model: str = "glm-4.1v-thinking-flash"
    max_tokens: int = 2048
    temperature: float = 0.3
    thinking_enabled: bool = True


@dataclass
class AgnesConfig:
    """agnes-2.0-flash 配置"""
    api_key: str = field(default_factory=lambda: os.environ.get("AGNES_API_KEY", ""))
    base_url: str = field(default_factory=lambda: os.environ.get("AGNES_BASE_URL", "https://apihub.agnes-ai.com/v1/chat/completions"))
    model: str = "agnes-2.0-flash"
    max_tokens: int = 2048
    temperature: float = 0.3


@dataclass
class OllamaConfig:
    """Ollama 本地模型配置"""
    base_url: str = field(default_factory=lambda: os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434/api/chat"))
    model: str = "qwen2.5vl:3b"


@dataclass
class ParamSpec:
    """修图参数规范"""
    exposure_ev_min: float = -3.0
    exposure_ev_max: float = 3.0
    wb_k_min: int = 2000
    wb_k_max: int = 10000
    adjust_min: int = -100
    adjust_max: int = 100


@dataclass
class ProcessingConfig:
    """图像处理配置"""
    output_format: str = "jpeg"  # jpeg / heic
    jpeg_quality: int = 95
    max_dimension: int = 8192  # 最大边长
    local_adjustment_radius: float = 0.15  # 局部调整默认半径(相对坐标)，grabcut 降级时使用
    mask_feather: int = 30  # mask 羽化像素


@dataclass
class SegmentationConfig:
    """分割引擎配置"""
    enable_sam2: bool = True        # 默认用 SAM2，加载失败/未装依赖时自动降级 grabcut
    use_gdino: bool = True          # 开启后先用 GroundingDINO 按文字描述检测 box，再喂给 SAM2（更准但首图更慢）
    sam2_model: str = "facebook/sam2-hiera-tiny"   # HuggingFace 上的 SAM2 模型名
    gdino_model: str = "IDEA-Research/grounding-dino-tiny"
    device: str = "cpu"             # Intel Arc 的 IPEX-XPU 对 py3.13 不稳，统一 CPU 推理


@dataclass
class AppConfig:
    """应用总配置"""
    glm: GLMConfig = field(default_factory=GLMConfig)
    agnes: AgnesConfig = field(default_factory=AgnesConfig)
    ollama: OllamaConfig = field(default_factory=OllamaConfig)
    param_spec: ParamSpec = field(default_factory=ParamSpec)
    processing: ProcessingConfig = field(default_factory=ProcessingConfig)
    segmentation: SegmentationConfig = field(default_factory=SegmentationConfig)
    default_backend: str = field(default_factory=lambda: os.environ.get("DEFAULT_BACKEND", "glm"))  # glm / agnes / ollama
    max_concurrent: int = 5  # 最大并发数


# 全局配置实例
cfg = AppConfig()
