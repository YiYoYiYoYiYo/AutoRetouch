"""全局配置"""

from dataclasses import dataclass, field


@dataclass
class GLMConfig:
    """GLM-4.1V-Thinking-Flash 配置"""
    api_key: str = "0df3aede210a403e8a6ae5866f60dcab.URnNJpdXUxeYMOpW"
    base_url: str = "https://open.bigmodel.cn/api/paas/v4/chat/completions"
    model: str = "glm-4.1v-thinking-flash"
    max_tokens: int = 2048
    temperature: float = 0.3
    thinking_enabled: bool = True


@dataclass
class AgnesConfig:
    """agnes-2.0-flash 配置"""
    api_key: str = "sk-bCJKsioXvFlKHwyxTlUSsJcOn4uIOqL1K81dRZHJYdA3ClfO"
    base_url: str = "https://apihub.agnes-ai.com/v1/chat/completions"
    model: str = "agnes-2.0-flash"
    max_tokens: int = 2048
    temperature: float = 0.3


@dataclass
class OllamaConfig:
    """Ollama 本地模型配置"""
    base_url: str = "http://localhost:11434/api/chat"
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
    local_adjustment_radius: float = 0.15  # 局部调整默认半径(相对坐标)
    mask_feather: int = 30  # mask 羽化像素


@dataclass
class AppConfig:
    """应用总配置"""
    glm: GLMConfig = field(default_factory=GLMConfig)
    agnes: AgnesConfig = field(default_factory=AgnesConfig)
    ollama: OllamaConfig = field(default_factory=OllamaConfig)
    param_spec: ParamSpec = field(default_factory=ParamSpec)
    processing: ProcessingConfig = field(default_factory=ProcessingConfig)
    default_backend: str = "glm"  # glm / agnes / ollama
    max_concurrent: int = 5  # 最大并发数


# 全局配置实例
cfg = AppConfig()
