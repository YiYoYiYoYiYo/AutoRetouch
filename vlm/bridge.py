"""VLM 统一调度桥接器"""

import logging

from PIL import Image

from config import cfg

from .agnes_provider import AgnesProvider
from .base import EditSuggestion, VLMProvider
from .glm_provider import GLMProvider
from .ollama_provider import OllamaProvider

logger = logging.getLogger(__name__)


class VLMBridge:
    """VLM 统一调度：主力 GLM → 降级 agnes → 离线 Ollama"""

    def __init__(self):
        self._providers: dict[str, VLMProvider] = {
            "glm": GLMProvider(cfg.glm),
            "agnes": AgnesProvider(cfg.agnes),
            "ollama": OllamaProvider(cfg.ollama),
        }
        self._fallback_order = ["glm", "agnes", "ollama"]

    def analyze(
        self,
        image: Image.Image,
        context: str = "",
        backend: str | None = None,
    ) -> EditSuggestion:
        """分析图片并返回修图建议

        Args:
            image: PIL Image
            context: 用户场景描述
            backend: 指定后端 (glm/agnes/ollama)，None 则按降级顺序尝试

        Returns:
            EditSuggestion
        """
        if backend:
            return self._try_provider(backend, image, context)

        last_error = None
        for name in self._fallback_order:
            try:
                result = self._try_provider(name, image, context)
                logger.info("VLM 分析完成，使用后端: %s", name)
                return result
            except Exception as e:
                logger.warning("VLM 后端 %s 失败: %s，尝试下一个", name, e)
                last_error = e

        raise RuntimeError(f"所有 VLM 后端均失败，最后错误: {last_error}")

    def _try_provider(
        self, name: str, image: Image.Image, context: str
    ) -> EditSuggestion:
        provider = self._providers.get(name)
        if not provider:
            raise ValueError(f"未知的 VLM 后端: {name}")
        suggestion = provider.analyze(image, context)
        # 参数规范化：全局 + 局部都夹取范围（对称处理，防止 VLM 越界值污染下游）
        suggestion.global_params = suggestion.global_params.clamp(cfg.param_spec)
        suggestion.local_adjustments = [
            la.clamp(cfg.param_spec) for la in suggestion.local_adjustments
        ]
        return suggestion

    @property
    def available_backends(self) -> list[str]:
        return list(self._providers.keys())
