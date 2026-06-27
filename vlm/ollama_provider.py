"""Ollama 本地模型提供者"""

import base64
import json
import re
from io import BytesIO

import requests
from PIL import Image

from config import OllamaConfig
from .base import EditSuggestion, GlobalParams, LocalAdjustment, VLMProvider

_PROMPT = """你是专业修图师。分析这张照片，输出JSON修图建议。
{context}
参数规范：exposure_ev:-3到+3(float), white_balance_k:2000到10000(int),
contrast/highlights/shadows/saturation:-100到+100(int), 坐标0-1(float)。

输出格式：
{{"analysis":"分析","style":"风格","global":{{"exposure_ev":0.0,"white_balance_k":5500,"contrast":0,"highlights":0,"shadows":0,"saturation":0,"vignette":0.0,"blur":0.0}},"local":[{{"description":"区域","x":0.5,"y":0.5,"type":"类型","exposure_ev":0.5,"temperature_shift":200,"reason":"理由"}}]}}
全局效果：vignette(暗角0~1,越大越明显); blur(镜头模糊0~1,以画面中心为焦点)。局部type：brighten/darken/shadows/highlights(曝光型,看exposure_ev); warm/warmth/cool/cooling(色温型,看temperature_shift,暖正冷负,仅用于小区域<15%,大面积暖色请用全局white_balance_k)。只输出JSON。"""


class OllamaProvider(VLMProvider):
    """Ollama 本地模型提供者"""

    def __init__(self, config: OllamaConfig | None = None):
        self._cfg = config or OllamaConfig()

    @property
    def name(self) -> str:
        return "ollama"

    def analyze(self, image: Image.Image, context: str = "") -> EditSuggestion:
        img_b64 = self._image_to_base64(image)
        ctx_line = f"场景描述：{context}" if context else "无额外场景描述。"
        prompt = _PROMPT.format(context=ctx_line)

        payload = {
            "model": self._cfg.model,
            "messages": [{
                "role": "user",
                "content": prompt,
                "images": [img_b64],
            }],
            "stream": False,
        }

        resp = requests.post(
            self._cfg.base_url,
            json=payload,
            timeout=300,
        )
        resp.raise_for_status()
        data = resp.json()
        raw = data["message"]["content"]

        return self._parse(raw)

    def _parse(self, raw: str) -> EditSuggestion:
        json_str = self._extract_json(raw)
        try:
            obj = json.loads(json_str)
        except json.JSONDecodeError:
            return EditSuggestion(
                analysis="解析失败",
                raw_response=raw,
                backend=self.name,
            )

        gp = obj.get("global", obj.get("global_adjustments", {}))
        return EditSuggestion(
            analysis=obj.get("analysis", obj.get("照片分析文字", "")),
            style=obj.get("style", obj.get("风格推荐", "")),
            global_params=GlobalParams(
                exposure_ev=float(gp.get("exposure_ev", gp.get("曝光EV值", 0))),
                white_balance_k=int(gp.get("white_balance_k", gp.get("色温K", 5500))),
                contrast=int(gp.get("contrast", gp.get("对比度", 0))),
                highlights=int(gp.get("highlights", gp.get("高光", 0))),
                shadows=int(gp.get("shadows", gp.get("阴影", 0))),
                saturation=int(gp.get("saturation", gp.get("饱和度", 0))),
            ),
            local_adjustments=self._parse_local(obj),
            raw_response=raw,
            backend=self.name,
        )

    def _parse_local(self, obj: dict) -> list[LocalAdjustment]:
        """解析本地模型可能的各种输出格式"""
        locals_ = obj.get("local", obj.get("local_adjustments", []))
        if isinstance(locals_, dict):
            # 处理 {"regions_to_lighten": [...], "regions_to_darken": [...]} 格式
            result = []
            for region in locals_.get("regions_to_lighten", []):
                result.append(LocalAdjustment(
                    x=float(region.get("x", region.get("coordinate", [0.5, 0.5])[0])),
                    y=float(region.get("y", region.get("coordinate", [0.5, 0.5])[1])),
                    adjustment_type="brighten",
                ))
            for region in locals_.get("regions_to_darken", []):
                result.append(LocalAdjustment(
                    x=float(region.get("x", region.get("coordinate", [0.5, 0.5])[0])),
                    y=float(region.get("y", region.get("coordinate", [0.5, 0.5])[1])),
                    adjustment_type="darken",
                ))
            return result
        return [
            LocalAdjustment(
                description=la.get("description", la.get("区域描述", "")),
                x=float(la.get("x", la.get("relative_position", la.get("相对坐标", [0.5, 0.5]))[0] if isinstance(la.get("relative_position", la.get("相对坐标")), list) else la.get("position", {}).get("x", 0.5))),
                y=float(la.get("y", la.get("relative_position", la.get("相对坐标", [0.5, 0.5]))[1] if isinstance(la.get("relative_position", la.get("相对坐标")), list) else la.get("position", {}).get("y", 0.5))),
                adjustment_type=la.get("type", la.get("调整类型", la.get("type", "brighten"))),
                exposure_ev=float(la.get("exposure_ev", la.get("params", {}).get("exposure_ev", 0))),
                reason=la.get("reason", ""),
            )
            for la in (locals_ if isinstance(locals_, list) else [])
        ]

    @staticmethod
    def _extract_json(text: str) -> str:
        m = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
        if m:
            return m.group(1).strip()
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if m:
            return m.group(0)
        return text.strip()

    @staticmethod
    def _image_to_base64(image: Image.Image) -> str:
        buf = BytesIO()
        image.save(buf, format="JPEG", quality=90)
        return base64.b64encode(buf.getvalue()).decode()
