"""agnes-2.0-flash 提供者"""

import base64
import json
import re
from io import BytesIO

import requests
from PIL import Image

from config import AgnesConfig
from .base import EditSuggestion, GlobalParams, LocalAdjustment, VLMProvider

_PROMPT = """你是专业修图师。分析这张照片，输出JSON修图建议。
{context}
参数规范：exposure_ev:-3到+3(float), white_balance_k:2000到10000(int),
contrast/highlights/shadows/saturation:-100到+100(int), 坐标0-1(float)。

输出格式：
{{"analysis":"分析","style":"风格","global":{{"exposure_ev":0.0,"white_balance_k":5500,"contrast":0,"highlights":0,"shadows":0,"saturation":0}},"local":[{{"description":"区域","x":0.5,"y":0.5,"type":"类型","exposure_ev":0.5,"temperature_shift":200,"reason":"理由"}}]}}
type 规则：brighten/darken/shadows/highlights(曝光型,看exposure_ev); warm/warmth/cool/cooling(色温型,看temperature_shift,暖正冷负,仅用于小区域<15%,大面积暖色请用全局white_balance_k); vignette(暗角,exposure_ev控制强度0.3~1.0); blur(背景模糊,exposure_ev控制模糊程度0.3~1.0)。只输出JSON。"""


class AgnesProvider(VLMProvider):
    """agnes-2.0-flash 云端提供者"""

    def __init__(self, config: AgnesConfig | None = None):
        self._cfg = config or AgnesConfig()

    @property
    def name(self) -> str:
        return "agnes"

    def analyze(self, image: Image.Image, context: str = "") -> EditSuggestion:
        img_b64 = self._image_to_base64(image)
        ctx_line = f"场景描述：{context}" if context else "无额外场景描述。"
        prompt = _PROMPT.format(context=ctx_line)

        payload = {
            "model": self._cfg.model,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {
                        "url": f"data:image/jpeg;base64,{img_b64}"
                    }},
                    {"type": "text", "text": prompt},
                ],
            }],
            "max_tokens": self._cfg.max_tokens,
            "temperature": self._cfg.temperature,
        }

        resp = requests.post(
            self._cfg.base_url,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self._cfg.api_key}",
            },
            json=payload,
            timeout=120,
        )
        resp.raise_for_status()
        data = resp.json()
        raw = data["choices"][0]["message"]["content"]

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

        gp = obj.get("global", {})
        return EditSuggestion(
            analysis=obj.get("analysis", ""),
            style=obj.get("style", ""),
            global_params=GlobalParams(
                exposure_ev=float(gp.get("exposure_ev", 0)),
                white_balance_k=int(gp.get("white_balance_k", 5500)),
                contrast=int(gp.get("contrast", 0)),
                highlights=int(gp.get("highlights", 0)),
                shadows=int(gp.get("shadows", 0)),
                saturation=int(gp.get("saturation", 0)),
            ),
            local_adjustments=[
                LocalAdjustment(
                    description=la.get("description", ""),
                    x=float(la.get("x", la.get("position", {}).get("x", 0.5))),
                    y=float(la.get("y", la.get("position", {}).get("y", 0.5))),
                    adjustment_type=la.get("type", "brighten"),
                    exposure_ev=float(la.get("exposure_ev", la.get("params", {}).get("exposure_ev", 0))),
                    temperature_shift=int(la.get("temperature_shift", 0)),
                    reason=la.get("reason", ""),
                )
                for la in obj.get("local", [])
            ],
            raw_response=raw,
            backend=self.name,
        )

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
