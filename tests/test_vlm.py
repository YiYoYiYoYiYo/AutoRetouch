"""VLM 解析测试（不需要实际 API 调用）"""

import json

from config import cfg
from vlm.base import GlobalParams, LocalAdjustment
from vlm.glm_provider import GLMProvider
from vlm.agnes_provider import AgnesProvider
from vlm.ollama_provider import OllamaProvider


def test_glm_parse_json():
    """测试 GLM 解析标准 JSON"""
    raw = json.dumps({
        "analysis": "测试照片",
        "style": "日系清新",
        "global": {
            "exposure_ev": 0.3,
            "white_balance_k": 5800,
            "contrast": 10,
            "highlights": -20,
            "shadows": 15,
            "saturation": 5,
        },
        "local": [{
            "description": "前景",
            "x": 0.3, "y": 0.8,
            "type": "brighten",
            "exposure_ev": 0.5,
            "reason": "提亮前景",
        }],
    }, ensure_ascii=False)

    provider = GLMProvider()
    result = provider._parse(raw)
    assert result.analysis == "测试照片"
    assert result.style == "日系清新"
    assert result.global_params.exposure_ev == 0.3
    assert result.global_params.white_balance_k == 5800
    assert len(result.local_adjustments) == 1
    assert result.local_adjustments[0].description == "前景"


def test_glm_parse_markdown():
    """测试 GLM 解析 markdown 代码块"""
    raw = '''```json
{
  "analysis": "风景照",
  "style": "自然风光",
  "global": {"exposure_ev": 0, "white_balance_k": 5500, "contrast": 0, "highlights": 0, "shadows": 0, "saturation": 0},
  "local": []
}
```'''
    provider = GLMProvider()
    result = provider._parse(raw)
    assert result.analysis == "风景照"
    assert len(result.local_adjustments) == 0


def test_ollama_parse_chinese_keys():
    """测试 Ollama 解析中文键名"""
    raw = json.dumps({
        "照片分析文字": "寺庙建筑",
        "风格推荐": "古典庄严",
        "global_adjustments": {
            "曝光EV值": -0.5,
            "色温K": 5200,
            "对比度": 10,
            "高光": -20,
            "阴影": 20,
            "饱和度": 15,
        },
        "local_adjustments": [{
            "区域描述": "金色屋顶",
            "相对坐标": [0.5, 0.3],
            "调整类型": "brighten",
        }],
    }, ensure_ascii=False)

    provider = OllamaProvider()
    result = provider._parse(raw)
    assert result.analysis == "寺庙建筑"
    assert result.style == "古典庄严"
    assert result.global_params.exposure_ev == -0.5
    assert result.global_params.white_balance_k == 5200


def test_ollama_parse_regions_dict():
    """测试 Ollama 解析 dict 格式的 local_adjustments"""
    raw = json.dumps({
        "global_adjustments": {"曝光EV值": 0, "色温K": 5500, "对比度": 0, "高光": 0, "阴影": 0, "饱和度": 0},
        "local_adjustments": {
            "regions_to_lighten": [{"x": 200, "y": 700}],
            "regions_to_darken": [{"x": 500, "y": 600}],
        },
    }, ensure_ascii=False)

    provider = OllamaProvider()
    result = provider._parse(raw)
    assert len(result.local_adjustments) == 2


def test_local_clamp():
    """测试 LocalAdjustment.clamp 对越界坐标的夹取"""
    la = LocalAdjustment(x=1.5, y=-0.3, exposure_ev=5.0, temperature_shift=200)
    clamped = la.clamp(cfg.param_spec)
    assert clamped.x == 1.0, f"x 应夹到 1.0，实际: {clamped.x}"
    assert clamped.y == 0.0, f"y 应夹到 0.0，实际: {clamped.y}"
    assert clamped.exposure_ev == 3.0, f"exposure_ev 应夹到 3.0，实际: {clamped.exposure_ev}"
    assert clamped.temperature_shift == 100, f"temperature_shift 应夹到 100，实际: {clamped.temperature_shift}"


def test_global_clamp_int():
    """测试 GlobalParams.clamp 对浮点字段强转 int"""
    gp = GlobalParams(contrast=10.5, highlights=-20.7, shadows=15.3, saturation=5.9)
    clamped = gp.clamp(cfg.param_spec)
    assert isinstance(clamped.contrast, int), f"contrast 应为 int，实际: {type(clamped.contrast)}"
    assert clamped.contrast == 10, f"contrast=10.5 → 10，实际: {clamped.contrast}"
    assert clamped.highlights == -20, f"highlights=-20.7 → -20，实际: {clamped.highlights}"
    assert clamped.shadows == 15, f"shadows=15.3 → 15，实际: {clamped.shadows}"


def test_warm_temperature_shift_parse():
    """测试解析含 temperature_shift 的 warm 局部调整"""
    raw = json.dumps({
        "analysis": "暖色测试",
        "style": "温暖",
        "global": {"exposure_ev": 0, "white_balance_k": 5500, "contrast": 0,
                   "highlights": 0, "shadows": 0, "saturation": 0},
        "local": [{"description": "天空", "x": 0.5, "y": 0.3,
                   "type": "warm", "exposure_ev": 0, "temperature_shift": 300, "reason": "加暖"}],
    }, ensure_ascii=False)
    provider = GLMProvider()
    result = provider._parse(raw)
    assert len(result.local_adjustments) == 1
    la = result.local_adjustments[0]
    assert la.adjustment_type == "warm"
    assert la.temperature_shift == 300, f"temperature_shift=300，实际: {la.temperature_shift}"


if __name__ == "__main__":
    test_glm_parse_json()
    print("✅ test_glm_parse_json")
    test_glm_parse_markdown()
    print("✅ test_glm_parse_markdown")
    test_ollama_parse_chinese_keys()
    print("✅ test_ollama_parse_chinese_keys")
    test_ollama_parse_regions_dict()
    print("✅ test_ollama_parse_regions_dict")
    test_local_clamp()
    print("✅ test_local_clamp")
    test_global_clamp_int()
    print("✅ test_global_clamp_int")
    test_warm_temperature_shift_parse()
    print("✅ test_warm_temperature_shift_parse")
    print("\n全部测试通过！")
