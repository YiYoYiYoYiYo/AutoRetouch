"""图像处理引擎测试"""

import numpy as np
from PIL import Image

from processor import ImageProcessor
from vlm.base import EditSuggestion, GlobalParams, LocalAdjustment


def make_test_image(w=200, h=200) -> Image.Image:
    """创建测试图片"""
    arr = np.random.randint(50, 200, (h, w, 3), dtype=np.uint8)
    return Image.fromarray(arr)


class _FakeSeg:
    """注入固定圆形 mask 的假分割器，绕过 torch 依赖"""
    def segment(self, image, description, x=0.5, y=0.5, radius=0.15):
        w, h = image.size
        cx, cy = int(x * w), int(y * h)
        r = int(radius * min(w, h))
        yy, xx = np.ogrid[:h, :w]
        dist = ((xx - cx) ** 2 + (yy - cy) ** 2) ** 0.5
        mask = np.where(dist <= r, 255, 0).astype(np.uint8)
        return mask


def test_global_exposure():
    """测试全局曝光调整"""
    proc = ImageProcessor()
    img = make_test_image()
    suggestion = EditSuggestion(
        global_params=GlobalParams(exposure_ev=1.0),
    )
    result = proc.process(img, suggestion)
    orig_arr = np.array(img).astype(float)
    result_arr = np.array(result).astype(float)
    # 曝光 +1EV 应该整体变亮
    assert result_arr.mean() > orig_arr.mean(), "曝光 +1EV 应该使图片变亮"


def test_global_saturation():
    """测试全局饱和度调整"""
    proc = ImageProcessor()
    img = make_test_image()
    suggestion = EditSuggestion(
        global_params=GlobalParams(saturation=50),
    )
    result = proc.process(img, suggestion)
    assert result.size == img.size, "输出尺寸应与输入一致"


def test_local_brighten():
    """测试局部调亮"""
    proc = ImageProcessor()
    img = make_test_image()
    suggestion = EditSuggestion(
        global_params=GlobalParams(),
        local_adjustments=[
            LocalAdjustment(
                description="中心区域",
                x=0.5, y=0.5,
                adjustment_type="brighten",
                exposure_ev=1.0,
            ),
        ],
    )
    result = proc.process(img, suggestion)
    assert result.size == img.size, "输出尺寸应与输入一致"


def test_no_adjustment():
    """测试无调整（参数全零）"""
    proc = ImageProcessor()
    img = make_test_image()
    suggestion = EditSuggestion()
    result = proc.process(img, suggestion)
    orig_arr = np.array(img)
    result_arr = np.array(result)
    diff = np.abs(orig_arr.astype(float) - result_arr.astype(float)).mean()
    assert diff < 1.0, f"无调整时差异应极小，实际: {diff}"


# ── 局部调整专项测试（用 _FakeSeg 注入固定 mask）──────────

def test_local_brighten_no_color_shift():
    """锁 bug 2 修复：brighten 类型不应改变 R/B 通道均值（只改亮度）"""
    proc = ImageProcessor()
    proc._segmenter = _FakeSeg()
    img = make_test_image()
    suggestion = EditSuggestion(
        global_params=GlobalParams(),
        local_adjustments=[LocalAdjustment(
            x=0.5, y=0.5, adjustment_type="brighten", exposure_ev=1.0,
        )],
    )
    result = proc.process(img, suggestion)
    orig = np.array(img).astype(float) / 255.0
    res = np.array(result).astype(float) / 255.0
    # 在 mask 区域内检查：brighten 应使整体变亮但 R/B 比例不变
    # 用 FakeSeg 圆形 mask 中心 30% 区域采样
    h, w = img.size[1], img.size[0]
    cx, cy = w // 2, h // 2
    r = int(0.15 * min(w, h))
    yy, xx = np.ogrid[:h, :w]
    inner = ((xx - cx) ** 2 + (yy - cy) ** 2) <= (r * 0.5) ** 2
    # brighten 不应造成 R/B 通道的偏移（只通过 gamma 改亮度）
    orig_rb_diff = orig[inner, 0].mean() - orig[inner, 2].mean()
    res_rb_diff = res[inner, 0].mean() - res[inner, 2].mean()
    assert abs(res_rb_diff - orig_rb_diff) < 0.03, (
        f"brighten 不应改变 R-B 差值，原始差={orig_rb_diff:.4f}，处理后差={res_rb_diff:.4f}"
    )


def test_local_warm_uses_temperature_shift():
    """warm + temperature_shift=400 应使 R 升 B 降"""
    proc = ImageProcessor()
    proc._segmenter = _FakeSeg()
    img = make_test_image()
    suggestion = EditSuggestion(
        global_params=GlobalParams(),
        local_adjustments=[LocalAdjustment(
            x=0.5, y=0.5, adjustment_type="warm",
            temperature_shift=400,
        )],
    )
    result = proc.process(img, suggestion)
    orig = np.array(img).astype(float) / 255.0
    res = np.array(result).astype(float) / 255.0
    h, w = img.size[1], img.size[0]
    cx, cy = w // 2, h // 2
    r = int(0.15 * min(w, h))
    yy, xx = np.ogrid[:h, :w]
    inner = ((xx - cx) ** 2 + (yy - cy) ** 2) <= (r * 0.5) ** 2
    r_diff = res[inner, 0].mean() - orig[inner, 0].mean()
    b_diff = res[inner, 2].mean() - orig[inner, 2].mean()
    assert r_diff > 0.01, f"warm 应使 R 通道升高，实际 R 差值={r_diff:.4f}"
    assert b_diff < -0.01, f"warm 应使 B 通道降低，实际 B 差值={b_diff:.4f}"


def test_local_warm_default_when_zero():
    """warm + temperature_shift=0 应仍有可见暖偏移（默认 0.20）"""
    proc = ImageProcessor()
    proc._segmenter = _FakeSeg()
    img = make_test_image()
    suggestion = EditSuggestion(
        global_params=GlobalParams(),
        local_adjustments=[LocalAdjustment(
            x=0.5, y=0.5, adjustment_type="warm",
            temperature_shift=0,  # 0 时应走默认 0.20
        )],
    )
    result = proc.process(img, suggestion)
    orig = np.array(img).astype(float) / 255.0
    res = np.array(result).astype(float) / 255.0
    h, w = img.size[1], img.size[0]
    cx, cy = w // 2, h // 2
    r = int(0.15 * min(w, h))
    yy, xx = np.ogrid[:h, :w]
    inner = ((xx - cx) ** 2 + (yy - cy) ** 2) <= (r * 0.5) ** 2
    r_diff = res[inner, 0].mean() - orig[inner, 0].mean()
    b_diff = res[inner, 2].mean() - orig[inner, 2].mean()
    assert r_diff > 0.01, f"warm 默认偏移应使 R 升高，实际={r_diff:.4f}"
    assert b_diff < -0.01, f"warm 默认偏移应使 B 降低，实际={b_diff:.4f}"


def test_local_unknown_type_noop():
    """未知 type 应该对该区域无变化（mask 区域不变）"""
    proc = ImageProcessor()
    proc._segmenter = _FakeSeg()
    img = make_test_image()
    suggestion = EditSuggestion(
        global_params=GlobalParams(),
        local_adjustments=[LocalAdjustment(
            x=0.5, y=0.5, adjustment_type="nonexistent_type",
        )],
    )
    result = proc.process(img, suggestion)
    orig = np.array(img)
    res = np.array(result)
    diff = np.abs(orig.astype(float) - res.astype(float)).mean()
    assert diff < 1.0, f"未知类型应无变化，差异应极小，实际: {diff}"


def test_area_scale_small_mask():
    """小面积 mask 不衰减，大面积 mask 衰减"""
    assert ImageProcessor._area_scale(0.10) == 1.0, "10% 不应衰减"
    assert ImageProcessor._area_scale(0.15) == 1.0, "15% 不应衰减"
    s50 = ImageProcessor._area_scale(0.50)
    assert s50 == 0.3, f"50% 应衰减到 0.3，实际: {s50}"
    s30 = ImageProcessor._area_scale(0.30)
    assert 0.3 < s30 < 1.0, f"30% 应在 0.3~1.0 之间，实际: {s30}"


class _BigFakeSeg:
    """覆盖全图的假分割器（用于 vignette/blur 等全图效果测试）"""
    def segment(self, image, description, x=0.5, y=0.5, radius=0.15):
        w, h = image.size
        return np.full((h, w), 255, dtype=np.uint8)


def test_vignette_darkens_edges():
    """vignette 全局效果应使边缘变暗，中心基本不变"""
    proc = ImageProcessor()
    img = make_test_image()
    suggestion = EditSuggestion(
        global_params=GlobalParams(vignette=0.8),
    )
    result = proc.process(img, suggestion)
    orig = np.array(img).astype(float) / 255.0
    res = np.array(result).astype(float) / 255.0
    # 边缘区域应变暗
    edge = res[0:10, 0:10, :].mean()
    orig_edge = orig[0:10, 0:10, :].mean()
    assert edge < orig_edge, f"vignette 边缘应变暗，原始={orig_edge:.3f}，处理后={edge:.3f}"


def test_blur_softens_edges():
    """blur 全局效果应使远离中心的区域变模糊（方差降低）"""
    proc = ImageProcessor()
    img = make_test_image()
    suggestion = EditSuggestion(
        global_params=GlobalParams(blur=0.5),
    )
    result = proc.process(img, suggestion)
    orig = np.array(img).astype(float) / 255.0
    res = np.array(result).astype(float) / 255.0
    # 边缘区域的局部方差应降低（模糊了）
    def local_var(arr, y, x, sz=20):
        return arr[y:y+sz, x:x+sz, :].var()
    orig_var = local_var(orig, 0, 0)
    res_var = local_var(res, 0, 0)
    assert res_var < orig_var, f"blur 边缘方差应降低，原始={orig_var:.5f}，处理后={res_var:.5f}"


if __name__ == "__main__":
    test_global_exposure()
    print("✅ test_global_exposure")
    test_global_saturation()
    print("✅ test_global_saturation")
    test_local_brighten()
    print("✅ test_local_brighten")
    test_no_adjustment()
    print("✅ test_no_adjustment")
    test_local_brighten_no_color_shift()
    print("✅ test_local_brighten_no_color_shift")
    test_local_warm_uses_temperature_shift()
    print("✅ test_local_warm_uses_temperature_shift")
    test_local_warm_default_when_zero()
    print("✅ test_local_warm_default_when_zero")
    test_local_unknown_type_noop()
    print("✅ test_local_unknown_type_noop")
    test_area_scale_small_mask()
    print("✅ test_area_scale_small_mask")
    test_vignette_darkens_edges()
    print("✅ test_vignette_darkens_edges")
    test_blur_softens_edges()
    print("✅ test_blur_softens_edges")
    print("\n全部测试通过！")
