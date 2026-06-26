"""图像处理引擎测试"""

import numpy as np
from PIL import Image

from processor import ImageProcessor
from vlm.base import EditSuggestion, GlobalParams, LocalAdjustment


def make_test_image(w=200, h=200) -> Image.Image:
    """创建测试图片"""
    arr = np.random.randint(50, 200, (h, w, 3), dtype=np.uint8)
    return Image.fromarray(arr)


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


if __name__ == "__main__":
    test_global_exposure()
    print("✅ test_global_exposure")
    test_global_saturation()
    print("✅ test_global_saturation")
    test_local_brighten()
    print("✅ test_local_brighten")
    test_no_adjustment()
    print("✅ test_no_adjustment")
    print("\n全部测试通过！")
