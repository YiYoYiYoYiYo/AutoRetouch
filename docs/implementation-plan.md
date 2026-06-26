# AI Beautify 实现计划

> 全自动 AI 修图与调色工作流

**目标：** 用户批量上传照片 → VLM 分析生成修图建议 → 自动分割生成 mask → 执行修图 → 批量导出

**架构：** VLM 理解层（GLM/agnes/Ollama）→ 分割层（GroundingDINO+SAM2）→ 处理层（numpy+OpenCV）→ UI（Gradio）

**技术栈：** Python 3.10+, Gradio, Pillow, numpy, OpenCV, torch, transformers, segment-anything-2

## 文件结构

```
AI Beautify/
├── pyproject.toml              # 项目配置和依赖
├── config.py                   # 全局配置（模型参数、路径、规范）
├── vlm/
│   ├── __init__.py
│   ├── base.py                 # VLMProvider 抽象基类
│   ├── glm_provider.py         # GLM-4.1V-Thinking-Flash
│   ├── agnes_provider.py       # agnes-2.0-flash
│   ├── ollama_provider.py      # Ollama 本地模型
│   └── bridge.py               # VLM Bridge 统一调度
├── segmentation/
│   ├── __init__.py
│   └── segmenter.py            # GroundingDINO + SAM2 分割
├── processor.py                # 图像处理引擎（全局+局部调整）
├── pipeline.py                 # 批量处理管线
├── app.py                      # Gradio 用户界面
└── tests/
    ├── test_vlm.py
    ├── test_processor.py
    └── test_pipeline.py
```

## 模块职责

### config.py
- VLM API 配置（endpoint、key、model name）
- 参数规范（EV范围、色温范围、调整值范围）
- 处理配置（输出格式、质量、尺寸限制）

### vlm/base.py
- `VLMProvider` 抽象基类：`analyze(image, context) -> EditSuggestion`
- `EditSuggestion` 数据类：analysis、style、global_params、local_adjustments

### vlm/glm_provider.py
- GLM-4.1V-Thinking-Flash API 调用
- thinking 模式控制
- 响应解析和参数规范化

### vlm/agnes_provider.py
- agnes-2.0-flash API 调用
- 兼容 OpenAI 格式

### vlm/ollama_provider.py
- Ollama REST API 调用
- 本地模型降级策略

### vlm/bridge.py
- `VLMBridge.analyze()`: 主力→降级→离线 三级切换
- 隐私开关控制

### segmentation/segmenter.py
- `ImageSegmenter.segment(description, image) -> mask`
- GroundingDINO: 文字→bounding box
- SAM2: bounding box→像素级 mask
- 降级方案：GrabCut（无模型时）

### processor.py
- `ImageProcessor.process(image, suggestion) -> processed_image`
- 全局调整：exposure、white_balance、contrast、highlights、shadows、saturation
- 局部调整：mask × 参数 → 区域修改
- 色彩空间转换（sRGB ↔ linear）

### pipeline.py
- `BatchPipeline.run(images, context, on_progress) -> results`
- RAW→JPEG 转换
- 批量 VLM 分析（并发控制）
- 批量分割+处理
- 导出 JPEG/HEIC

### app.py
- Gradio Blocks 界面
- 批量上传 + 场景描述
- VLM 建议预览（逐张）
- 用户确认/编辑
- 批量导出
