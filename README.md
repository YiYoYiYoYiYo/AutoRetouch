# 🎨 AutoRetouch

> 不做像素的篡改者，只做光影的引路人 — 每一张照片，都值得保留它最初的灵魂

全自动 AI 修图与调色工作流——批量上传照片，AI 自动分析并生成专业修图建议，一键批量出片。

## ✨ 功能特性

- **AI 智能分析** — VLM 视觉大模型识别场景、判断风格、生成专业修图参数
- **局部调整** — 自动分割图像区域，对指定区域进行独立调亮/调暗/调色
- **批量处理** — 批量上传、批量分析、批量导出，一键完成
- **多后端切换** — 云端 GLM（高质量）/ agnes（快速）/ 本地 Ollama（离线隐私）
- **RAW 支持** — 支持 CR2/NEF/ARW/DNG 等 RAW 格式自动转 JPEG
- **多格式导出** — JPEG / HEIC 输出

## 🚀 快速开始

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 配置 API Key
cp .env.example .env
# 编辑 .env，填入你的 API Key

# 3. 启动应用
python app.py

# 4. 浏览器打开 http://localhost:7860
```

### 环境变量

| 变量 | 必填 | 说明 |
|------|------|------|
| `GLM_API_KEY` | ✅ | GLM 服务密钥，[申请地址](https://open.bigmodel.cn/) |
| `AGNES_API_KEY` | ✅ | Agnes 服务密钥 |
| `SERVER_HOST` | ❌ | 服务器绑定地址，默认 `127.0.0.1` |
| `SERVER_PORT` | ❌ | 服务器端口，默认 `7860` |

## 📸 使用流程

```
上传照片 → 描述场景 → AI 分析 → 确认建议 → 批量导出
```

1. **上传照片** — 支持 JPEG/PNG/RAW，可批量上传
2. **填写场景描述** — 如"苏州园林旅行照，阴天拍摄"
3. **选择 VLM 后端** — 留空自动降级（推荐）
4. **点击"分析"** — AI 给出逐张修图建议
5. **确认后点击"处理并导出"** — 批量出片

## 🧠 VLM 后端

| 后端 | 模型 | 速度 | 质量 | 要求 |
|------|------|------|------|------|
| `glm` | GLM-4.1V-Thinking-Flash | ~11秒/张 | ⭐⭐⭐ | 网络 + API Key |
| `agnes` | agnes-2.0-flash | ~5秒/张 | ⭐⭐ | 网络 + API Key |
| `ollama` | qwen2.5vl:3b | ~3分钟/张 | ⭐ | 本地 Ollama |

**自动降级：** glm → agnes → ollama

**隐私模式：** 敏感照片手动选择 `ollama`，数据不出本机。

## ⚙️ 参数规范

| 参数 | 范围 | 说明 |
|------|------|------|
| `exposure_ev` | -3.0 ~ +3.0 | 曝光值（EV） |
| `white_balance_k` | 2000 ~ 10000 | 色温（K） |
| `contrast` | -100 ~ +100 | 对比度 |
| `highlights` | -100 ~ +100 | 高光 |
| `shadows` | -100 ~ +100 | 阴影 |
| `saturation` | -100 ~ +100 | 饱和度 |

## 🏗️ 项目结构

```
AI Beautify/
├── app.py                  # Gradio 用户界面
├── config.py               # 全局配置
├── pipeline.py             # 批量处理管线
├── processor.py            # 图像处理引擎
├── vlm/                    # VLM 桥接层
│   ├── base.py             # 数据模型
│   ├── glm_provider.py     # GLM 云端
│   ├── agnes_provider.py   # agnes 云端
│   ├── ollama_provider.py  # Ollama 本地
│   └── bridge.py           # 统一调度
├── segmentation/           # 图像分割
│   └── segmenter.py        # GrabCut / GroundingDINO
├── tests/                  # 测试
└── docs/                   # 文档
```

## 🔧 命令行使用

```python
from pipeline import BatchPipeline, load_images
from pathlib import Path

images = load_images(["photo1.jpg", "photo2.jpg"])
pipe = BatchPipeline()

suggestions = pipe.analyze_batch(images, context="旅行风景照")
batch = pipe.process_batch(images, suggestions)
paths = pipe.export_batch(batch, Path("output"))
```

## 📋 依赖

**核心：** Python 3.10+, Gradio, Pillow, NumPy, OpenCV, Requests

**可选：** rawpy (RAW 支持), pillow-heif (HEIC 导出), PyTorch + Transformers (GroundingDINO 分割)

## 📄 许可证

MIT
