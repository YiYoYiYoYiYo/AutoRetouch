# 🎨 AutoRetouch

> Not a pixel fabricator, but a light guide — every photo deserves to keep its original soul

[中文](./README_zh.md)

AI-powered photo retouching workflow — batch upload, auto-analyze with VLM, generate professional editing parameters, and export in one click.

## ✨ Features

- **AI Analysis** — VLM vision models identify scenes, determine style, and generate professional retouching parameters
- **Local Adjustments** — Auto-segment image regions for independent brighten/darken/color grading
- **Batch Processing** — Batch upload, batch analyze, batch export — one click
- **Multi-Backend** — Cloud GLM (high quality) / Agnes (fast) / Local Ollama (offline & private)
- **RAW Support** — CR2/NEF/ARW/DNG auto-converted to JPEG
- **Multi-Format Export** — JPEG / HEIC

## 🚀 Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Configure API keys
cp .env.example .env
# Edit .env and fill in your API keys

# 3. Launch
python app.py

# 4. Open http://localhost:7860
```

### Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `GLM_API_KEY` | ✅ | GLM API key, [get it here](https://open.bigmodel.cn/) |
| `AGNES_API_KEY` | ✅ | Agnes API key |
| `SERVER_HOST` | ❌ | Server bind address, default `127.0.0.1` |
| `SERVER_PORT` | ❌ | Server port, default `7860` |

## 📸 Workflow

```
Upload → Describe Scene → AI Analyze → Review Suggestions → Export
```

1. **Upload** — JPEG/PNG/RAW, batch supported
2. **Describe** — e.g. "Travel photos from a garden, overcast weather"
3. **Choose Backend** — leave blank for auto-fallback (recommended)
4. **Click "Analyze"** — AI generates per-photo retouching suggestions
5. **Review & Click "Process & Export"** — batch output

## 🧠 VLM Backends

| Backend | Model | Speed | Quality | Requirement |
|---------|-------|-------|---------|-------------|
| `glm` | GLM-4.1V-Thinking-Flash | ~11s/photo | ⭐⭐⭐ | Network + API Key |
| `agnes` | Agnes-2.0-flash | ~5s/photo | ⭐⭐ | Network + API Key |
| `ollama` | qwen2.5vl:3b | ~3min/photo | ⭐ | Local Ollama |

**Auto-fallback:** glm → agnes → ollama

**Privacy mode:** select `ollama` manually — your photos never leave your machine.

## ⚙️ Parameter Specs

| Parameter | Range | Description |
|-----------|-------|-------------|
| `exposure_ev` | -3.0 ~ +3.0 | Exposure value (EV) |
| `white_balance_k` | 2000 ~ 10000 | White balance (K) |
| `contrast` | -100 ~ +100 | Contrast |
| `highlights` | -100 ~ +100 | Highlights |
| `shadows` | -100 ~ +100 | Shadows |
| `saturation` | -100 ~ +100 | Saturation |

## 🏗️ Project Structure

```
AutoRetouch/
├── app.py                  # Gradio UI
├── config.py               # Global config
├── pipeline.py             # Batch processing pipeline
├── processor.py            # Image processing engine
├── vlm/                    # VLM bridge layer
│   ├── base.py             # Data models
│   ├── glm_provider.py     # GLM cloud
│   ├── agnes_provider.py   # Agnes cloud
│   ├── ollama_provider.py  # Ollama local
│   └── bridge.py           # Unified dispatcher
├── segmentation/           # Image segmentation
│   └── segmenter.py        # GrabCut / GroundingDINO
├── tests/                  # Tests
└── docs/                   # Documentation
```

## 🔧 CLI Usage

```python
from pipeline import BatchPipeline, load_images
from pathlib import Path

images = load_images(["photo1.jpg", "photo2.jpg"])
pipe = BatchPipeline()

suggestions = pipe.analyze_batch(images, context="Travel landscape photos")
batch = pipe.process_batch(images, suggestions)
paths = pipe.export_batch(batch, Path("output"))
```

## 📋 Dependencies

**Core:** Python 3.10+, Gradio, Pillow, NumPy, OpenCV, Requests

**Optional:** rawpy (RAW support), pillow-heif (HEIC export), PyTorch + Transformers (GroundingDINO segmentation)

## 📄 License

MIT
