"""AI Beautify - Gradio 用户界面"""

import json
import logging
import tempfile
from pathlib import Path

import gradio as gr

from config import cfg
from pipeline import BatchPipeline, load_images
from vlm.base import EditSuggestion

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# 全局管线实例
pipeline = BatchPipeline()


def format_suggestion(s: EditSuggestion) -> str:
    """格式化修图建议为可读文本"""
    gp = s.global_params
    lines = [
        f"## 📷 分析\n{s.analysis}",
        f"## 🎨 风格\n{s.style}",
        "## 🌐 全局参数",
        f"- 曝光: {gp.exposure_ev:+.2f} EV",
        f"- 色温: {gp.white_balance_k}K",
        f"- 对比度: {gp.contrast:+d}",
        f"- 高光: {gp.highlights:+d}",
        f"- 阴影: {gp.shadows:+d}",
        f"- 饱和度: {gp.saturation:+d}",
    ]
    if s.local_adjustments:
        lines.append("## 🎯 局部调整")
        for i, la in enumerate(s.local_adjustments, 1):
            lines.append(
                f"{i}. **{la.description}** ({la.x:.2f}, {la.y:.2f}) "
                f"- {la.adjustment_type}: {la.exposure_ev:+.2f}EV"
            )
            if la.reason:
                lines.append(f"   > {la.reason}")
    else:
        lines.append("## 🎯 局部调整\n无需局部调整")
    lines.append(f"\n---\n*后端: {s.backend}*")
    return "\n".join(lines)


def analyze_images(files, context, backend):
    """批量分析图片"""
    if not files:
        return "请上传图片", "", "{}"

    images = load_images([f.name for f in files])
    if not images:
        return "图片加载失败", "", "{}"

    # 只分析第一张用于预览
    name, img = images[0]
    suggestion = pipeline._bridge.analyze(img, context, backend or None)

    # 格式化显示
    display = format_suggestion(suggestion)

    # JSON 数据
    json_data = json.dumps({
        "analysis": suggestion.analysis,
        "style": suggestion.style,
        "global_params": {
            "exposure_ev": suggestion.global_params.exposure_ev,
            "white_balance_k": suggestion.global_params.white_balance_k,
            "contrast": suggestion.global_params.contrast,
            "highlights": suggestion.global_params.highlights,
            "shadows": suggestion.global_params.shadows,
            "saturation": suggestion.global_params.saturation,
        },
        "local_adjustments": [
            {
                "description": la.description,
                "position": {"x": la.x, "y": la.y},
                "type": la.adjustment_type,
                "exposure_ev": la.exposure_ev,
                "reason": la.reason,
            }
            for la in suggestion.local_adjustments
        ],
    }, ensure_ascii=False, indent=2)

    return display, json_data, img


def process_and_export(files, context, backend, output_format):
    """批量处理并导出"""
    if not files:
        return [], "请先上传图片"

    images = load_images([f.name for f in files])
    if not images:
        return [], "图片加载失败"

    # 分析
    suggestions = pipeline.analyze_batch(images, context, backend or None)

    # 处理
    batch = pipeline.process_batch(images, suggestions)

    # 导出到临时目录
    output_dir = Path(tempfile.mkdtemp(prefix="ai_beautify_"))
    paths = pipeline.export_batch(batch, output_dir, output_format)

    # 生成结果摘要
    summary = (
        f"✅ 处理完成\n"
        f"- 总计: {batch.total} 张\n"
        f"- 成功: {batch.success_count} 张\n"
        f"- 失败: {batch.fail_count} 张\n"
        f"- 输出目录: {output_dir}"
    )

    return [str(p) for p in paths], summary


# ── Gradio 界面 ──────────────────────────────────────

def build_app() -> gr.Blocks:
    with gr.Blocks(title="AI Beautify") as app:
        gr.Markdown("# 🎨 AI Beautify\n全自动 AI 修图与调色工作流")

        with gr.Row():
            # 左侧：输入
            with gr.Column(scale=1):
                files = gr.File(
                    label="上传照片",
                    file_count="multiple",
                    file_types=["image"],
                )
                context = gr.Textbox(
                    label="场景描述",
                    placeholder="例如：这是我苏州旅行拍的照片，主要是园林和古建筑",
                    lines=2,
                )
                with gr.Row():
                    backend = gr.Radio(
                        choices=["glm", "agnes", "ollama"],
                        label="VLM 后端",
                        value="",
                        info="留空自动降级",
                    )
                    output_format = gr.Radio(
                        choices=["jpeg", "heic"],
                        label="输出格式",
                        value="jpeg",
                    )
                with gr.Row():
                    analyze_btn = gr.Button("📷 分析", variant="secondary")
                    process_btn = gr.Button("🚀 处理并导出", variant="primary")

            # 右侧：输出
            with gr.Column(scale=2):
                with gr.Tabs():
                    with gr.Tab("📋 建议"):
                        suggestion_display = gr.Markdown(label="修图建议")
                    with gr.Tab("📄 JSON"):
                        json_display = gr.Code(language="json", label="原始 JSON")
                    with gr.Tab("🖼️ 预览"):
                        preview_img = gr.Image(label="原图预览")
                    with gr.Tab("📁 导出"):
                        output_gallery = gr.Gallery(label="处理结果")
                        export_summary = gr.Markdown()

        # 事件绑定
        analyze_btn.click(
            fn=analyze_images,
            inputs=[files, context, backend],
            outputs=[suggestion_display, json_display, preview_img],
        )
        process_btn.click(
            fn=process_and_export,
            inputs=[files, context, backend, output_format],
            outputs=[output_gallery, export_summary],
        )

    return app


def main():
    import socket

    def find_free_port(start=7860, end=7870):
        for port in range(start, end):
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                if s.connect_ex(("127.0.0.1", port)) != 0:
                    return port
        return start

    port = find_free_port()
    app = build_app()
    app.launch(server_name="0.0.0.0", server_port=port, share=False)


if __name__ == "__main__":
    main()
