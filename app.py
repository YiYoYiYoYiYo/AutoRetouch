"""AI Beautify - Gradio 用户界面"""

import json
import logging
import tempfile
import time
from pathlib import Path

import gradio as gr

from config import cfg
from pipeline import BatchPipeline, load_images
from vlm.base import EditSuggestion

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

pipeline = BatchPipeline()


# ── 日志收集器 ──────────────────────────────────────

class StepLogger:
    def __init__(self):
        self.logs: list[str] = []
        self._step_start: float = 0

    def step(self, msg: str):
        now = time.time()
        elapsed = f" ({now - self._step_start:.1f}s)" if self._step_start else ""
        ts = time.strftime("%H:%M:%S")
        self.logs.append(f"[{ts}] {msg}{elapsed}")
        self._step_start = now

    def to_html(self) -> str:
        if not self.logs:
            return "<div class='log-empty'>等待操作...</div>"
        lines = self.logs[-20:]
        return "<br>".join(lines)


# ── 格式化 ──────────────────────────────────────────

def format_suggestion(s: EditSuggestion) -> str:
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
            lines.append(f"{i}. **{la.description}** ({la.x:.2f}, {la.y:.2f}) - {la.adjustment_type}: {la.exposure_ev:+.2f}EV")
            if la.reason:
                lines.append(f"   > {la.reason}")
    else:
        lines.append("## 🎯 局部调整\n无需局部调整")
    lines.append(f"\n---\n*后端: {s.backend}*")
    return "\n".join(lines)


def format_suggestion_json(s: EditSuggestion) -> str:
    return json.dumps({
        "analysis": s.analysis, "style": s.style,
        "global_params": {
            "exposure_ev": s.global_params.exposure_ev,
            "white_balance_k": s.global_params.white_balance_k,
            "contrast": s.global_params.contrast,
            "highlights": s.global_params.highlights,
            "shadows": s.global_params.shadows,
            "saturation": s.global_params.saturation,
        },
        "local_adjustments": [
            {"description": la.description, "position": {"x": la.x, "y": la.y},
             "type": la.adjustment_type, "exposure_ev": la.exposure_ev, "reason": la.reason}
            for la in s.local_adjustments
        ],
    }, ensure_ascii=False, indent=2)


# ── CSS ──────────────────────────────────────────

CUSTOM_CSS = """
#log-panel {
    background: #1a1a2e;
    color: #e0e0e0;
    font-family: 'Consolas', 'Monaco', monospace;
    font-size: 12px;
    padding: 10px 14px;
    border-radius: 8px;
    max-height: 160px;
    overflow-y: auto;
    margin-top: 8px;
    line-height: 1.6;
}
.log-empty { color: #666; font-style: italic; }
.status-idle { background: #f3f4f6; color: #6b7280; padding: 8px 12px; border-radius: 6px; }
.status-loading { background: #dbeafe; color: #2563eb; padding: 8px 12px; border-radius: 6px; animation: pulse 1.5s infinite; }
.status-done { background: #d1fae5; color: #059669; padding: 8px 12px; border-radius: 6px; }
.status-error { background: #fee2e2; color: #dc2626; padding: 8px 12px; border-radius: 6px; }
@keyframes pulse { 0%,100%{opacity:1} 50%{opacity:0.6} }
"""


# ── 核心逻辑 ──────────────────────────────────────

def analyze_images(files, context, backend):
    log = StepLogger()
    if not files:
        log.step("❌ 请先上传照片")
        return "请上传照片", "", None, [], log.to_html()

    log.step(f"📷 加载 {len(files)} 张图片...")
    images = load_images([f.name for f in files])
    if not images:
        log.step("❌ 图片加载失败")
        return "图片加载失败", "", None, [], log.to_html()

    log.step(f"✅ 加载完成，开始 VLM 分析...")
    suggestions = pipeline.analyze_batch(
        images, context, backend or None,
        on_progress=lambda cur, total, name: log.step(f"🧠 分析 [{cur}/{total}]: {Path(name).name}"),
    )
    log.step(f"✅ 分析完成，共 {len(suggestions)} 张")

    suggestion = suggestions[0]
    return (
        format_suggestion(suggestion),
        format_suggestion_json(suggestion),
        images[0][1],
        [{"name": Path(images[i][0]).name, "image": images[i][1], "suggestion": s} for i, s in enumerate(suggestions)],
        log.to_html(),
    )


def process_and_export(files, context, backend, output_format):
    log = StepLogger()
    if not files:
        log.step("❌ 请先上传照片")
        return [], "请先上传照片", {}, None, None, True, "👁 按住看原图", log.to_html()

    log.step(f"📷 加载 {len(files)} 张图片...")
    images = load_images([f.name for f in files])
    if not images:
        log.step("❌ 图片加载失败")
        return [], "图片加载失败", {}, None, None, True, "👁 按住看原图", log.to_html()
    log.step(f"✅ 加载完成")

    log.step("🧠 开始 VLM 分析...")
    suggestions = pipeline.analyze_batch(
        images, context, backend or None,
        on_progress=lambda cur, total, name: log.step(f"🧠 分析 [{cur}/{total}]: {Path(name).name}"),
    )
    log.step(f"✅ VLM 分析完成")

    log.step("🎨 开始图像处理...")
    batch = pipeline.process_batch(
        images, suggestions,
        on_progress=lambda cur, total, name: log.step(f"🎨 处理 [{cur}/{total}]: {Path(name).name}"),
    )
    log.step(f"✅ 处理完成: 成功 {batch.success_count}, 失败 {batch.fail_count}")

    output_dir = Path(tempfile.mkdtemp(prefix="ai_beautify_"))
    log.step(f"💾 导出到 {output_dir}...")
    paths = pipeline.export_batch(
        batch, output_dir, output_format,
        on_progress=lambda cur, total, name: log.step(f"💾 导出 [{cur}/{total}]: {Path(name).name}"),
    )
    log.step(f"✅ 全部完成! 共 {len(paths)} 张")

    comparison = {}
    for i, result in enumerate(batch.results):
        if result.success:
            comparison[i] = {"original": result.original, "processed": result.processed, "name": Path(result.source_path).name}

    summary = f"✅ 处理完成\n- 总计: {batch.total} 张\n- 成功: {batch.success_count} 张\n- 失败: {batch.fail_count} 张\n- 输出目录: {output_dir}"
    first_processed = batch.results[0].processed if batch.results and batch.results[0].success else None
    first_original = comparison[0]["original"] if 0 in comparison else None

    return [str(p) for p in paths], summary, comparison, first_processed, first_original, False, "👁 按住看原图", log.to_html()


def toggle_compare(comparison, showing_original):
    """切换原图/处理后显示"""
    if not comparison or 0 not in comparison:
        return None, True, "👁 按住看原图"
    item = comparison[0]
    if showing_original:
        # 当前显示原图 → 切换到处理后
        return item["processed"], False, "👁 按住看原图"
    else:
        # 当前显示处理后 → 切换到原图
        return item["original"], True, "👁 松开恢复处理后"


# ── 构建界面 ──────────────────────────────────────

def build_app() -> gr.Blocks:
    with gr.Blocks(title="AI Beautify", css=CUSTOM_CSS) as app:
        all_data_state = gr.State([])
        comparison_state = gr.State({})
        showing_original = gr.State(False)

        gr.Markdown("# 🎨 AI Beautify\n全自动 AI 修图与调色工作流")

        with gr.Row():
            # ── 左侧 ──
            with gr.Column(scale=1):
                files = gr.File(label="上传照片", file_count="multiple", file_types=["image"])
                context = gr.Textbox(label="场景描述", placeholder="例如：苏州园林旅行照", lines=2)
                with gr.Row():
                    backend = gr.Radio(choices=["glm", "agnes", "ollama"], label="VLM 后端", value="", info="留空自动降级")
                    output_format = gr.Radio(choices=["jpeg", "heic"], label="输出格式", value="jpeg")
                status = gr.HTML(value='<div class="status-idle">📷 等待上传照片</div>')
                with gr.Row():
                    analyze_btn = gr.Button("📷 分析", variant="secondary", size="lg")
                    process_btn = gr.Button("🚀 处理并导出", variant="primary", size="lg")

            # ── 右侧 ──
            with gr.Column(scale=2):
                with gr.Tabs():
                    with gr.Tab("📋 建议"):
                        suggestion_display = gr.Markdown(value="*上传照片后点击「分析」*")
                        json_display = gr.Code(language="json", label="JSON", visible=False)

                    with gr.Tab("🔍 预览"):
                        preview_img = gr.Image(label="预览", interactive=False, height=450)
                        with gr.Row():
                            compare_btn = gr.Button("👁 按住看原图", variant="secondary", size="sm")
                            compare_hint = gr.Markdown("*处理完成后，点击按钮切换查看原图和处理后效果*")

                    with gr.Tab("📁 导出"):
                        output_gallery = gr.Gallery(label="处理结果", columns=3, height=400)
                        export_summary = gr.Markdown()

        # ── 日志面板 ──
        log_display = gr.HTML(value="<div id='log-panel'><div class='log-empty'>等待操作...</div></div>")

        # ── 事件绑定 ──

        def on_analyze(files, context, backend):
            if not files:
                yield "请上传照片", "", None, [], \
                    '<div class="status-error">❌ 请先上传照片</div>', \
                    "<div id='log-panel'><div class='log-empty'>等待操作...</div></div>"
                return
            yield "⏳ 正在分析...", "", None, [], \
                '<div class="status-loading">⏳ 正在分析，请稍候...</div>', \
                "<div id='log-panel'>⏳ 开始分析...</div>"
            display, json_data, first_img, all_data, log_html = analyze_images(files, context, backend)
            yield display, json_data, first_img, all_data, \
                f'<div class="status-done">✅ 分析完成，共 {len(all_data)} 张</div>', \
                f"<div id='log-panel'>{log_html}</div>"

        analyze_btn.click(
            fn=on_analyze, inputs=[files, context, backend],
            outputs=[suggestion_display, json_display, preview_img, all_data_state, status, log_display],
        )

        def on_process(files, context, backend, output_format):
            if not files:
                yield [], "请先上传照片", {}, None, None, True, "👁 按住看原图", \
                    '<div class="status-error">❌ 请先上传照片</div>', \
                    "<div id='log-panel'><div class='log-empty'>等待操作...</div></div>"
                return
            yield [], "⏳ 正在处理...", {}, None, None, True, "👁 按住看原图", \
                '<div class="status-loading">⏳ 正在处理（VLM + 图像处理 + 导出）...</div>', \
                "<div id='log-panel'>⏳ 开始处理...</div>"
            result = process_and_export(files, context, backend, output_format)
            paths, summary, comparison, first_processed, first_original, show_orig, btn_text, log_html = result
            yield paths, summary, comparison, first_processed, first_original, show_orig, btn_text, \
                '<div class="status-done">✅ 处理完成</div>', \
                f"<div id='log-panel'>{log_html}</div>"

        process_btn.click(
            fn=on_process, inputs=[files, context, backend, output_format],
            outputs=[output_gallery, export_summary, comparison_state, preview_img, preview_img,
                     showing_original, compare_btn, status, log_display],
        )

        def on_toggle(comparison, showing_original):
            img, showing, btn_text = toggle_compare(comparison, showing_original)
            return img, showing, btn_text

        compare_btn.click(
            fn=on_toggle, inputs=[comparison_state, showing_original],
            outputs=[preview_img, showing_original, compare_btn],
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
