"""AI Beautify - Gradio 用户界面"""

import base64
import io
import json
import logging
import tempfile
import time
from pathlib import Path

import gradio as gr
from PIL import Image as PILImage

from config import cfg
from pipeline import BatchPipeline, load_images
from vlm.base import EditSuggestion

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

pipeline = BatchPipeline()


# ── 工具函数 ──────────────────────────────────────

def img_to_base64(img: PILImage.Image, max_w=1200) -> str:
    """PIL Image → base64 data URL，限制最大宽度"""
    w, h = img.size
    if w > max_w:
        ratio = max_w / w
        img = img.resize((max_w, int(h * ratio)), PILImage.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    b64 = base64.b64encode(buf.getvalue()).decode()
    return f"data:image/jpeg;base64,{b64}"


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


def build_slider_html(original_b64: str, processed_b64: str) -> str:
    """构建滑动对比 HTML"""
    return f"""
    <div id="slider-container" style="position:relative;width:100%;max-width:900px;overflow:hidden;border-radius:8px;cursor:col-resize;user-select:none;">
        <img src="{processed_b64}" style="width:100%;display:block;" />
        <div id="slider-clip" style="position:absolute;top:0;left:0;width:50%;height:100%;overflow:hidden;">
            <img src="{original_b64}" style="width:900px;max-width:none;display:block;" />
        </div>
        <div id="slider-handle" style="position:absolute;top:0;left:50%;width:3px;height:100%;background:#fff;box-shadow:0 0 8px rgba(0,0,0,0.5);transform:translateX(-50%);pointer-events:none;">
            <div style="position:absolute;top:50%;left:50%;transform:translate(-50%,-50%);background:rgba(0,0,0,0.7);color:#fff;padding:6px 10px;border-radius:20px;font-size:12px;white-space:nowrap;pointer-events:none;">
                ◀ 原图 | 处理后 ▶
            </div>
        </div>
    </div>
    <script>
    (function() {{
        const container = document.getElementById('slider-container');
        const clip = document.getElementById('slider-clip');
        const handle = document.getElementById('slider-handle');
        const clipImg = clip.querySelector('img');
        if (!container || !clip || !handle) return;

        function setPosition(clientX) {{
            const rect = container.getBoundingClientRect();
            let pct = ((clientX - rect.left) / rect.width) * 100;
            pct = Math.max(0, Math.min(100, pct));
            clip.style.width = pct + '%';
            handle.style.left = pct + '%';
            // clip 内的图片宽度 = 容器实际宽度
            clipImg.style.width = rect.width + 'px';
        }}

        // 初始化：图片加载后设置 clip 图片宽度
        const mainImg = container.querySelector(':scope > img');
        if (mainImg.complete) {{
            clipImg.style.width = container.offsetWidth + 'px';
        }} else {{
            mainImg.onload = () => {{ clipImg.style.width = container.offsetWidth + 'px'; }};
        }}

        let dragging = false;
        container.addEventListener('mousedown', (e) => {{ dragging = true; setPosition(e.clientX); }});
        document.addEventListener('mousemove', (e) => {{ if (dragging) setPosition(e.clientX); }});
        document.addEventListener('mouseup', () => {{ dragging = false; }});
        container.addEventListener('touchstart', (e) => {{ dragging = true; setPosition(e.touches[0].clientX); }});
        container.addEventListener('touchmove', (e) => {{ if (dragging) {{ e.preventDefault(); setPosition(e.touches[0].clientX); }} }});
        container.addEventListener('touchend', () => {{ dragging = false; }});

        window.addEventListener('resize', () => {{ clipImg.style.width = container.offsetWidth + 'px'; }});
    }})();
    </script>
    """


def make_log_html(logs: list[str]) -> str:
    """构建日志 HTML"""
    if not logs:
        return "<div id='log-panel' style='background:#1a1a2e;color:#e0e0e0;font-family:Consolas,Monospace;font-size:12px;padding:10px 14px;border-radius:8px;max-height:160px;overflow-y:auto;line-height:1.6;'>等待操作...</div>"
    lines = logs[-30:]
    content = "<br>".join(lines)
    return f"<div id='log-panel' style='background:#1a1a2e;color:#e0e0e0;font-family:Consolas,Monospace;font-size:12px;padding:10px 14px;border-radius:8px;max-height:160px;overflow-y:auto;line-height:1.6;'>{content}<script>var p=document.getElementById('log-panel');if(p)p.scrollTop=p.scrollHeight;</script></div>"


# ── CSS ──────────────────────────────────────────

CUSTOM_CSS = """
.status-idle { background: #f3f4f6; color: #6b7280; padding: 8px 12px; border-radius: 6px; }
.status-loading { background: #dbeafe; color: #2563eb; padding: 8px 12px; border-radius: 6px; animation: pulse 1.5s infinite; }
.status-done { background: #d1fae5; color: #059669; padding: 8px 12px; border-radius: 6px; }
.status-error { background: #fee2e2; color: #dc2626; padding: 8px 12px; border-radius: 6px; }
@keyframes pulse { 0%,100%{opacity:1} 50%{opacity:0.6} }
/* 导出图片自适应 */
.gallery-item img, .gallery-item video { object-fit: contain !important; max-height: 300px !important; }
"""


# ── 核心逻辑（逐张处理 + 实时 yield）──────────────

def analyze_images(files, context, backend):
    """逐张分析，实时 yield 进度"""
    logs = []
    t0 = time.time()

    def log(msg):
        elapsed = time.time() - t0
        logs.append(f"[{elapsed:.1f}s] {msg}")

    empty_slider = "<div style='text-align:center;color:#888;padding:40px;'>分析完成后显示预览</div>"

    if not files:
        log("❌ 请先上传照片")
        yield "请上传照片", "", make_log_html(logs), empty_slider, '<div class="status-error">❌ 请先上传照片</div>'
        return

    log(f"📷 加载 {len(files)} 张图片...")
    yield "⏳ 加载中...", "", make_log_html(logs), empty_slider, '<div class="status-loading">⏳ 加载图片...</div>'

    images = load_images([f.name for f in files])
    if not images:
        log("❌ 图片加载失败")
        yield "图片加载失败", "", make_log_html(logs), empty_slider, '<div class="status-error">❌ 图片加载失败</div>'
        return

    log(f"✅ 加载完成 ({len(images)} 张)")
    suggestions = []

    for i, (name, img) in enumerate(images):
        log(f"🧠 分析 [{i+1}/{len(images)}]: {Path(name).name}")
        yield "⏳ 分析中...", "", make_log_html(logs), empty_slider, f'<div class="status-loading">⏳ 分析 [{i+1}/{len(images)}]...</div>'

        try:
            suggestion = pipeline._bridge.analyze(img, context, backend or None)
            suggestions.append(suggestion)
            log(f"  → 风格: {suggestion.style}")
        except Exception as e:
            log(f"  ❌ 分析失败: {e}")
            suggestions.append(EditSuggestion(analysis=f"分析失败: {e}", backend="error"))

    s = suggestions[0]
    total_time = time.time() - t0
    log(f"🎉 全部分析完成! 耗时 {total_time:.1f}s")

    # 用第一张原图生成预览
    preview_html = f"<div style='text-align:center'><img src='{img_to_base64(images[0][1])}' style='max-width:100%;border-radius:8px;' /></div>"

    yield (
        format_suggestion(s),
        format_suggestion_json(s),
        make_log_html(logs),
        preview_html,
        f'<div class="status-done">✅ 分析完成 {len(images)} 张，耗时 {total_time:.1f}s</div>',
    )


def process_and_export(files, context, backend, output_format):
    """逐张处理，实时 yield 进度"""
    logs = []
    t0 = time.time()

    def log(msg):
        elapsed = time.time() - t0
        logs.append(f"[{elapsed:.1f}s] {msg}")

    empty_slider = "<div style='text-align:center;color:#888;padding:40px;'>处理完成后自动显示对比</div>"

    if not files:
        log("❌ 请先上传照片")
        yield [], "请先上传照片", empty_slider, \
            '<div class="status-error">❌ 请先上传照片</div>', make_log_html(logs)
        return

    log(f"📷 加载 {len(files)} 张图片...")
    yield [], "⏳ 加载中...", empty_slider, \
        '<div class="status-loading">⏳ 加载图片...</div>', make_log_html(logs)

    images = load_images([f.name for f in files])
    if not images:
        log("❌ 图片加载失败")
        yield [], "图片加载失败", empty_slider, \
            '<div class="status-error">❌ 图片加载失败</div>', make_log_html(logs)
        return
    log(f"✅ 加载完成 ({len(images)} 张)")

    # 逐张：VLM 分析 → 图像处理 → 导出
    results = []
    export_paths = []
    output_dir = Path(tempfile.mkdtemp(prefix="ai_beautify_"))
    first_slider = None

    for i, (name, img) in enumerate(images):
        # VLM 分析
        log(f"🧠 分析 [{i+1}/{len(images)}]: {Path(name).name}")
        yield [], "⏳ 处理中...", empty_slider if first_slider is None else first_slider, \
            f'<div class="status-loading">⏳ 分析 [{i+1}/{len(images)}]: {Path(name).name}</div>', make_log_html(logs)

        try:
            suggestion = pipeline._bridge.analyze(img, context, backend or None)
            log(f"  → 风格: {suggestion.style}")
        except Exception as e:
            log(f"  ❌ 分析失败: {e}")
            suggestion = EditSuggestion(analysis=f"分析失败: {e}", backend="error")

        # 图像处理
        log(f"🎨 处理 [{i+1}/{len(images)}]: {Path(name).name}")
        yield [], "⏳ 处理中...", empty_slider if first_slider is None else first_slider, \
            f'<div class="status-loading">⏳ 处理 [{i+1}/{len(images)}]: {Path(name).name}</div>', make_log_html(logs)

        try:
            processed = pipeline._processor.process(img, suggestion)
            log(f"  ✅ 处理完成")
        except Exception as e:
            log(f"  ❌ 处理失败: {e}")
            processed = img

        # 导出
        fmt = output_format or cfg.processing.output_format
        stem = Path(name).stem
        out_path = output_dir / f"{stem}_edited.{fmt}"
        try:
            if fmt == "heic":
                try:
                    import pillow_heif
                    processed.save(str(out_path), format="HEIF")
                except ImportError:
                    out_path = output_dir / f"{stem}_edited.jpeg"
                    processed.save(str(out_path), format="JPEG", quality=cfg.processing.jpeg_quality)
            else:
                processed.save(str(out_path), format="JPEG", quality=cfg.processing.jpeg_quality)
            export_paths.append(str(out_path))
            log(f"  💾 导出: {out_path.name}")
        except Exception as e:
            log(f"  ❌ 导出失败: {e}")

        # 第一张生成滑动对比
        if i == 0:
            orig_b64 = img_to_base64(img)
            proc_b64 = img_to_base64(processed)
            first_slider = build_slider_html(orig_b64, proc_b64)

    total_time = time.time() - t0
    success_count = len(export_paths)
    log(f"🎉 全部完成! 成功 {success_count}/{len(images)}，耗时 {total_time:.1f}s")

    summary = f"✅ 处理完成\n- 总计: {len(images)} 张\n- 成功: {success_count} 张\n- 输出目录: {output_dir}"

    yield (
        export_paths,
        summary,
        first_slider or empty_slider,
        f'<div class="status-done">✅ 完成 {success_count}/{len(images)} 张，耗时 {total_time:.1f}s</div>',
        make_log_html(logs),
    )


# ── 构建界面 ──────────────────────────────────────

def build_app() -> gr.Blocks:
    with gr.Blocks(title="AI Beautify", css=CUSTOM_CSS) as app:
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

                    with gr.Tab("🔍 对比预览"):
                        gr.Markdown("*处理完成后左右拖动滑块对比原图和处理后*")
                        slider_display = gr.HTML(
                            value="<div style='text-align:center;color:#888;padding:40px;'>处理完成后自动显示对比</div>",
                        )

                    with gr.Tab("📁 导出"):
                        output_gallery = gr.Gallery(
                            label="处理结果", columns=3,
                            height=500, object_fit="contain",
                        )
                        export_summary = gr.Markdown()

        # ── 日志面板（底部）──
        log_display = gr.HTML(value=make_log_html([]))

        # ── 事件绑定 ──

        analyze_btn.click(
            fn=analyze_images,
            inputs=[files, context, backend],
            outputs=[suggestion_display, json_display, log_display, slider_display, status],
        )

        process_btn.click(
            fn=process_and_export,
            inputs=[files, context, backend, output_format],
            outputs=[output_gallery, export_summary, slider_display, status, log_display],
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
