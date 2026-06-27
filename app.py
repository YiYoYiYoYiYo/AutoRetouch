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
    """构建滑动对比 HTML（用 iframe 确保 JS 执行）"""
    return f"""
    <iframe srcdoc='<!DOCTYPE html><html><head><style>
    *{{margin:0;padding:0;box-sizing:border-box}}
    body{{background:#1a1a2e;display:flex;justify-content:center;padding:10px}}
    #sc{{position:relative;width:100%;max-width:900px;overflow:hidden;border-radius:8px;cursor:col-resize;user-select:none}}
    #sc img{{display:block;width:100%}}
    #clip{{position:absolute;top:0;left:0;width:50%;height:100%;overflow:hidden}}
    #clip img{{width:900px;max-width:none;display:block}}
    #handle{{position:absolute;top:0;left:50%;width:4px;height:100%;background:#fff;box-shadow:0 0 8px rgba(0,0,0,.5);transform:translateX(-50%);pointer-events:none}}
    #label{{position:absolute;top:50%;left:50%;transform:translate(-50%,-50%);background:rgba(0,0,0,.7);color:#fff;padding:6px 12px;border-radius:20px;font-size:13px;white-space:nowrap;pointer-events:none}}
    </style></head><body>
    <div id="sc">
        <img src="{processed_b64}" id="pi" />
        <div id="clip"><img src="{original_b64}" id="oi" /></div>
        <div id="handle"><div id="label">◀ 原图 | 处理后 ▶</div></div>
    </div>
    <script>
    var sc=document.getElementById("sc"),clip=document.getElementById("clip"),
    handle=document.getElementById("handle"),oi=document.getElementById("oi"),
    pi=document.getElementById("pi"),drag=false;
    function pos(x){{var r=sc.getBoundingClientRect(),p=Math.max(0,Math.min(100,((x-r.left)/r.width)*100));
    clip.style.width=p+"%";handle.style.left=p+"%";oi.style.width=r.width+"px"}}
    pi.onload=function(){{oi.style.width=sc.offsetWidth+"px"}};
    if(pi.complete)oi.style.width=sc.offsetWidth+"px";
    sc.onmousedown=function(e){{drag=true;pos(e.clientX);e.preventDefault()}};
    document.onmousemove=function(e){{if(drag)pos(e.clientX)}};
    document.onmouseup=function(){{drag=false}};
    sc.ontouchstart=function(e){{drag=true;pos(e.touches[0].clientX);e.preventDefault()}};
    document.ontouchmove=function(e){{if(drag){{pos(e.touches[0].clientX);e.preventDefault()}}}};
    document.ontouchend=function(){{drag=false}};
    window.onresize=function(){{oi.style.width=sc.offsetWidth+"px"}};
    </script></body></html>'
    style="width:100%;height:650px;border:none;border-radius:8px;"></iframe>
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

def _auto_effects(suggestion: EditSuggestion):
    """根据 VLM 分析文本自动补全 vignette/blur（VLM 经常漏填这两个）

    保守策略：不叠加、不冲突，只在真正需要时补一个效果
    """
    text = (suggestion.analysis + " " + suggestion.style).lower()
    gp = suggestion.global_params

    # 统计 VLM 已建议的局部调整类型
    local_types = {la.adjustment_type for la in suggestion.local_adjustments}
    has_darken = "darken" in local_types or "highlights" in local_types

    # blur：分析提到背景杂乱/虚化/聚焦，且 VLM 没建议 darken（避免叠加过重）
    if gp.blur == 0 and not has_darken and any(kw in text for kw in (
        "杂乱", "虚化", "聚焦", "景深", "突出主体", "干扰",
        "适度处理", "淡化", "散焦",
    )):
        gp.blur = 0.3

    # vignette：人像/建筑/静物场景，且没加 blur、没 darken（避免叠加）
    elif gp.vignette == 0 and gp.blur == 0 and not has_darken and any(kw in text for kw in (
        "人像", "建筑", "静物", "肖像", "特写",
        "桌面", "花卉", "美食", "产品",
    )):
        gp.vignette = 0.3

    suggestion.global_params = gp


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
            suggestion = pipeline._bridge.analyze(img, context, None if backend in ("自动", "") else backend)
            _auto_effects(suggestion)
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
            '<div class="status-error">❌ 请先上传照片</div>', make_log_html(logs), None
        return

    log(f"📷 加载 {len(files)} 张图片...")
    yield [], "⏳ 加载中...", empty_slider, \
        '<div class="status-loading">⏳ 加载图片...</div>', make_log_html(logs), None

    images = load_images([f.name for f in files])
    if not images:
        log("❌ 图片加载失败")
        yield [], "图片加载失败", empty_slider, \
            '<div class="status-error">❌ 图片加载失败</div>', make_log_html(logs), None
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
            f'<div class="status-loading">⏳ 分析 [{i+1}/{len(images)}]: {Path(name).name}</div>', make_log_html(logs), None

        try:
            suggestion = pipeline._bridge.analyze(img, context, None if backend in ("自动", "") else backend)
            # 自动检测：VLM 分析文本中提到背景/虚化/聚焦时自动加 blur/vignette
            _auto_effects(suggestion)
            gp = suggestion.global_params
            log(f"  → 后端: {suggestion.backend} | EV={gp.exposure_ev:+.2f} WB={gp.white_balance_k}K "
                f"contrast={gp.contrast:+d} highlights={gp.highlights:+d} shadows={gp.shadows:+d} "
                f"sat={gp.saturation:+d} vignette={gp.vignette:.2f} blur={gp.blur:.2f} "
                f"局部={len(suggestion.local_adjustments)}个")
        except Exception as e:
            log(f"  ❌ 分析失败: {e}")
            suggestion = EditSuggestion(analysis=f"分析失败: {e}", backend="error")

        # 图像处理
        log(f"🎨 处理 [{i+1}/{len(images)}]: {Path(name).name}")
        yield [], "⏳ 处理中...", empty_slider if first_slider is None else first_slider, \
            f'<div class="status-loading">⏳ 处理 [{i+1}/{len(images)}]: {Path(name).name}</div>', make_log_html(logs), None

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

    # 打包 zip
    zip_path = None
    if export_paths:
        import zipfile
        zip_path = str(output_dir / "ai_beautify_export.zip")
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for p in export_paths:
                zf.write(p, Path(p).name)
        log(f"📦 已打包: {Path(zip_path).name}")

    summary = f"✅ 处理完成\n- 总计: {len(images)} 张\n- 成功: {success_count} 张\n- 输出目录: {output_dir}"

    yield (
        export_paths,
        summary,
        first_slider or empty_slider,
        f'<div class="status-done">✅ 完成 {success_count}/{len(images)} 张，耗时 {total_time:.1f}s</div>',
        make_log_html(logs),
        zip_path,
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
                    backend = gr.Radio(choices=["自动", "glm", "agnes", "ollama"], label="VLM 后端", value="自动", info="自动=按优先级降级")
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
                        with gr.Row():
                            export_summary = gr.Markdown()
                            zip_file = gr.File(label="📦 批量下载", interactive=False)

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
            outputs=[output_gallery, export_summary, slider_display, status, log_display, zip_file],
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
