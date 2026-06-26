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


def format_suggestion_json(s: EditSuggestion) -> str:
    return json.dumps({
        "analysis": s.analysis,
        "style": s.style,
        "global_params": {
            "exposure_ev": s.global_params.exposure_ev,
            "white_balance_k": s.global_params.white_balance_k,
            "contrast": s.global_params.contrast,
            "highlights": s.global_params.highlights,
            "shadows": s.global_params.shadows,
            "saturation": s.global_params.saturation,
        },
        "local_adjustments": [
            {
                "description": la.description,
                "position": {"x": la.x, "y": la.y},
                "type": la.adjustment_type,
                "exposure_ev": la.exposure_ev,
                "reason": la.reason,
            }
            for la in s.local_adjustments
        ],
    }, ensure_ascii=False, indent=2)


# ── 分析（带加载状态）─────────────────────────────

def analyze_images(files, context, backend):
    """批量分析图片"""
    if not files:
        return "请上传图片", "", None, []

    images = load_images([f.name for f in files])
    if not images:
        return "图片加载失败", "", None, []

    # 分析所有图片
    suggestions = pipeline.analyze_batch(images, context, backend or None)

    # 第一张用于预览
    suggestion = suggestions[0]
    display = format_suggestion(suggestion)
    json_data = format_suggestion_json(suggestion)
    first_img = images[0][1]

    # 存储所有原图和建议
    all_data = [
        {"name": Path(images[i][0]).name, "image": images[i][1], "suggestion": s}
        for i, s in enumerate(suggestions)
    ]

    return display, json_data, first_img, all_data


# ── 处理并导出（带进度）─────────────────────────────

def process_and_export(files, context, backend, output_format):
    """批量处理并导出"""
    if not files:
        return [], "请先上传图片", {}, None

    logger.info("开始处理 %d 张图片", len(files))
    images = load_images([f.name for f in files])
    if not images:
        return [], "图片加载失败", {}, None
    logger.info("图片加载完成: %d 张", len(images))

    # 分析
    logger.info("开始 VLM 分析...")
    suggestions = pipeline.analyze_batch(
        images, context, backend or None,
        on_progress=lambda cur, total, name: logger.info("分析进度: %d/%d - %s", cur, total, name),
    )
    logger.info("VLM 分析完成")

    # 处理
    logger.info("开始图像处理...")
    batch = pipeline.process_batch(images, suggestions)
    logger.info("图像处理完成: 成功 %d, 失败 %d", batch.success_count, batch.fail_count)

    # 导出到临时目录
    output_dir = Path(tempfile.mkdtemp(prefix="ai_beautify_"))
    logger.info("开始导出到 %s", output_dir)
    paths = pipeline.export_batch(batch, output_dir, output_format)
    logger.info("导出完成: %d 张", len(paths))

    # 构建前后对比数据
    comparison = {}
    for i, result in enumerate(batch.results):
        if result.success:
            comparison[i] = {
                "original": result.original,
                "processed": result.processed,
                "name": Path(result.source_path).name,
            }

    summary = (
        f"✅ 处理完成\n"
        f"- 总计: {batch.total} 张\n"
        f"- 成功: {batch.success_count} 张\n"
        f"- 失败: {batch.fail_count} 张\n"
        f"- 输出目录: {output_dir}"
    )

    # 第一张处理结果用于预览
    first_processed = batch.results[0].processed if batch.results and batch.results[0].success else None

    return [str(p) for p in paths], summary, comparison, first_processed


# ── 对比交互 ──────────────────────────────────────

def select_image(comparison, index):
    """选择一张图片进行对比"""
    if not comparison or index not in comparison:
        return None, None, "无数据"
    item = comparison[index]
    return item["original"], item["original"], f"当前: {item['name']}"


# ── 自定义 CSS + JS ──────────────────────────────

CUSTOM_CSS = """
#compare-btn {
    position: absolute;
    top: 10px;
    right: 10px;
    z-index: 100;
    background: rgba(0,0,0,0.7);
    color: white;
    border: none;
    border-radius: 8px;
    padding: 8px 16px;
    cursor: pointer;
    font-size: 14px;
    backdrop-filter: blur(4px);
    transition: background 0.2s;
}
#compare-btn:hover {
    background: rgba(0,0,0,0.9);
}
#compare-btn:active {
    background: rgba(59,130,246,0.9);
}
#compare-hint {
    text-align: center;
    color: #888;
    font-size: 12px;
    margin-top: 4px;
}
#status-indicator {
    padding: 8px 12px;
    border-radius: 6px;
    margin: 4px 0;
    font-size: 13px;
}
.status-idle { background: #f3f4f6; color: #6b7280; }
.status-loading { background: #dbeafe; color: #2563eb; animation: pulse 1.5s infinite; }
.status-done { background: #d1fae5; color: #059669; }
.status-error { background: #fee2e2; color: #dc2626; }
@keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.6; }
}
"""

CUSTOM_JS = """
function() {
    // 按住按钮显示原图，松开恢复新图
    let compareInterval = null;

    window.setupCompare = function() {
        const btn = document.getElementById('compare-btn');
        const img = document.getElementById('compare-display');
        if (!btn || !img) return;

        // 获取 Gradio image 组件中的 img 元素
        const gradioImg = img.querySelector('img') || img;

        btn.addEventListener('mousedown', function(e) {
            e.preventDefault();
            if (window._originalSrc) {
                gradioImg.src = window._originalSrc;
                btn.textContent = '👁 查看中...';
            }
        });

        btn.addEventListener('mouseup', function(e) {
            e.preventDefault();
            if (window._processedSrc) {
                gradioImg.src = window._processedSrc;
                btn.textContent = '👁 按住看原图';
            }
        });

        btn.addEventListener('mouseleave', function(e) {
            if (window._processedSrc) {
                gradioImg.src = window._processedSrc;
                btn.textContent = '👁 按住看原图';
            }
        });

        // 触摸支持
        btn.addEventListener('touchstart', function(e) {
            e.preventDefault();
            if (window._originalSrc) {
                gradioImg.src = window._originalSrc;
                btn.textContent = '👁 查看中...';
            }
        });

        btn.addEventListener('touchend', function(e) {
            e.preventDefault();
            if (window._processedSrc) {
                gradioImg.src = window._processedSrc;
                btn.textContent = '👁 按住看原图';
            }
        });
    };

    // 定期检查并设置（Gradio 动态渲染）
    setInterval(function() {
        const btn = document.getElementById('compare-btn');
        if (btn && !btn._bound) {
            window.setupCompare();
            btn._bound = true;
        }
    }, 500);
}
"""


def build_app() -> gr.Blocks:
    with gr.Blocks(title="AI Beautify", css=CUSTOM_CSS, js=CUSTOM_JS) as app:
        # 状态存储
        all_data_state = gr.State([])
        comparison_state = gr.State({})
        current_index = gr.State(0)

        gr.Markdown("# 🎨 AI Beautify\n全自动 AI 修图与调色工作流")

        with gr.Row():
            # ── 左侧：输入 ──
            with gr.Column(scale=1):
                files = gr.File(
                    label="上传照片",
                    file_count="multiple",
                    file_types=["image"],
                )
                context = gr.Textbox(
                    label="场景描述",
                    placeholder="例如：苏州园林旅行照，阴天",
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

                # 状态指示器
                status = gr.HTML(
                    value='<div id="status-indicator" class="status-idle">📷 等待上传照片</div>',
                )

                with gr.Row():
                    analyze_btn = gr.Button("📷 分析", variant="secondary", size="lg")
                    process_btn = gr.Button("🚀 处理并导出", variant="primary", size="lg")

            # ── 右侧：输出 ──
            with gr.Column(scale=2):
                with gr.Tabs():
                    # 分析建议 Tab
                    with gr.Tab("📋 建议"):
                        suggestion_display = gr.Markdown(
                            value="*上传照片后点击「分析」查看建议*",
                        )
                        json_display = gr.Code(
                            language="json",
                            label="原始 JSON",
                            visible=False,
                        )

                    # 对比预览 Tab
                    with gr.Tab("🔍 对比预览"):
                        with gr.Row():
                            with gr.Column():
                                gr.Markdown("### 原图")
                                original_preview = gr.Image(
                                    label="原图",
                                    interactive=False,
                                    height=400,
                                )
                            with gr.Column():
                                gr.Markdown("### 处理后")
                                processed_preview = gr.Image(
                                    label="处理后",
                                    interactive=False,
                                    height=400,
                                )
                        compare_status = gr.Markdown("*处理后自动显示对比*")

                    # 导出 Tab
                    with gr.Tab("📁 导出"):
                        output_gallery = gr.Gallery(
                            label="处理结果",
                            columns=3,
                            height=400,
                        )
                        export_summary = gr.Markdown()

        # ── 事件绑定 ──

        def on_analyze(files, context, backend):
            if not files:
                return (
                    "请上传照片", "", None,
                    [], {},
                    '<div id="status-indicator" class="status-error">❌ 请先上传照片</div>',
                )
            status_html = '<div id="status-indicator" class="status-loading">⏳ 正在分析，请稍候...</div>'
            yield "⏳ 正在分析...", "", None, [], {}, status_html

            result = analyze_images(files, context, backend)
            display, json_data, first_img, all_data = result

            done_html = f'<div id="status-indicator" class="status-done">✅ 分析完成，共 {len(all_data)} 张</div>'
            yield display, json_data, first_img, all_data, {}, done_html

        analyze_btn.click(
            fn=on_analyze,
            inputs=[files, context, backend],
            outputs=[suggestion_display, json_display, original_preview, all_data_state, comparison_state, status],
        )

        def on_process(files, context, backend, output_format):
            if not files:
                return [], "请先上传照片", {}, None, None, \
                    '<div id="status-indicator" class="status-error">❌ 请先上传照片</div>'

            loading_html = '<div id="status-indicator" class="status-loading">⏳ 正在处理，请稍候（VLM 分析 + 图像处理）...</div>'
            yield [], "⏳ 正在处理...", {}, None, None, loading_html

            result = process_and_export(files, context, backend, output_format)
            paths, summary, comparison, first_processed = result

            # 获取第一张的原图
            first_original = comparison[0]["original"] if 0 in comparison else None

            done_html = f'<div id="status-indicator" class="status-done">✅ {summary.split(chr(10))[0]}</div>'
            yield paths, summary, comparison, first_processed, first_original, done_html

        process_btn.click(
            fn=on_process,
            inputs=[files, context, backend, output_format],
            outputs=[output_gallery, export_summary, comparison_state, processed_preview, original_preview, status],
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
