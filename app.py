"""小学至高中数学错题分析 Demo（Streamlit 页面）。"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import os
import queue
import time

import streamlit as st
from dotenv import load_dotenv

from exam_analyzer import (
    DEFAULT_MODEL,
    ExamAnalyzerError,
    analyze_exam,
    prepare_images,
    recommended_timeout_seconds,
)


load_dotenv()

MODEL_OPTIONS = {
    "Qwen3-VL 8B Instruct｜最快、低成本": "Qwen/Qwen3-VL-8B-Instruct",
    "Qwen3-VL 30B A3B Instruct｜速度与效果平衡": "Qwen/Qwen3-VL-30B-A3B-Instruct",
    "Qwen3-VL 32B Instruct｜默认、OCR 与数学较稳": "Qwen/Qwen3-VL-32B-Instruct",
    "Qwen3-VL 32B Thinking｜复杂几何与推理、较慢": "Qwen/Qwen3-VL-32B-Thinking",
    "Qwen3.5 9B｜新一代轻量多模态": "Qwen/Qwen3.5-9B",
    "Qwen3.5 35B A3B｜新一代 MoE 多模态": "Qwen/Qwen3.5-35B-A3B",
    "Qwen3.5 122B A10B｜大型 MoE、复杂推理": "Qwen/Qwen3.5-122B-A10B",
    "Qwen3.5 397B A17B｜旗舰超大 MoE、能力最强": "Qwen/Qwen3.5-397B-A17B",
    "Qwen3.6 27B｜通用稠密多模态、较慢": "Qwen/Qwen3.6-27B",
    "Qwen3.6 35B A3B｜最新一代 MoE 多模态": "Qwen/Qwen3.6-35B-A3B",
    "GLM-4.5V｜106B 视觉推理、费用较高": "zai-org/GLM-4.5V",
    "Kimi K2.6 Pro｜超大多模态、强推理、高费用": "Pro/moonshotai/Kimi-K2.6",
    "Qwen3 Omni 30B Instruct｜图像/音频/视频": "Qwen/Qwen3-Omni-30B-A3B-Instruct",
    "DeepSeek OCR｜只推荐文字提取": "deepseek-ai/DeepSeek-OCR",
    "自定义模型 ID": "",
}

st.set_page_config(
    page_title="小学至高中数学智能批改",
    page_icon="🧮",
    layout="wide",
)

st.markdown(
    """
    <style>
    .block-container {max-width: 1100px; padding-top: 2rem;}
    [data-testid="stFileUploaderDropzone"] {min-height: 150px;}
    .small-note {color: #667085; font-size: 0.9rem;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🧮 小学至高中数学智能批改")
st.caption(f"图片识别与分析模型：`{os.getenv('SILICONFLOW_MODEL', DEFAULT_MODEL)}`")
st.write(
    "上传小学、初中或高中数学题及学生答案照片，系统会逐题判断对错，"
    "并给出错因、正确答案和订正步骤。"
)

api_key = os.getenv("SILICONFLOW_API_KEY", "").strip()
if not api_key:
    st.error(
        "尚未配置 API 密钥。请复制 `.env.example` 为 `.env`，"
        "填写 `SILICONFLOW_API_KEY` 后重启页面。"
    )

with st.sidebar:
    st.header("批改设置")
    grade = st.selectbox(
        "学段与年级",
        [
            "小学一年级",
            "小学二年级",
            "小学三年级",
            "小学四年级",
            "小学五年级",
            "小学六年级",
            "初中一年级（初一）",
            "初中二年级（初二）",
            "初中三年级（初三）",
            "高中一年级（高一）",
            "高中二年级（高二）",
            "高中三年级（高三）",
        ],
        index=3,
        help="年级会影响知识范围、解题方法、数学术语和讲解深度。",
    )
    analysis_mode = st.radio(
        "分析模式",
        ["快速批改", "详细分析"],
        horizontal=True,
        help="快速模式会缩短输出；详细模式给出更完整的逐题解释。",
    )
    default_model_id = os.getenv("SILICONFLOW_MODEL", DEFAULT_MODEL)
    model_labels = list(MODEL_OPTIONS)
    default_model_index = next(
        (
            index
            for index, label in enumerate(model_labels)
            if MODEL_OPTIONS[label] == default_model_id
        ),
        2,
    )
    model_label = st.selectbox(
        "多模态模型",
        model_labels,
        index=default_model_index,
        help=(
            "8B 通常最快；122B、397B、GLM-4.5V 和 Pro 模型能力更强，"
            "但通常更慢、费用更高，并需要账户有足够付费额度。"
        ),
    )
    selected_model = MODEL_OPTIONS[model_label]
    if not selected_model:
        selected_model = st.text_input(
            "自定义模型 ID",
            placeholder="例如：Qwen/Qwen3-VL-8B-Instruct",
        ).strip()
    selected_timeout = recommended_timeout_seconds(selected_model) if selected_model else 0
    if selected_timeout:
        st.caption(
            f"无响应数据超时：{selected_timeout} 秒。流式数据持续到达时会自动继续等待。"
        )
    max_edge = st.select_slider(
        "图片清晰度",
        options=[1600, 2000, 2400, 3000, 3584],
        value=2000,
        help="字迹较小时选 3000 或 3584；越清晰，视觉 Token 消耗通常越高。",
    )
    st.divider()
    st.markdown(
        """
        **拍照建议**

        - 题目和学生答案都要完整入镜
        - 保持纸面平整、光线均匀
        - 尽量正对试卷，避免阴影和反光
        - 一张照片不要包含过多小字
        """
    )
    st.warning("AI 批改可能出错，重要作业请由老师或家长复核。")

uploaded_files = st.file_uploader(
    "上传题目图片（最多 8 张）",
    type=["jpg", "jpeg", "png", "webp"],
    accept_multiple_files=True,
    help="每张图片不超过 15 MB。支持连续上传多页试卷。",
)

if len(uploaded_files) > 8:
    st.error("一次最多上传 8 张图片，请删除多余图片。")

if uploaded_files:
    st.subheader("图片预览")
    columns = st.columns(min(3, len(uploaded_files)))
    for index, uploaded in enumerate(uploaded_files):
        with columns[index % len(columns)]:
            st.image(uploaded, caption=uploaded.name, width="stretch")

with st.expander("可选：提供参考答案或补充说明"):
    reference_answer = st.text_area(
        "参考答案",
        placeholder="例如：第1题 24；第2题 3/4。没有可留空，模型会独立计算。",
        height=100,
    )
    extra_context = st.text_area(
        "补充说明",
        placeholder="例如：只批改红框内的题；铅笔字是学生答案，红笔字是老师批注。",
        height=90,
    )

can_analyze = bool(
    api_key and selected_model and uploaded_files and len(uploaded_files) <= 8
)
analyze_clicked = st.button(
    "开始批改",
    type="primary",
    width="stretch",
    disabled=not can_analyze,
)

if analyze_clicked:
    started_at = time.monotonic()
    progress = st.progress(0, text="0% · 正在检查上传文件")
    try:
        raw_files = [(item.name, item.getvalue()) for item in uploaded_files]
        progress.progress(8, text="8% · 正在校正图片方向和尺寸")
        images = prepare_images(raw_files, max_edge=max_edge)
        progress.progress(20, text="20% · 图片处理完成，正在上传至模型")

        is_fast = analysis_mode == "快速批改"
        api_events: queue.Queue[tuple[str, dict]] = queue.Queue()

        def report_api_progress(event: str, data: dict) -> None:
            api_events.put((event, data))

        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(
                analyze_exam,
                images,
                api_key=api_key,
                grade=grade,
                reference_answer=reference_answer,
                extra_context=extra_context,
                model=selected_model,
                max_tokens=3000 if is_fast else 6000,
                concise=is_fast,
                timeout_seconds=selected_timeout,
                stream=True,
                progress_callback=report_api_progress,
            )
            request_id = None
            server_trace_id = None
            headers_received = False
            first_content_received = False
            network_fallback_message = None
            while not future.done():
                while True:
                    try:
                        event, event_data = api_events.get_nowait()
                    except queue.Empty:
                        break
                    if event == "request_start":
                        request_id = event_data.get("request_id")
                    elif event == "response_headers":
                        headers_received = True
                        server_trace_id = event_data.get("trace_id")
                    elif event == "first_content":
                        first_content_received = True
                    elif event == "network_fallback":
                        from_route = event_data.get("from_route")
                        to_route = event_data.get("to_route")
                        route_names = {
                            "direct": "直连",
                            "system_proxy": "系统代理",
                        }
                        network_fallback_message = (
                            f"{route_names.get(from_route, from_route)}失败，正在通过"
                            f"{route_names.get(to_route, to_route)}重试"
                        )

                elapsed = time.monotonic() - started_at
                # API 不提供服务端百分比；在 20%～92% 间按等待时间平滑估算。
                estimated = min(92, 20 + int(elapsed * 1.2))
                if first_content_received:
                    phase = "已收到首个内容，正在持续接收批改结果"
                elif headers_received:
                    phase = "硅基流动已接收请求，模型正在识别和计算"
                elif network_fallback_message:
                    phase = network_fallback_message
                elif request_id:
                    phase = "正在连接硅基流动并提交图片"
                else:
                    phase = "正在创建 API 请求"
                diagnostic = server_trace_id or request_id
                diagnostic_text = f" · ID {diagnostic}" if diagnostic else ""
                progress.progress(
                    estimated,
                    text=(
                        f"{estimated}%（估算）· {phase} · 已等待 {elapsed:.0f} 秒"
                        f"{diagnostic_text}"
                    ),
                )
                time.sleep(0.5)
            result = future.result()

        elapsed = time.monotonic() - started_at
        progress.progress(100, text=f"100% · 批改完成，共用时 {elapsed:.1f} 秒")
        st.session_state["analysis_result"] = result
        st.session_state["analysis_elapsed"] = elapsed
        st.success("批改完成")
    except ExamAnalyzerError as exc:
        progress.empty()
        st.error(str(exc))
        st.caption("详细诊断已写入项目根目录的 api_requests.log。")
    except Exception:
        progress.empty()
        st.error("发生了未预期错误。请检查图片格式和配置后重试。")
        st.caption("详细诊断请查看项目根目录的 api_requests.log。")

result = st.session_state.get("analysis_result")
if result:
    st.divider()
    st.markdown(result.content)
    metadata = [f"模型：{result.model}"]
    if result.prompt_tokens is not None:
        metadata.append(f"输入 Token：{result.prompt_tokens}")
    if result.completion_tokens is not None:
        metadata.append(f"输出 Token：{result.completion_tokens}")
    if result.trace_id:
        metadata.append(f"Trace ID：{result.trace_id}")
    if result.request_id:
        metadata.append(f"本地请求 ID：{result.request_id}")
    if result.headers_seconds is not None:
        metadata.append(f"响应头：{result.headers_seconds:.1f} 秒")
    if result.first_content_seconds is not None:
        metadata.append(f"首内容：{result.first_content_seconds:.1f} 秒")
    if result.network_route:
        route_name = {
            "direct": "直连",
            "system_proxy": "系统代理",
        }.get(result.network_route, result.network_route)
        metadata.append(f"网络线路：{route_name}")
    elapsed = st.session_state.get("analysis_elapsed")
    if elapsed is not None:
        metadata.append(f"耗时：{elapsed:.1f} 秒")
    st.caption("｜".join(metadata))
    st.download_button(
        "下载批改结果（Markdown）",
        data=result.content,
        file_name="数学题批改结果.md",
        mime="text/markdown",
    )
