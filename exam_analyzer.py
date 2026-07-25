"""硅基流动 Qwen3-VL 小学数学试卷分析核心逻辑。"""

from __future__ import annotations

import base64
import io
import json
import logging
import os
from collections.abc import Callable
from dataclasses import dataclass
from logging.handlers import RotatingFileHandler
from pathlib import Path
import time
from typing import Any, Iterable
import uuid

import requests
from PIL import Image, ImageOps, UnidentifiedImageError


DEFAULT_BASE_URL = "https://api.siliconflow.cn/v1"
DEFAULT_MODEL = "Qwen/Qwen3-VL-32B-Instruct"
MAX_IMAGE_BYTES = 15 * 1024 * 1024
MAX_IMAGES = 8
API_LOG_PATH = Path(__file__).with_name("api_requests.log")


class ExamAnalyzerError(RuntimeError):
    """可安全展示给页面用户的异常。"""


@dataclass(frozen=True)
class PreparedImage:
    filename: str
    data_url: str
    width: int
    height: int


@dataclass(frozen=True)
class AnalysisResult:
    content: str
    model: str
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    trace_id: str | None = None
    request_id: str | None = None
    headers_seconds: float | None = None
    first_content_seconds: float | None = None
    api_seconds: float | None = None
    network_route: str | None = None


ProgressCallback = Callable[[str, dict[str, Any]], None]


def _get_api_logger() -> logging.Logger:
    logger = logging.getLogger("exam_analyzer.api")
    if not logger.handlers:
        handler = RotatingFileHandler(
            API_LOG_PATH,
            maxBytes=2 * 1024 * 1024,
            backupCount=2,
            encoding="utf-8",
        )
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
    return logger


def _env_flag(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def recommended_timeout_seconds(model: str) -> int:
    """返回无响应数据时的建议读取超时；流式数据到达会重置该计时。"""
    lowered = model.lower()
    if "thinking" in lowered or lowered.startswith("pro/"):
        return 900
    if any(marker in lowered for marker in ("32b", "27b", "35b")):
        return 600
    if "30b" in lowered:
        return 420
    return 240


def _emit_progress(
    callback: ProgressCallback | None,
    event: str,
    **data: Any,
) -> None:
    if callback is not None:
        callback(event, data)


def prepare_image(
    file_bytes: bytes,
    filename: str,
    *,
    max_edge: int = 3000,
) -> PreparedImage:
    """校正方向、限制尺寸，并编码成 API 可接受的 Data URL。"""
    if not file_bytes:
        raise ExamAnalyzerError(f"{filename} 是空文件。")
    if len(file_bytes) > MAX_IMAGE_BYTES:
        raise ExamAnalyzerError(f"{filename} 超过 15 MB，请压缩后重试。")
    if not 1200 <= max_edge <= 3584:
        raise ValueError("max_edge 必须位于 1200 到 3584 之间。")

    try:
        with Image.open(io.BytesIO(file_bytes)) as source:
            image = ImageOps.exif_transpose(source)
            image.load()
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
        raise ExamAnalyzerError(f"{filename} 不是有效的图片文件。") from exc

    if image.width < 8 or image.height < 8:
        raise ExamAnalyzerError(f"{filename} 尺寸过小，无法识别题目。")

    # 常见的单行错题截图可能不足 56 px 高。尽量等比放大，避免直接拒绝。
    short_edge = min(image.width, image.height)
    if short_edge < 224:
        scale = min(224 / short_edge, max_edge / max(image.width, image.height))
        if scale > 1:
            image = image.resize(
                (
                    max(1, round(image.width * scale)),
                    max(1, round(image.height * scale)),
                ),
                Image.Resampling.LANCZOS,
            )

    image.thumbnail((max_edge, max_edge), Image.Resampling.LANCZOS)

    # API 对 Qwen 视觉图片的技术下限是 56×56；极长截图用白边补足短边。
    if image.width < 56 or image.height < 56:
        canvas = Image.new("RGB", (max(56, image.width), max(56, image.height)), "white")
        canvas.paste(
            image.convert("RGB"),
            ((canvas.width - image.width) // 2, (canvas.height - image.height) // 2),
        )
        image = canvas

    # JPEG 不支持透明通道；白底也更适合试卷 OCR。
    if image.mode in ("RGBA", "LA") or (
        image.mode == "P" and "transparency" in image.info
    ):
        rgba = image.convert("RGBA")
        white = Image.new("RGBA", rgba.size, "white")
        image = Image.alpha_composite(white, rgba).convert("RGB")
    elif image.mode != "RGB":
        image = image.convert("RGB")

    output = io.BytesIO()
    image.save(
        output,
        format="JPEG",
        quality=94,
        optimize=True,
        subsampling=0,
    )
    encoded = base64.b64encode(output.getvalue()).decode("ascii")
    return PreparedImage(
        filename=filename,
        data_url=f"data:image/jpeg;base64,{encoded}",
        width=image.width,
        height=image.height,
    )


def prepare_images(
    files: Iterable[tuple[str, bytes]], *, max_edge: int = 3000
) -> list[PreparedImage]:
    items = list(files)
    if not items:
        raise ExamAnalyzerError("请至少上传一张题目图片。")
    if len(items) > MAX_IMAGES:
        raise ExamAnalyzerError(f"一次最多上传 {MAX_IMAGES} 张图片。")
    return [
        prepare_image(content, filename, max_edge=max_edge)
        for filename, content in items
    ]


def build_system_prompt() -> str:
    return """你是一位严谨、耐心的小学数学老师，负责批改学生拍照上传的数学题。

工作原则：
1. 逐题读取题干、图形、单位、选项和学生的手写/印刷作答，然后独立计算验证。
2. 不要只凭红勾、红叉或参考答案判断；教师批注只能作为辅助线索。
3. 状态只能是：正确、错误、未作答、无法辨认。看不清时不要猜，明确指出需重拍的位置。
4. 对错误题说明具体错因，并给出正确答案和适合小学生理解的简明订正步骤。
5. 区分“学生答案”和“题目中原有的印刷内容”；不要把草稿或批注误当作最终答案。
6. 计算题需核对运算过程；应用题需核对列式、单位和答句；几何题需核对图形条件。
7. 只给必要的解题依据和可检查的步骤，不输出冗长的内心推理过程。
8. 若图片里有多道题，必须逐题批改，不得只分析其中一道。

请始终使用简体中文回答。"""


def build_user_prompt(
    *,
    grade: str,
    reference_answer: str = "",
    extra_context: str = "",
    concise: bool = False,
) -> str:
    reference = reference_answer.strip() or "未提供，请你独立求解核对"
    context = extra_context.strip() or "无"
    speed_requirement = (
        "当前为快速批改模式：判断依据控制在 1 句话，订正步骤最多 3 步，避免重复题干和冗长说明。"
        if concise
        else "当前为详细分析模式：在保持清晰的前提下说明关键核对过程。"
    )
    return f"""请批改以上图片中的小学数学题。

学生年级：{grade}
用户提供的参考答案或教师说明：{reference}
其他说明：{context}
输出要求：{speed_requirement}

请严格按以下结构输出 Markdown：

# 批改结论
- 共识别到：X 题
- 正确：X 题
- 错误：X 题
- 未作答：X 题
- 无法辨认：X 题

# 逐题批改
## 第 1 题（状态：正确/错误/未作答/无法辨认）
- 题目：准确转述题意；过长时可概括，但数字和单位必须完整
- 学生答案：图片中识别到的答案；没有则写“未作答”
- 正确答案：给出最终答案；无法辨认时写“无法确定”
- 判断依据：用 1～3 句话说明核对方法
- 错误原因：错误题要具体说明；其他状态写“无”
- 订正步骤：错误题给出清晰步骤；正确题可写更简洁的方法
- 知识点：对应的小学数学知识点

（按图片中的实际题数继续编号）

# 学习建议
针对本次错误总结 1～3 条可执行的练习建议。若没有错题，就说明保持正确率的方法。

重要要求：
- 数量统计必须与逐题状态一致。
- 不确定题号时按从上到下、从左到右编号。
- 如果学生答案被遮挡、裁切或字迹不清，状态用“无法辨认”，并说明如何重拍。
"""


def _extract_error_message(response: requests.Response) -> str:
    status_messages = {
        400: "请求格式或图片不符合模型要求。",
        401: "API 密钥无效或已失效。",
        403: "当前账户无权调用该模型，请检查实名认证、余额和模型权限。",
        404: "接口或模型不存在，请检查模型名称。",
        413: "上传内容过大，请减少图片数量或尺寸。",
        429: "调用过于频繁或额度不足，请稍后重试并检查账户余额。",
        503: "模型服务暂时繁忙，请稍后重试。",
        504: "模型处理超时，请减少图片数量后重试。",
    }
    fallback = status_messages.get(response.status_code, f"API 返回 HTTP {response.status_code}。")
    try:
        payload = response.json()
        api_message = payload.get("message")
        if not api_message and isinstance(payload.get("error"), dict):
            api_message = payload["error"].get("message")
        if isinstance(api_message, str) and api_message.strip():
            return f"{fallback} 服务信息：{api_message.strip()}"
    except (ValueError, AttributeError):
        pass
    return fallback


def _normalize_content(content: Any) -> str:
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        texts: list[str] = []
        for item in content:
            if isinstance(item, dict) and isinstance(item.get("text"), str):
                texts.append(item["text"])
        return "\n".join(texts).strip()
    return ""


def _read_streaming_response(
    response: requests.Response,
    *,
    started_at: float,
    callback: ProgressCallback | None,
    logger: logging.Logger,
    request_id: str,
) -> tuple[str, dict[str, Any], float | None]:
    parts: list[str] = []
    usage: dict[str, Any] = {}
    first_content_seconds: float | None = None

    # 不依赖响应头的 charset。部分 SSE 网关没有声明 UTF-8，Requests 会按
    # ISO-8859-1 解码，导致中文变成“æ…”形式的乱码。
    for raw_line in response.iter_lines(decode_unicode=False):
        if isinstance(raw_line, bytes):
            line = raw_line.decode("utf-8", errors="replace").strip()
        else:
            line = (raw_line or "").strip()
        if not line or not line.startswith("data:"):
            continue
        event_data = line[5:].strip()
        if event_data == "[DONE]":
            break
        try:
            chunk = json.loads(event_data)
        except json.JSONDecodeError:
            # 某些网关会发送非 JSON 的心跳事件；忽略即可，不能视为请求失败。
            logger.debug("request_id=%s ignored_non_json_sse_event", request_id)
            continue

        chunk_usage = chunk.get("usage")
        if isinstance(chunk_usage, dict):
            usage = chunk_usage

        choices = chunk.get("choices")
        if not isinstance(choices, list) or not choices:
            continue
        delta = choices[0].get("delta")
        if not isinstance(delta, dict):
            continue
        raw_content = delta.get("content")
        piece = raw_content if isinstance(raw_content, str) else _normalize_content(raw_content)
        if not piece:
            continue
        if first_content_seconds is None:
            first_content_seconds = time.monotonic() - started_at
            _emit_progress(
                callback,
                "first_content",
                seconds=first_content_seconds,
                request_id=request_id,
            )
            logger.info(
                "request_id=%s first_content_seconds=%.3f",
                request_id,
                first_content_seconds,
            )
        parts.append(piece)

    return "".join(parts).strip(), usage, first_content_seconds


def _diagnostic_suffix(request_id: str, trace_id: str | None = None) -> str:
    values = [f"本地请求 ID：{request_id}"]
    if trace_id:
        values.append(f"硅基流动 Trace ID：{trace_id}")
    return "（" + "；".join(values) + "）"


def analyze_exam(
    images: list[PreparedImage],
    *,
    api_key: str,
    grade: str,
    reference_answer: str = "",
    extra_context: str = "",
    base_url: str | None = None,
    model: str | None = None,
    timeout_seconds: int | None = None,
    max_tokens: int = 6000,
    concise: bool = False,
    stream: bool = True,
    use_system_proxy: bool | None = None,
    auto_network_fallback: bool | None = None,
    progress_callback: ProgressCallback | None = None,
) -> AnalysisResult:
    """调用硅基流动 Chat Completions 接口分析试题。"""
    if not api_key.strip():
        raise ExamAnalyzerError("未配置 SILICONFLOW_API_KEY。")
    if not images:
        raise ExamAnalyzerError("没有可分析的图片。")

    api_base = (base_url or os.getenv("SILICONFLOW_BASE_URL") or DEFAULT_BASE_URL).rstrip("/")
    model_name = model or os.getenv("SILICONFLOW_MODEL") or DEFAULT_MODEL
    read_timeout = timeout_seconds or recommended_timeout_seconds(model_name)
    proxy_enabled = (
        _env_flag("SILICONFLOW_USE_SYSTEM_PROXY", False)
        if use_system_proxy is None
        else use_system_proxy
    )
    fallback_enabled = (
        _env_flag("SILICONFLOW_AUTO_NETWORK_FALLBACK", True)
        if auto_network_fallback is None
        else auto_network_fallback
    )
    request_id = uuid.uuid4().hex[:12]
    started_at = time.monotonic()
    logger = _get_api_logger()
    image_dimensions = ",".join(f"{image.width}x{image.height}" for image in images)
    logger.info(
        "request_id=%s start model=%s images=%d dimensions=%s stream=%s "
        "system_proxy=%s auto_fallback=%s read_timeout=%s",
        request_id,
        model_name,
        len(images),
        image_dimensions,
        stream,
        proxy_enabled,
        fallback_enabled,
        read_timeout,
    )
    _emit_progress(
        progress_callback,
        "request_start",
        request_id=request_id,
        model=model_name,
        timeout_seconds=read_timeout,
        system_proxy=proxy_enabled,
        auto_network_fallback=fallback_enabled,
    )

    user_content: list[dict[str, Any]] = []
    for image in images:
        user_content.append(
            {
                "type": "image_url",
                "image_url": {"url": image.data_url, "detail": "high"},
            }
        )
    user_content.append(
        {
            "type": "text",
            "text": build_user_prompt(
                grade=grade,
                reference_answer=reference_answer,
                extra_context=extra_context,
                concise=concise,
            ),
        }
    )

    payload = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": build_system_prompt()},
            {"role": "user", "content": user_content},
        ],
        "stream": stream,
        "max_tokens": max_tokens,
        "temperature": 0.1,
        "top_p": 0.8,
    }
    headers = {
        "Authorization": f"Bearer {api_key.strip()}",
        "Content-Type": "application/json",
    }

    api_url = f"{api_base}/chat/completions"
    session = requests.Session()
    session.trust_env = proxy_enabled
    network_route = "system_proxy" if proxy_enabled else "direct"
    response: requests.Response | None = None
    trace_id: str | None = None
    headers_seconds: float | None = None
    first_content_seconds: float | None = None
    usage: dict[str, Any] = {}
    response_model = model_name

    try:
        try:
            response = session.post(
                api_url,
                headers=headers,
                json=payload,
                stream=stream,
                timeout=(15, read_timeout),
            )
        except (requests.ConnectionError, requests.ConnectTimeout) as first_exc:
            environment_proxies = requests.utils.get_environ_proxies(api_url)
            fallback_available = proxy_enabled or bool(environment_proxies)
            if not fallback_enabled or not fallback_available:
                raise

            fallback_proxy_enabled = not proxy_enabled
            fallback_route = (
                "system_proxy" if fallback_proxy_enabled else "direct"
            )
            logger.warning(
                "request_id=%s network_fallback from=%s to=%s error_type=%s "
                "detail=%s",
                request_id,
                network_route,
                fallback_route,
                type(first_exc).__name__,
                str(first_exc)[:500].replace("\n", " "),
            )
            _emit_progress(
                progress_callback,
                "network_fallback",
                request_id=request_id,
                from_route=network_route,
                to_route=fallback_route,
                error_type=type(first_exc).__name__,
            )
            session.close()
            session = requests.Session()
            session.trust_env = fallback_proxy_enabled
            proxy_enabled = fallback_proxy_enabled
            network_route = fallback_route
            response = session.post(
                api_url,
                headers=headers,
                json=payload,
                stream=stream,
                timeout=(15, read_timeout),
            )
        headers_seconds = time.monotonic() - started_at
        trace_id = response.headers.get("x-siliconcloud-trace-id")
        logger.info(
            "request_id=%s headers status=%s seconds=%.3f trace_id=%s",
            request_id,
            response.status_code,
            headers_seconds,
            trace_id or "none",
        )
        _emit_progress(
            progress_callback,
            "response_headers",
            request_id=request_id,
            status_code=response.status_code,
            seconds=headers_seconds,
            trace_id=trace_id,
            network_route=network_route,
        )

        if response.status_code != 200:
            message = _extract_error_message(response)
            logger.warning(
                "request_id=%s http_error status=%s trace_id=%s message=%s",
                request_id,
                response.status_code,
                trace_id or "none",
                message,
            )
            raise ExamAnalyzerError(
                message + " " + _diagnostic_suffix(request_id, trace_id)
            )

        content_type = response.headers.get("content-type", "").lower()
        if stream and "text/event-stream" in content_type:
            content, usage, first_content_seconds = _read_streaming_response(
                response,
                started_at=started_at,
                callback=progress_callback,
                logger=logger,
                request_id=request_id,
            )
        else:
            data = response.json()
            content = _normalize_content(data["choices"][0]["message"]["content"])
            usage = data.get("usage") if isinstance(data.get("usage"), dict) else {}
            response_model = str(data.get("model") or model_name)
    except requests.ConnectTimeout as exc:
        logger.error("request_id=%s connect_timeout", request_id)
        raise ExamAnalyzerError(
            "连接硅基流动超时，请检查本机代理、DNS 和网络；API 很可能尚未收到请求。"
            + " "
            + _diagnostic_suffix(request_id)
        ) from exc
    except requests.ReadTimeout as exc:
        logger.error(
            "request_id=%s read_timeout trace_id=%s read_timeout=%s",
            request_id,
            trace_id or "none",
            read_timeout,
        )
        if trace_id:
            message = (
                f"硅基流动已接收请求，但连续 {read_timeout} 秒没有返回新数据。"
            )
        else:
            message = (
                f"请求等待超过 {read_timeout} 秒，未取得服务端响应头；"
                "可能卡在本机代理、网络或硅基流动网关排队阶段。"
            )
        raise ExamAnalyzerError(
            message + " " + _diagnostic_suffix(request_id, trace_id)
        ) from exc
    except requests.Timeout as exc:
        logger.error("request_id=%s timeout trace_id=%s", request_id, trace_id or "none")
        raise ExamAnalyzerError(
            "请求超时。" + " " + _diagnostic_suffix(request_id, trace_id)
        ) from exc
    except ExamAnalyzerError:
        raise
    except requests.RequestException as exc:
        logger.error(
            "request_id=%s request_error type=%s route=%s trace_id=%s detail=%s",
            request_id,
            type(exc).__name__,
            network_route,
            trace_id or "none",
            str(exc)[:500].replace("\n", " "),
        )
        raise ExamAnalyzerError(
            f"无法连接硅基流动 API：{type(exc).__name__}；"
            f"最后尝试线路：{network_route}。 "
            + _diagnostic_suffix(request_id, trace_id)
        ) from exc
    except (ValueError, KeyError, IndexError, TypeError) as exc:
        logger.error(
            "request_id=%s parse_error type=%s trace_id=%s",
            request_id,
            type(exc).__name__,
            trace_id or "none",
        )
        raise ExamAnalyzerError(
            "API 返回了无法解析的响应。 " + _diagnostic_suffix(request_id, trace_id)
        ) from exc
    finally:
        if response is not None:
            response.close()
        session.close()

    if not content:
        logger.error("request_id=%s empty_content trace_id=%s", request_id, trace_id or "none")
        raise ExamAnalyzerError(
            "模型未返回批改结果，请重试。 " + _diagnostic_suffix(request_id, trace_id)
        )

    api_seconds = time.monotonic() - started_at
    logger.info(
        "request_id=%s complete seconds=%.3f trace_id=%s chars=%d",
        request_id,
        api_seconds,
        trace_id or "none",
        len(content),
    )
    _emit_progress(
        progress_callback,
        "complete",
        request_id=request_id,
        seconds=api_seconds,
        trace_id=trace_id,
    )
    return AnalysisResult(
        content=content,
        model=response_model,
        prompt_tokens=usage.get("prompt_tokens"),
        completion_tokens=usage.get("completion_tokens"),
        trace_id=trace_id,
        request_id=request_id,
        headers_seconds=headers_seconds,
        first_content_seconds=first_content_seconds,
        api_seconds=api_seconds,
        network_route=network_route,
    )
