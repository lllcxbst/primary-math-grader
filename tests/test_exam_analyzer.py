from __future__ import annotations

import io
from unittest.mock import Mock, patch

import pytest
import requests
from PIL import Image

from exam_analyzer import (
    ExamAnalyzerError,
    PreparedImage,
    analyze_exam,
    build_user_prompt,
    prepare_image,
    recommended_timeout_seconds,
)


def make_image_bytes(size=(4000, 2000), mode="RGB") -> bytes:
    image = Image.new(mode, size, "white")
    output = io.BytesIO()
    image.save(output, format="PNG")
    return output.getvalue()


def test_prepare_image_resizes_and_builds_data_url():
    result = prepare_image(make_image_bytes(), "math.png", max_edge=2000)
    assert result.filename == "math.png"
    assert result.width == 2000
    assert result.height == 1000
    assert result.data_url.startswith("data:image/jpeg;base64,")


def test_prepare_image_rejects_non_image():
    with pytest.raises(ExamAnalyzerError, match="不是有效"):
        prepare_image(b"not an image", "bad.jpg")


def test_prepare_image_upscales_short_screenshot():
    result = prepare_image(make_image_bytes(size=(1200, 40)), "strip.png", max_edge=1600)
    assert result.width == 1600
    assert result.height >= 56


def test_prompt_contains_grading_requirements():
    prompt = build_user_prompt(
        grade="五年级",
        reference_answer="第1题：12",
        extra_context="只看红框",
    )
    assert "五年级" in prompt
    assert "第1题：12" in prompt
    assert "正确/错误/未作答/无法辨认" in prompt


def test_fast_prompt_requests_concise_output():
    prompt = build_user_prompt(grade="三年级", concise=True)
    assert "快速批改模式" in prompt
    assert "订正步骤最多 3 步" in prompt


def test_recommended_timeout_grows_for_large_models():
    assert recommended_timeout_seconds("Qwen/Qwen3-VL-8B-Instruct") == 240
    assert recommended_timeout_seconds("Qwen/Qwen3-VL-32B-Instruct") == 600
    assert recommended_timeout_seconds("Qwen/Qwen3-VL-32B-Thinking") == 900
    assert recommended_timeout_seconds("Qwen/Qwen3.5-122B-A10B") == 900
    assert recommended_timeout_seconds("Qwen/Qwen3.5-397B-A17B") == 1200
    assert recommended_timeout_seconds("zai-org/GLM-4.5V") == 900


@patch("exam_analyzer.requests.Session.post")
def test_analyze_exam_calls_siliconflow(mock_post):
    response = Mock()
    response.status_code = 200
    response.headers = {
        "x-siliconcloud-trace-id": "trace-123",
        "content-type": "application/json",
    }
    response.json.return_value = {
        "model": "Qwen/Qwen3-VL-32B-Instruct",
        "choices": [{"message": {"content": "# 批改结论\n正确：1题"}}],
        "usage": {"prompt_tokens": 100, "completion_tokens": 50},
    }
    mock_post.return_value = response
    image = PreparedImage("one.jpg", "data:image/jpeg;base64,AA==", 100, 100)

    result = analyze_exam(
        [image],
        api_key="test-key",
        grade="四年级",
        base_url="https://api.example.test/v1",
        stream=False,
    )

    assert result.content.startswith("# 批改结论")
    assert result.trace_id == "trace-123"
    args, kwargs = mock_post.call_args
    assert args[0] == "https://api.example.test/v1/chat/completions"
    assert kwargs["headers"]["Authorization"] == "Bearer test-key"
    assert kwargs["json"]["model"] == "Qwen/Qwen3-VL-32B-Instruct"
    assert kwargs["json"]["max_tokens"] == 6000
    assert kwargs["json"]["stream"] is False
    assert kwargs["stream"] is False
    assert kwargs["json"]["messages"][1]["content"][0]["image_url"]["detail"] == "high"


@patch("exam_analyzer.requests.Session.post")
def test_analyze_exam_shows_friendly_unauthorized_error(mock_post):
    response = Mock()
    response.status_code = 401
    response.json.return_value = {"message": "Unauthorized"}
    response.headers = {"content-type": "application/json"}
    mock_post.return_value = response
    image = PreparedImage("one.jpg", "data:image/jpeg;base64,AA==", 100, 100)

    with pytest.raises(ExamAnalyzerError, match="API 密钥无效"):
        analyze_exam([image], api_key="bad-key", grade="四年级", stream=False)


@patch("exam_analyzer.requests.Session.post")
def test_analyze_exam_shows_friendly_payment_required_error(mock_post):
    response = Mock()
    response.status_code = 402
    response.json.return_value = {"message": "Insufficient balance"}
    response.headers = {"content-type": "application/json"}
    mock_post.return_value = response
    image = PreparedImage("one.jpg", "data:image/jpeg;base64,AA==", 100, 100)

    with pytest.raises(ExamAnalyzerError, match="账户余额或付费额度不足"):
        analyze_exam([image], api_key="test-key", grade="四年级", stream=False)


@patch("exam_analyzer.requests.Session.post")
def test_analyze_exam_streams_content_and_reports_progress(mock_post):
    response = Mock()
    response.status_code = 200
    response.headers = {
        "x-siliconcloud-trace-id": "trace-stream",
        "content-type": "text/event-stream; charset=utf-8",
    }
    response.iter_lines.return_value = [
        'data: {"choices":[{"delta":{"content":"# 批改"}}]}'.encode("utf-8"),
        'data: {"choices":[{"delta":{"content":"结论\\n正确"}}]}'.encode("utf-8"),
        b"data: [DONE]",
    ]
    mock_post.return_value = response
    image = PreparedImage("one.jpg", "data:image/jpeg;base64,AA==", 100, 100)
    events = []

    result = analyze_exam(
        [image],
        api_key="test-key",
        grade="四年级",
        stream=True,
        progress_callback=lambda event, data: events.append((event, data)),
    )

    assert result.content == "# 批改结论\n正确"
    assert "æ" not in result.content
    assert result.trace_id == "trace-stream"
    assert result.request_id
    assert [event for event, _ in events] == [
        "request_start",
        "response_headers",
        "first_content",
        "complete",
    ]
    assert mock_post.call_args.kwargs["stream"] is True


@patch("exam_analyzer.requests.Session.post")
def test_connect_timeout_says_api_may_not_have_received_request(mock_post):
    mock_post.side_effect = requests.ConnectTimeout("connect timeout")
    image = PreparedImage("one.jpg", "data:image/jpeg;base64,AA==", 100, 100)

    with pytest.raises(ExamAnalyzerError, match="API 很可能尚未收到请求"):
        analyze_exam([image], api_key="test-key", grade="四年级")


@patch("exam_analyzer.requests.Session.post")
def test_read_timeout_after_headers_includes_trace_id(mock_post):
    response = Mock()
    response.status_code = 200
    response.headers = {
        "x-siliconcloud-trace-id": "trace-timeout",
        "content-type": "text/event-stream",
    }
    response.iter_lines.side_effect = requests.ReadTimeout("read timeout")
    mock_post.return_value = response
    image = PreparedImage("one.jpg", "data:image/jpeg;base64,AA==", 100, 100)

    with pytest.raises(ExamAnalyzerError, match="硅基流动已接收请求") as exc_info:
        analyze_exam([image], api_key="test-key", grade="四年级", timeout_seconds=30)

    assert "trace-timeout" in str(exc_info.value)
