from __future__ import annotations

import asyncio
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.config import ConfigError, Settings  # noqa: E402
from app.deps import create_llm  # noqa: E402
from app.llm.base import EndEvent, LLMError, TokenEvent, UsageEvent  # noqa: E402


async def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    settings = Settings()
    try:
        settings.validate_runtime()
    except ConfigError as exc:
        print(f"配置错误：{exc}")
        return 2

    if settings.is_mock_llm:
        print("当前 LLM_PROVIDER=mock，将使用离线规则模型。")
        print("要测试真实 DeepSeek，请复制 .env.example 为 .env 并填写 LLM_API_KEY。")

    client = create_llm(settings)
    print(f"模型：{client.model}，接口：{settings.llm_base_url}")
    user_text = (
        "你好，请用一句中文问候。"
        if settings.is_mock_llm
        else "退换货政策一般关注哪些要点？"
    )
    started = time.perf_counter()
    text = ""
    usage = None
    try:
        async for event in client.stream_chat(
            [
                {
                    "role": "system",
                    "content": "你是悦家电器售后助手，请用一句简短中文回答。",
                },
                {"role": "user", "content": user_text},
            ]
        ):
            if isinstance(event, TokenEvent):
                text += event.text
                print(event.text, end="", flush=True)
            elif isinstance(event, UsageEvent):
                usage = event
            elif isinstance(event, EndEvent):
                print(f"\n[finish_reason={event.finish_reason}]")
    except LLMError as exc:
        print(f"\n调用失败：{exc}")
        return 1
    finally:
        await client.close()

    elapsed = (time.perf_counter() - started) * 1000
    print(f"耗时：{elapsed:.0f} ms")
    if usage:
        print(
            f"tokens：prompt={usage.prompt_tokens}, completion={usage.completion_tokens}, "
            f"total={usage.total_tokens}"
        )
    if not text.strip():
        print("模型没有返回文本。")
        return 1
    print("冒烟测试通过。")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
