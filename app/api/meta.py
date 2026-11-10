from __future__ import annotations

from fastapi import APIRouter, Request

from app.api.deps import get_container

router = APIRouter(tags=["system"])

_SUGGESTIONS = [
    "订单 A2026001 物流三天没更新，能退货吗？",
    "七天无理由退货有什么条件？",
    "订单 A2026001 物流异常，帮我建工单。",
    "查一下订单 A2026011 现在到哪了？",
    "发票多久能开出来？",
    "退款一般多久到账？",
]


@router.get("/meta")
async def meta(request: Request) -> dict:
    settings = get_container(request).settings
    return {
        "app": settings.app_name,
        "mode": "mock" if settings.is_mock_llm else "live",
        "mode_label": (
            "离线演示模式"
            if settings.is_mock_llm
            else f"DeepSeek · {settings.llm_model}"
        ),
        "embedding": (
            f"hash-ngram-{settings.embedding_dim}"
            if settings.embedding_provider == "hash"
            else settings.embedding_model
        ),
        "demo_customer_id": settings.demo_customer_id,
        "admin_auth_enabled": bool(settings.admin_token_value),
        "suggestions": _SUGGESTIONS,
        "limits": {
            "message_chars": settings.session_max_chars,
            "upload_mb": settings.max_upload_mb,
            "agent_steps": settings.max_agent_steps,
        },
    }


__all__ = ["router"]
