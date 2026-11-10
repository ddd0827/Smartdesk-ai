from __future__ import annotations

import base64
import json

import httpx
from fastapi.testclient import TestClient
from pydantic import SecretStr

from app.config import Settings
from app.llm.fake import FakeLLM
from app.main import create_app


def sse_events(response) -> list[tuple[str, dict]]:
    events: list[tuple[str, dict]] = []
    event = "message"
    data: list[str] = []
    for raw_line in response.iter_lines():
        line = raw_line.rstrip("\r")
        if line == "":
            if data:
                events.append((event, json.loads("\n".join(data))))
            event = "message"
            data = []
        elif line.startswith(":"):
            continue
        elif line.startswith("event:"):
            event = line[6:].strip()
        elif line.startswith("data:"):
            data.append(line[5:].strip())
    if data:
        events.append((event, json.loads("\n".join(data))))
    return events


def run_chat(
    client: TestClient, message: str, session_id: str | None = None
) -> list[tuple[str, dict]]:
    payload: dict = {"message": message}
    if session_id:
        payload["session_id"] = session_id
    with client.stream("POST", "/api/chat/stream", json=payload) as response:
        assert response.status_code == 200, response.text
        return sse_events(response)


def test_optional_basic_auth_gate(settings: Settings) -> None:
    secure = settings.model_copy(
        update={
            "auto_seed": True,
            "access_username": "demo",
            "access_password": SecretStr("demo-pass"),
        }
    )
    app = create_app(secure, llm=FakeLLM())
    good = "Basic " + base64.b64encode(b"demo:demo-pass").decode()
    bad = "Basic " + base64.b64encode(b"demo:wrong").decode()
    with TestClient(app) as client:
        assert client.get("/api/health").status_code == 200
        unauthenticated = client.get("/")
        assert unauthenticated.status_code == 401
        assert unauthenticated.headers["WWW-Authenticate"].startswith("Basic")
        assert client.get("/", headers={"Authorization": bad}).status_code == 401
        assert client.get("/", headers={"Authorization": good}).status_code == 200
        assert client.get("/api/sessions").status_code == 401
        assert (
            client.get(
                "/api/sessions", headers={"Authorization": good}
            ).status_code
            == 200
        )


def test_health_meta_documents_and_stats(client: TestClient) -> None:
    health = client.get("/api/health").json()
    assert health["status"] == "ok"
    assert health["database"] is True
    assert "llm_api_key" not in health["config"]

    meta = client.get("/api/meta").json()
    assert meta["mode"] == "mock"
    assert meta["suggestions"]
    assert meta["limits"]["message_chars"] > 0

    documents = client.get("/api/documents").json()
    assert documents["count"] == 8
    assert documents["chunks"] == 41

    stats = client.get("/api/stats").json()
    assert stats["documents"] == 8
    assert stats["chunks"] == 41


def test_chat_sse_and_session_history(client: TestClient) -> None:
    events = run_chat(client, "七天无理由退货有什么条件？")
    names = [name for name, _ in events]
    assert names[0] == "session"
    assert names.count("done") + names.count("error") == 1
    assert "done" in names

    session_event = events[0][1]
    session_id = session_event["session_id"]
    assert session_event["message_id"]

    text = "".join(data["delta"] for name, data in events if name == "token")
    assert text
    assert any(name == "sources" for name, _ in events)
    assert any(
        name == "trace" and data.get("tool") == "search_knowledge_base"
        for name, data in events
    )
    done = next(data for name, data in events if name == "done")
    assert done["invalid_citations"] == []

    history = client.get(f"/api/sessions/{session_id}/messages").json()
    assert len(history["messages"]) == 2
    assert history["messages"][0]["role"] == "user"
    assistant = history["messages"][1]
    assert assistant["role"] == "assistant"
    assert assistant["content"]
    assert assistant["meta"]["sources"]
    assert assistant["meta"]["trace"]


def test_order_ticket_flow_and_idempotency(client: TestClient) -> None:
    container = client.app.state.container
    ticket_tool = container.tools.get("create_ticket")
    assert ticket_tool is not None
    client.portal.call(
        ticket_tool.set_transport, httpx.ASGITransport(app=client.app)
    )

    events = run_chat(client, "订单 A2026001 物流异常，帮我建工单。")
    session_id = events[0][1]["session_id"]
    tools = [
        data["tool"]
        for name, data in events
        if name == "trace" and data.get("phase") == "end"
    ]
    assert "lookup_order" in tools
    assert "search_knowledge_base" in tools
    assert "create_ticket" in tools
    assert sum(1 for name, _ in events if name == "done") == 1
    assert client.get("/api/stats").json()["tickets"] == 1

    second = run_chat(
        client, "订单 A2026001 物流异常，帮我建工单。", session_id
    )
    second_summaries = [
        data.get("summary", "")
        for name, data in second
        if name == "trace" and data.get("phase") == "end"
    ]
    assert any("重复请求" in summary for summary in second_summaries)
    assert client.get("/api/stats").json()["tickets"] == 1


def test_upload_requires_admin_token_and_deduplicates(
    settings: Settings,
) -> None:
    secure = settings.model_copy(
        update={"auto_seed": True, "admin_token": SecretStr("secret")}
    )
    app = create_app(secure, llm=FakeLLM())
    content = "# 新的售后说明\n\n## 规则\n\n这是新增的演示文档。".encode()
    with TestClient(app) as client:
        response = client.post(
            "/api/documents",
            files={"file": ("new.md", content, "text/markdown")},
        )
        assert response.status_code == 401

        headers = {"X-Admin-Token": "secret"}
        response = client.post(
            "/api/documents",
            files={"file": ("new.md", content, "text/markdown")},
            headers=headers,
        )
        assert response.status_code == 200
        data = response.json()
        assert data["chunks"] >= 1
        assert data["duplicate"] is False
        doc_id = data["doc_id"]

        duplicate = client.post(
            "/api/documents",
            files={"file": ("new.md", content, "text/markdown")},
            headers=headers,
        )
        assert duplicate.status_code == 200
        assert duplicate.json()["duplicate"] is True
        assert client.get("/api/documents").json()["count"] == 9

        reindex = client.post("/api/documents/reindex", headers=headers)
        assert reindex.status_code == 200
        assert reindex.json()["documents"] >= 9

        deleted = client.delete(f"/api/documents/{doc_id}", headers=headers)
        assert deleted.status_code == 200
        assert client.get("/api/documents").json()["count"] == 8


def test_upload_rejects_bad_type_and_oversized_file(
    settings: Settings,
) -> None:
    small = settings.model_copy(
        update={"auto_seed": False, "max_upload_mb": 1}
    )
    app = create_app(small, llm=FakeLLM())
    with TestClient(app) as client:
        wrong_type = client.post(
            "/api/documents",
            files={"file": ("bad.exe", b"data", "application/octet-stream")},
        )
        assert wrong_type.status_code == 415
        oversized = client.post(
            "/api/documents",
            files={"file": ("big.txt", b"x" * (1024 * 1024 + 1), "text/plain")},
        )
        assert oversized.status_code == 413


def test_feedback_and_message_validation(client: TestClient) -> None:
    events = run_chat(client, "发票多久能开出来？")
    session_id = events[0][1]["session_id"]
    done = next(data for name, data in events if name == "done")
    message_id = done["message_id"]

    response = client.post(
        "/api/feedback", json={"message_id": message_id, "rating": 1}
    )
    assert response.status_code == 200
    assert client.get("/api/stats").json()["feedback_positive"] == 1

    history = client.get(f"/api/sessions/{session_id}/messages").json()
    assert history["messages"][1]["feedback"]["rating"] == 1
    user_message_id = history["messages"][0]["id"]
    assert (
        client.post(
            "/api/feedback",
            json={"message_id": user_message_id, "rating": 1},
        ).status_code
        == 400
    )
    assert (
        client.post(
            "/api/feedback", json={"message_id": "missing-id", "rating": 1}
        ).status_code
        == 404
    )

    assert client.post("/api/chat/stream", json={"message": "   "}).status_code == 400
    assert (
        client.post(
            "/api/chat/stream", json={"message": "长" * 2001}
        ).status_code
        == 400
    )


def test_session_not_found_and_delete(client: TestClient) -> None:
    assert client.get("/api/sessions/nope/messages").status_code == 404
    assert client.delete("/api/sessions/nope").status_code == 404

    events = run_chat(client, "保修期多久？")
    session_id = events[0][1]["session_id"]
    assert client.delete(f"/api/sessions/{session_id}").status_code == 200
    assert client.get(f"/api/sessions/{session_id}/messages").status_code == 404


def test_mock_itsm_endpoint_auth_and_idempotency(client: TestClient) -> None:
    events = run_chat(client, "你好")
    session_id = events[0][1]["session_id"]
    payload = {
        "order_id": "A2026001",
        "session_id": session_id,
        "summary": "物流异常需要人工跟进",
        "priority": "normal",
    }
    assert client.post("/api/mock/itsm/tickets", json=payload).status_code == 401
    assert (
        client.post(
            "/api/mock/itsm/tickets",
            json=payload,
            headers={"X-API-Key": "demo-itsm-key"},
        ).status_code
        == 422
    )
    headers = {
        "X-API-Key": "demo-itsm-key",
        "Idempotency-Key": "idem-api-test",
    }
    first = client.post("/api/mock/itsm/tickets", json=payload, headers=headers)
    assert first.status_code == 201
    ticket_id = first.json()["ticket_id"]
    second = client.post("/api/mock/itsm/tickets", json=payload, headers=headers)
    assert second.status_code == 200
    assert second.json()["duplicate"] is True
    assert second.json()["ticket_id"] == ticket_id
    assert client.get("/api/stats").json()["tickets"] == 1


def test_secret_values_do_not_appear_in_meta(client: TestClient) -> None:
    body = json.dumps(client.get("/api/meta").json(), ensure_ascii=False)
    assert "demo-itsm-key" not in body
    assert "test-key" not in body
