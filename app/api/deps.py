from __future__ import annotations

import secrets

from fastapi import Header, HTTPException, Request

from app.deps import AppContainer


def get_container(request: Request) -> AppContainer:
    return request.app.state.container


async def require_admin(
    request: Request,
    x_admin_token: str | None = Header(default=None, alias="X-Admin-Token"),
) -> None:
    token = get_container(request).settings.admin_token_value
    if not token:
        return
    if not x_admin_token or not secrets.compare_digest(x_admin_token, token):
        raise HTTPException(
            status_code=401,
            detail="需要管理员令牌",
            headers={"WWW-Authenticate": "X-Admin-Token"},
        )


__all__ = ["get_container", "require_admin"]
