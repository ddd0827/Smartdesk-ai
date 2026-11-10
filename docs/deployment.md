# 部署指南（本机 / Docker / 云主机）

> 本仓库开发环境为 Windows 11 + Python 3.11.8，**本机未安装 Docker**。GitHub Actions 已真实执行 `docker build` 并启动容器完成健康检查/种子数据/SSE 烟测（运行记录见仓库 Actions 页面）；下面的本机/云主机步骤仍应在目标环境实际验证和压测。

## 1. 本机运行

```bash
python -m venv .venv
# Windows
.venv/Scripts/pip install -r requirements-dev.txt
# macOS / Linux
# source .venv/bin/activate && pip install -r requirements-dev.txt

# 离线演示（默认 Settings 就是 mock，无需 .env）
python scripts/seed.py
uvicorn app.main:app --host 127.0.0.1 --port 8000
```

接入 DeepSeek：

```bash
cp .env.example .env
# 编辑 .env：
# LLM_PROVIDER=deepseek
# LLM_API_KEY=sk-...
# LLM_MODEL=deepseek-v4-flash
python scripts/smoke.py
uvicorn app.main:app --host 127.0.0.1 --port 8000
```

- 旧 `deepseek-chat` 已停用；模型名变化时只改 `LLM_MODEL`。
- 工具调用模式下 thinking 会被强制关闭，避免 `reasoning_content` 回传问题。
- 生产环境必须设置 `ADMIN_TOKEN`，否则启动会报错（`APP_ENV=production`）。

## 1.1 临时公网演示（隧道 + 访问密码）

适合把链接发给面试官或朋友，不需要路由器端口映射。**不要把没有访问密码的实例直接暴露到公网**：任何人都会消耗你的 DeepSeek 余额。

先在 `.env` 设置：

```dotenv
# 访问密码（浏览器会弹 HTTP Basic 登录，用户名默认 demo）
ACCESS_USERNAME=demo
ACCESS_PASSWORD=换成一个随机长密码
# 管理端点额外令牌；查看者聊天不需要它
ADMIN_TOKEN=换成一个随机长令牌
```

重启服务。健康检查 `/api/health` 不要求认证，其余页面/API 都需要。

### 方式 A：Cloudflare Tunnel（最快）

```bash
# Windows 可先试 winget；没有 winget 就下载官方 cloudflared.exe
winget install --id Cloudflare.cloudflared -e

# 保持服务在本机 127.0.0.1:8000，隧道会反向连接
cloudflared tunnel --url http://127.0.0.1:8000 --no-autoupdate
```

命令会打印一个 `https://随机名.trycloudflare.com` 地址。打开它，浏览器会要求输入用户名 `demo` 和 `ACCESS_PASSWORD`。

验证：

```bash
curl -I https://随机名.trycloudflare.com/            # 401
curl -I -u demo:你的密码 https://随机名.trycloudflare.com/   # 200
```

- 该地址只在 `cloudflared` 进程运行时有效，重启后 URL 会变。
- 中国大陆网络下 `trycloudflare.com` 可能不稳定；如果打不开，改用 cpolar/natapp（国内，需要注册）或直接上云服务器。
- 不要把 `ACCESS_PASSWORD` 放到公开仓库或截图里；发给对方时通过私聊。

### 方式 B：cpolar / natapp / ngrok

它们原理相同：本地进程把 `127.0.0.1:8000` 反向映射到公网。以 cpolar 为例（需注册并按服务商文档配置 token）：

```bash
cpolar http 8000
```

ngrok：

```bash
ngrok http 8000
```

隧道服务商若有 Basic Auth/Access 功能，建议和应用层的 `ACCESS_PASSWORD` 叠加使用。无论用哪个服务，都建议：

- 使用单独的 DeepSeek Key，并在控制台设置消费上限；
- 演示结束立即停止隧道进程；
- 不要把管理端账号密码告诉查看者；
- 更高要求时加 Cloudflare Access、Nginx `limit_req` 或只允许指定 IP。

## 2. Docker Compose（推荐给客户演示）

```bash
cp .env.example .env       # 填 Key；没有 Key 可保持 LLM_PROVIDER=mock
docker compose up -d --build
docker compose ps
curl http://127.0.0.1:8000/api/health
```

数据持久化在命名卷 `smartdesk-data`，容器内路径 `/app/data`。

`docker-compose.yml` 使用 `env_file` 的 `required: false`，需要 Docker Compose v2.24+；旧版本请先创建 `.env` 或删掉该字段。

构建/运行注意：

- 镜像内固定 `--workers 1`：检索索引是进程内缓存，多 worker 会各建一份且与 SQLite 写入竞争。
- 健康检查通过 `python urllib` 请求 `/api/health`，不依赖 curl。
- 文档上传限制 5 MB；如果前面还有 Nginx，记得 `client_max_body_size` 大于 5m。

## 3. 云主机（以 Ubuntu 22.04 轻量应用服务器为例）

### 3.1 基础环境

```bash
sudo apt update && sudo apt install -y nginx docker.io docker-compose-v2
sudo systemctl enable --now nginx
```

将项目上传到 `/opt/smartdesk-ai`，创建 `.env`：

```bash
cd /opt/smartdesk-ai
cp .env.example .env
chmod 600 .env
# 填 LLM_API_KEY、ADMIN_TOKEN、ITSM_API_KEY
# APP_PORT=8000；容器内自调用 ITSM 默认 http://127.0.0.1:8000 可直接工作
sudo docker compose up -d --build
```

安全组/防火墙只开放 22、80、443；**不要**把 8000 直接暴露到公网。

### 3.2 Nginx 反向代理

仓库提供了 [`deploy/nginx.conf`](../deploy/nginx.conf)。核心是 SSE 不能缓冲：

```nginx
location / {
    proxy_pass http://127.0.0.1:8000;
    proxy_http_version 1.1;
    proxy_set_header Host $host;
    proxy_set_header X-Real-IP $remote_addr;
    proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    proxy_set_header X-Forwarded-Proto $scheme;
    proxy_buffering off;      # 关键：否则 token 会被攒着
    proxy_cache off;
    proxy_read_timeout 300s;
    add_header X-Accel-Buffering no;
}
```

```bash
sudo cp deploy/nginx.conf /etc/nginx/sites-available/smartdesk
sudo ln -s /etc/nginx/sites-available/smartdesk /etc/nginx/sites-enabled/smartdesk
sudo nginx -t && sudo systemctl reload nginx
```

TLS：

```bash
sudo apt install -y certbot python3-certbot-nginx
sudo certbot --nginx -d smartdesk.example.com
```

把 `server_name` 改成你的域名；证书续期由 certbot 定时任务处理。

### 3.3 健康检查与日志

```bash
curl -fsS https://smartdesk.example.com/api/health
docker compose logs -f --tail=200
```

正常应返回 `{"status":"ok","database":true,...}`；`database:false` 时检查卷权限/磁盘。

## 4. 配置与密钥

| 变量 | 说明 | 生产建议 |
|---|---|---|
| `LLM_API_KEY` | DeepSeek Key | 只放 `.env`/密钥管理，权限 600，不提交 |
| `LLM_MODEL` | 当前 `deepseek-v4-flash` | 以控制台为准，可换 `deepseek-v4-pro` |
| `ADMIN_TOKEN` | 文档管理端点令牌 | 生产必须设置，使用随机长串 |
| `ITSM_API_KEY` | 调第三方工单系统的 Key | 与客户系统约定，定期轮换 |
| `DEMO_CUSTOMER_ID` | 演示客户 | 生产必须从登录态取，不能写死 |
| `APP_ENV` | `development`/`production` | 生产设 `production` |
| `DATA_DIR` | SQLite 与数据目录 | 生产挂持久卷，做备份 |
| `EMBEDDING_*` | 远程 embedding | 生产建议 BGE-M3，切换后重建索引 |

不要把 `.env` 打进镜像；`.gitignore` 已忽略。Ansible/Vault/云密钥服务都可以注入同名环境变量。

## 5. 备份与恢复（SQLite）

```bash
# 备份：先停写或使用 sqlite3 .backup，避免直接拷贝 WAL 中的不一致状态
docker compose exec smartdesk python -c "
import sqlite3
src = sqlite3.connect('/app/data/smartdesk.db')
dst = sqlite3.connect('/app/data/backup.db')
src.backup(dst)
dst.close(); src.close()
"
docker compose cp smartdesk:/app/data/backup.db ./smartdesk-backup.db
```

恢复时停止服务，替换卷内 `smartdesk.db`（同时清理 `-wal`/`-shm`），再启动。生产建议每日定时备份并异地保存。

## 6. 升级与回滚

```bash
# 升级
git pull          # 或上传新版本
sudo docker compose build --no-cache smartdesk
sudo docker compose up -d smartdesk

# 回滚
sudo docker compose down
# 切回上一个镜像 tag / 代码版本
sudo docker compose up -d
```

数据库变更目前是 `CREATE TABLE IF NOT EXISTS`；后续如果加字段，应引入版本化迁移（Alembic/自建 `schema_meta` 迁移），不能依赖删库重建。

## 7. 扩缩容注意事项

当前版本适合“1 个客户演示实例”：

- 单进程 + 内存索引；
- SQLite 单文件；
- 进程内 `asyncio.Semaphore` 限制并发流；
- 会话锁在进程内。

扩大后需要：

1. 向量库：pgvector/Qdrant，索引共享；
2. 会话/限流：Redis；
3. 多副本 API + 负载均衡（SSE 需要支持长连接）；
4. 文档入库改异步队列；
5. 审计、PII 脱敏、SSO、租户隔离。

## 8. 故障排查

| 现象 | 排查 |
|---|---|
| 启动报 `LLM_API_KEY 未填写` | `.env` 没填/仍为占位符；或先设 `LLM_PROVIDER=mock` |
| 401/403 | Key 错误、权限不足；看服务端日志，不要在浏览器暴露 Key |
| 404 model not found | 模型名过时；改 `LLM_MODEL`，确认 `LLM_BASE_URL` |
| 回答是“离线演示模式” | 仍是 mock；检查 `.env` 的 `LLM_PROVIDER=deepseek` 并重启 |
| 页面有回答但突然中断 | Nginx 缓冲/超时；确认 `proxy_buffering off`、`proxy_read_timeout` |
| 工单创建失败 | 检查 `ITSM_BASE_URL` 是否能在容器内访问；Key 是否一致；看 4xx/5xx |
| 检索为空 | 文档未导入或 embedding 配置变更；运行 `POST /api/documents/reindex` |
| 上传 413 | `MAX_UPLOAD_MB` 或 Nginx `client_max_body_size` |
| 多 worker 下结果不一致 | 当前版本固定单 worker；多副本需共享索引与锁 |

## 9. 面试可讲的“云计算经验”

- 一次镜像构建 + Compose 编排，环境变量注入，数据卷持久化。
- Nginx 反向代理 + SSE 的特殊配置（`proxy_buffering off`、长超时）。
- 健康检查、日志、备份、回滚、安全组最小暴露面。
- 明确说明：**GitHub Actions 已真实构建并烟测镜像**，但本机未安装 Docker、尚未部署到云主机；到目标环境仍要实际压测，不把“CI 构建通过”说成“线上生产验证”。
