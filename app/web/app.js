(() => {
  "use strict";

  const $ = (selector, root = document) => root.querySelector(selector);
  const $$ = (selector, root = document) => Array.from(root.querySelectorAll(selector));

  const els = {
    sidebar: $("#sidebar"),
    sidebarBackdrop: $("#sidebarBackdrop"),
    menuBtn: $("#menuBtn"),
    newChatBtn: $("#newChatBtn"),
    sessionList: $("#sessionList"),
    sessionCount: $("#sessionCount"),
    docList: $("#docList"),
    uploadBtn: $("#uploadBtn"),
    reindexBtn: $("#reindexBtn"),
    statDocs: $("#statDocs"),
    statChunks: $("#statChunks"),
    modeDot: $("#modeDot"),
    modeText: $("#modeText"),
    modeBadge: $("#modeBadge"),
    customerChip: $("#customerChip"),
    chatScroll: $("#chatScroll"),
    emptyState: $("#emptyState"),
    suggestions: $("#suggestions"),
    messages: $("#messages"),
    composer: $("#composer"),
    messageInput: $("#messageInput"),
    sendBtn: $("#sendBtn"),
    sendIcon: $("#sendIcon"),
    stopIcon: $("#stopIcon"),
    uploadModal: $("#uploadModal"),
    uploadLimit: $("#uploadLimit"),
    dropzone: $("#dropzone"),
    fileInput: $("#fileInput"),
    fileName: $("#fileName"),
    uploadConfirm: $("#uploadConfirm"),
    uploadStatus: $("#uploadStatus"),
    tokenModal: $("#tokenModal"),
    tokenInput: $("#tokenInput"),
    tokenConfirm: $("#tokenConfirm"),
    toast: $("#toast"),
  };

  const state = {
    meta: null,
    sessions: [],
    documents: [],
    stats: null,
    currentSessionId: null,
    liveMessageId: null,
    streaming: false,
    abortController: null,
    adminToken: localStorage.getItem("smartdesk_admin_token") || "",
    pendingFile: null,
    tokenResolver: null,
    userScrolledUp: false,
    toastTimer: null,
    toastHideTimer: null,
  };

  const buffers = new Map();
  const TOOL_LABELS = {
    search_knowledge_base: "检索知识库",
    lookup_order: "查询订单",
    create_ticket: "创建工单",
  };

  // ------------------------------------------------------------------
  // Utilities
  // ------------------------------------------------------------------
  function el(tag, className, text) {
    const node = document.createElement(tag);
    if (className) node.className = className;
    if (text !== undefined) node.textContent = text;
    return node;
  }

  function formatNumber(value) {
    return new Intl.NumberFormat("zh-CN").format(Number(value) || 0);
  }

  function formatRelative(iso) {
    const date = new Date(iso);
    if (Number.isNaN(date.getTime())) return "";
    const seconds = Math.max(0, (Date.now() - date.getTime()) / 1000);
    if (seconds < 60) return "刚刚";
    if (seconds < 3600) return `${Math.floor(seconds / 60)} 分钟前`;
    if (seconds < 86400) return `${Math.floor(seconds / 3600)} 小时前`;
    if (seconds < 604800) return `${Math.floor(seconds / 86400)} 天前`;
    return date.toLocaleDateString("zh-CN", { month: "2-digit", day: "2-digit" });
  }

  function formatTime(iso) {
    const date = iso ? new Date(iso) : new Date();
    if (Number.isNaN(date.getTime())) return "";
    return date.toLocaleTimeString("zh-CN", { hour: "2-digit", minute: "2-digit" });
  }

  function toast(message) {
    els.toast.textContent = message;
    els.toast.classList.remove("hidden");
    requestAnimationFrame(() => els.toast.classList.add("show"));
    if (state.toastTimer) clearTimeout(state.toastTimer);
    if (state.toastHideTimer) clearTimeout(state.toastHideTimer);
    state.toastTimer = setTimeout(() => {
      els.toast.classList.remove("show");
      state.toastHideTimer = setTimeout(
        () => els.toast.classList.add("hidden"),
        220
      );
    }, 3200);
  }

  function isNearBottom() {
    const node = els.chatScroll;
    return node.scrollHeight - node.scrollTop - node.clientHeight < 140;
  }

  function scrollToBottom(force = false) {
    if (!force && state.userScrolledUp) return;
    els.chatScroll.scrollTo({ top: els.chatScroll.scrollHeight, behavior: "auto" });
  }

  async function api(path, options = {}) {
    const response = await fetch(path, options);
    let data = {};
    try {
      data = await response.json();
    } catch {
      data = {};
    }
    if (!response.ok) {
      throw new Error(data.detail || `请求失败（HTTP ${response.status}）`);
    }
    return data;
  }

  function setStatus(node, message, kind) {
    node.textContent = message;
    node.classList.remove("hidden", "error", "success");
    if (kind) node.classList.add(kind);
  }

  // ------------------------------------------------------------------
  // Modals & admin token
  // ------------------------------------------------------------------
  function openModal(id) {
    const modal = document.getElementById(id);
    if (!modal) return;
    modal.classList.remove("hidden");
  }

  function closeModal(id) {
    const modal = document.getElementById(id);
    if (!modal) return;
    modal.classList.add("hidden");
    if (id === "tokenModal") resolveToken(null);
  }

  function requestToken() {
    if (state.tokenResolver) resolveToken(null);
    return new Promise((resolve) => {
      state.tokenResolver = resolve;
      els.tokenInput.value = state.adminToken;
      openModal("tokenModal");
      setTimeout(() => els.tokenInput.focus(), 40);
    });
  }

  function resolveToken(value) {
    const resolver = state.tokenResolver;
    state.tokenResolver = null;
    if (value) {
      state.adminToken = value;
      localStorage.setItem("smartdesk_admin_token", value);
    }
    if (resolver) resolver(value || null);
  }

  async function adminFetch(path, options = {}) {
    const headers = new Headers(options.headers || {});
    if (state.adminToken) headers.set("X-Admin-Token", state.adminToken);
    let response = await fetch(path, { ...options, headers });
    if (response.status === 401) {
      const token = await requestToken();
      if (!token) return null;
      headers.set("X-Admin-Token", token);
      response = await fetch(path, { ...options, headers });
    }
    return response;
  }

  // ------------------------------------------------------------------
  // Meta / stats / sessions / documents
  // ------------------------------------------------------------------
  async function loadMeta() {
    try {
      const meta = await api("/api/meta");
      state.meta = meta;
      els.modeBadge.textContent = meta.mode === "mock" ? "离线演示" : meta.mode_label;
      els.modeDot.classList.remove("live", "mock");
      els.modeDot.classList.add(meta.mode === "mock" ? "mock" : "live");
      els.modeText.textContent = meta.mode === "mock" ? "Mock 规则模型" : meta.mode_label;
      els.customerChip.textContent = `客户 ${meta.demo_customer_id}`;
      els.uploadLimit.textContent = meta.limits?.upload_mb ?? 5;
      els.messageInput.maxLength = meta.limits?.message_chars ?? 2000;
      renderSuggestions(meta.suggestions || []);
    } catch (error) {
      els.modeText.textContent = "服务不可用";
      els.modeBadge.textContent = "离线";
      toast(error.message);
    }
  }

  function renderSuggestions(items) {
    els.suggestions.replaceChildren();
    items.slice(0, 4).forEach((text) => {
      const button = el("button", "suggestion-chip", text);
      button.type = "button";
      button.addEventListener("click", () => {
        els.messageInput.value = text;
        autoGrow();
        els.messageInput.focus();
      });
      els.suggestions.appendChild(button);
    });
  }

  async function loadStats() {
    try {
      const stats = await api("/api/stats");
      state.stats = stats;
      els.statDocs.textContent = `${formatNumber(stats.documents)} 篇文档`;
      const parts = [`${formatNumber(stats.chunks)} 个片段`];
      if (stats.feedback_total > 0) {
        parts.push(`满意度 ${Math.round((stats.feedback_positive_rate || 0) * 100)}%`);
      }
      els.statChunks.textContent = parts.join(" · ");
    } catch {
      els.statDocs.textContent = "统计不可用";
      els.statChunks.textContent = "";
    }
  }

  async function loadSessions() {
    try {
      const data = await api("/api/sessions");
      state.sessions = data.sessions || [];
      renderSessions();
    } catch (error) {
      toast(`加载会话失败：${error.message}`);
    }
  }

  function renderSessions() {
    els.sessionCount.textContent = state.sessions.length ? String(state.sessions.length) : "";
    els.sessionList.replaceChildren();
    if (!state.sessions.length) {
      els.sessionList.appendChild(el("div", "empty-side", "还没有会话，点击「新对话」开始。"));
      return;
    }
    state.sessions.forEach((session) => {
      const row = el("div", "session-item");
      if (session.id === state.currentSessionId) row.classList.add("active");

      const main = el("button", "session-main");
      main.type = "button";
      main.title = session.title;
      main.appendChild(el("span", "session-title", session.title || "新会话"));
      const meta = el(
        "span",
        "session-meta",
        `${session.message_count || 0} 条消息 · ${formatRelative(session.updated_at)}`
      );
      main.appendChild(meta);
      main.addEventListener("click", () => openSession(session.id));
      row.appendChild(main);

      const remove = el("button", "session-delete", "×");
      remove.type = "button";
      remove.title = "删除会话";
      remove.addEventListener("click", (event) => {
        event.stopPropagation();
        deleteSession(session.id);
      });
      row.appendChild(remove);
      els.sessionList.appendChild(row);
    });
  }

  async function openSession(sessionId) {
    if (state.streaming) {
      toast("请先停止当前回答，再切换会话。");
      return;
    }
    try {
      const data = await api(`/api/sessions/${encodeURIComponent(sessionId)}/messages`);
      state.currentSessionId = sessionId;
      state.liveMessageId = null;
      els.messages.replaceChildren();
      buffers.clear();
      const messages = data.messages || [];
      if (!messages.length) {
        els.emptyState.classList.remove("hidden");
      } else {
        els.emptyState.classList.add("hidden");
        messages.forEach(renderHistoryMessage);
      }
      renderSessions();
      closeSidebar();
      requestAnimationFrame(() => scrollToBottom(true));
    } catch (error) {
      toast(`打开会话失败：${error.message}`);
    }
  }

  async function deleteSession(sessionId) {
    if (state.streaming && state.currentSessionId === sessionId) {
      toast("回答生成中，暂时不能删除会话。");
      return;
    }
    if (!window.confirm("确定删除这个会话及其全部消息吗？")) return;
    try {
      await api(`/api/sessions/${encodeURIComponent(sessionId)}`, { method: "DELETE" });
      if (state.currentSessionId === sessionId) newChat();
      await Promise.all([loadSessions(), loadStats()]);
      toast("会话已删除");
    } catch (error) {
      toast(error.message);
    }
  }

  async function loadDocuments() {
    try {
      const data = await api("/api/documents");
      state.documents = data.documents || [];
      renderDocuments();
    } catch (error) {
      toast(`加载知识库失败：${error.message}`);
    }
  }

  function renderDocuments() {
    els.docList.replaceChildren();
    if (!state.documents.length) {
      els.docList.appendChild(el("div", "empty-side", "暂无文档，点击右上角「＋」上传。"));
      return;
    }
    state.documents.forEach((doc) => {
      const row = el("div", "doc-item");
      const extension = (doc.filename.split(".").pop() || "md").toUpperCase();
      row.appendChild(el("span", "doc-icon", extension.slice(0, 3)));
      const info = el("div", "doc-info");
      info.appendChild(el("span", "doc-title", doc.title));
      info.appendChild(
        el("span", "doc-sub", `${doc.chunk_count} 个片段 · ${doc.filename}`)
      );
      row.appendChild(info);
      const remove = el("button", "doc-delete", "×");
      remove.type = "button";
      remove.title = "删除文档";
      remove.addEventListener("click", () => deleteDocument(doc.id, doc.title));
      row.appendChild(remove);
      els.docList.appendChild(row);
    });
  }

  async function deleteDocument(docId, title) {
    if (!window.confirm(`删除文档「${title}」？对应片段会一并删除。`)) return;
    try {
      const response = await adminFetch(`/api/documents/${encodeURIComponent(docId)}`, {
        method: "DELETE",
      });
      if (!response) return;
      const data = await response.json().catch(() => ({}));
      if (!response.ok) throw new Error(data.detail || "删除失败");
      toast("文档已删除");
      await Promise.all([loadDocuments(), loadStats()]);
    } catch (error) {
      toast(error.message);
    }
  }

  async function reindexDocuments() {
    const button = els.reindexBtn;
    button.disabled = true;
    try {
      const response = await adminFetch("/api/documents/reindex", { method: "POST" });
      if (!response) return;
      const data = await response.json().catch(() => ({}));
      if (!response.ok) throw new Error(data.detail || "重建失败");
      const suffix = data.errors?.length ? `，${data.errors.length} 个失败` : "";
      toast(`索引已重建：${data.documents} 篇 / ${data.chunks} 个片段${suffix}`);
      await Promise.all([loadDocuments(), loadStats()]);
    } catch (error) {
      toast(error.message);
    } finally {
      button.disabled = false;
    }
  }

  // ------------------------------------------------------------------
  // Upload
  // ------------------------------------------------------------------
  function openUploadModal() {
    state.pendingFile = null;
    els.fileInput.value = "";
    els.fileName.textContent = "";
    els.uploadConfirm.disabled = true;
    els.uploadStatus.classList.add("hidden");
    els.uploadStatus.textContent = "";
    openModal("uploadModal");
  }

  function pickFile(file) {
    if (!file) return;
    const extension = file.name.slice(file.name.lastIndexOf(".")).toLowerCase();
    if (![".md", ".markdown", ".txt", ".pdf"].includes(extension)) {
      toast("仅支持 .md / .txt / .pdf 文件");
      return;
    }
    const limitMb = state.meta?.limits?.upload_mb || 5;
    if (file.size > limitMb * 1024 * 1024) {
      toast(`文件超过 ${limitMb} MB 限制`);
      return;
    }
    state.pendingFile = file;
    els.fileName.textContent = `${file.name} · ${(file.size / 1024).toFixed(1)} KB`;
    els.uploadConfirm.disabled = false;
    els.uploadStatus.classList.add("hidden");
  }

  async function uploadFile() {
    if (!state.pendingFile) return;
    els.uploadConfirm.disabled = true;
    setStatus(els.uploadStatus, "正在上传、切分并建立索引…");
    try {
      const form = new FormData();
      form.append("file", state.pendingFile);
      const response = await adminFetch("/api/documents", {
        method: "POST",
        body: form,
      });
      if (!response) {
        setStatus(els.uploadStatus, "已取消上传。", "error");
        return;
      }
      const data = await response.json().catch(() => ({}));
      if (!response.ok) throw new Error(data.detail || "上传失败");
      setStatus(
        els.uploadStatus,
        `${data.message}：${data.title}${data.chunks ? `（${data.chunks} 个片段）` : ""}`,
        "success"
      );
      state.pendingFile = null;
      els.fileInput.value = "";
      els.fileName.textContent = "";
      els.uploadConfirm.disabled = true;
      await Promise.all([loadDocuments(), loadStats()]);
    } catch (error) {
      setStatus(els.uploadStatus, error.message, "error");
      els.uploadConfirm.disabled = !state.pendingFile;
    }
  }

  // ------------------------------------------------------------------
  // Markdown rendering (DOM-based, no innerHTML for model output)
  // ------------------------------------------------------------------
  function isBlockStart(line) {
    return (
      /^#{1,6}\s+/.test(line) ||
      /^\s*>\s?/.test(line) ||
      /^\s*[-*+]\s+/.test(line) ||
      /^\s*\d+[.、)]\s+/.test(line) ||
      /^\s*```/.test(line)
    );
  }

  function renderMarkdown(container, text, sources) {
    container.replaceChildren();
    const lines = String(text || "").replace(/\r\n/g, "\n").split("\n");
    let i = 0;
    while (i < lines.length) {
      const line = lines[i];
      if (/^\s*```/.test(line)) {
        const language = line.trim().slice(3).trim();
        const code = [];
        i += 1;
        while (i < lines.length && !/^\s*```/.test(lines[i])) {
          code.push(lines[i]);
          i += 1;
        }
        i += 1;
        const pre = el("pre");
        const codeEl = el("code", "", code.join("\n"));
        if (language) codeEl.dataset.lang = language;
        pre.appendChild(codeEl);
        container.appendChild(pre);
        continue;
      }
      if (/^\s*$/.test(line)) {
        i += 1;
        continue;
      }
      const heading = line.match(/^(#{1,6})\s+(.*)$/);
      if (heading) {
        const tag = heading[1].length <= 2 ? "h2" : "h3";
        const node = el(tag);
        renderInline(node, heading[2], sources);
        container.appendChild(node);
        i += 1;
        continue;
      }
      if (/^\s*>\s?/.test(line)) {
        const quoteLines = [];
        while (i < lines.length && /^\s*>\s?/.test(lines[i])) {
          quoteLines.push(lines[i].replace(/^\s*>\s?/, ""));
          i += 1;
        }
        const quote = el("blockquote");
        quoteLines.forEach((part, index) => {
          if (index) quote.appendChild(el("br"));
          renderInline(quote, part, sources);
        });
        container.appendChild(quote);
        continue;
      }
      if (/^\s*[-*+]\s+/.test(line)) {
        const list = el("ul");
        while (i < lines.length && /^\s*[-*+]\s+/.test(lines[i])) {
          const item = el("li");
          renderInline(item, lines[i].replace(/^\s*[-*+]\s+/, ""), sources);
          list.appendChild(item);
          i += 1;
        }
        container.appendChild(list);
        continue;
      }
      if (/^\s*\d+[.、)]\s+/.test(line)) {
        const list = el("ol");
        while (i < lines.length && /^\s*\d+[.、)]\s+/.test(lines[i])) {
          const item = el("li");
          renderInline(item, lines[i].replace(/^\s*\d+[.、)]\s+/, ""), sources);
          list.appendChild(item);
          i += 1;
        }
        container.appendChild(list);
        continue;
      }
      const paragraphLines = [line];
      i += 1;
      while (
        i < lines.length &&
        !/^\s*$/.test(lines[i]) &&
        !isBlockStart(lines[i])
      ) {
        paragraphLines.push(lines[i]);
        i += 1;
      }
      const paragraph = el("p");
      paragraphLines.forEach((part, index) => {
        if (index) paragraph.appendChild(el("br"));
        renderInline(paragraph, part, sources);
      });
      container.appendChild(paragraph);
    }
  }

  function renderInline(parent, text, sources) {
    const pattern = /(\*\*[^*]+\*\*|`[^`]+`|\[S\d+\]|\[[^\]]+\]\(https?:\/\/[^)\s]+\))/g;
    let lastIndex = 0;
    for (const match of text.matchAll(pattern)) {
      if (match.index > lastIndex) {
        parent.appendChild(document.createTextNode(text.slice(lastIndex, match.index)));
      }
      const token = match[0];
      if (token.startsWith("**")) {
        parent.appendChild(el("strong", "", token.slice(2, -2)));
      } else if (token.startsWith("`")) {
        parent.appendChild(el("code", "", token.slice(1, -1)));
      } else if (/^\[S\d+\]$/.test(token)) {
        const sourceId = token.slice(1, -1);
        const button = el("button", "cite", sourceId);
        button.type = "button";
        button.dataset.sourceId = sourceId;
        if (!sources.has(sourceId)) {
          button.classList.add("invalid");
          button.title = "该引用没有对应的检索来源";
        } else {
          button.title = `查看来源 ${sourceId}`;
        }
        parent.appendChild(button);
      } else {
        const linkMatch = token.match(/^\[([^\]]+)\]\((https?:\/\/[^)\s]+)\)$/);
        if (linkMatch) {
          const anchor = el("a", "", linkMatch[1]);
          anchor.href = linkMatch[2];
          anchor.target = "_blank";
          anchor.rel = "noopener noreferrer";
          parent.appendChild(anchor);
        } else {
          parent.appendChild(document.createTextNode(token));
        }
      }
      lastIndex = match.index + token.length;
    }
    if (lastIndex < text.length) {
      parent.appendChild(document.createTextNode(text.slice(lastIndex)));
    }
  }

  // ------------------------------------------------------------------
  // Message rendering
  // ------------------------------------------------------------------
  function hideEmptyState() {
    els.emptyState.classList.add("hidden");
  }

  function makeMetaLine(parts) {
    const node = el("div", "message-meta");
    parts.filter(Boolean).forEach((part) => node.appendChild(el("span", "", part)));
    return node;
  }

  function appendUserMessage(text, createdAt) {
    hideEmptyState();
    const message = el("article", "msg user");
    message.appendChild(el("div", "avatar user", "你"));
    const wrap = el("div", "bubble-wrap");
    const bubble = el("div", "bubble");
    const body = el("div", "md");
    body.textContent = text;
    bubble.appendChild(body);
    wrap.appendChild(bubble);
    wrap.appendChild(makeMetaLine([formatTime(createdAt)]));
    message.appendChild(wrap);
    els.messages.appendChild(message);
    scrollToBottom();
  }

  function createAssistantMessage(messageId) {
    if (buffers.has(messageId)) return buffers.get(messageId);
    hideEmptyState();
    const message = el("article", "msg assistant");
    message.dataset.messageId = messageId;
    message.appendChild(el("div", "avatar assistant", "智"));
    const wrap = el("div", "bubble-wrap");
    const bubble = el("div", "bubble");
    const body = el("div", "md streaming");
    const errorSlot = el("div", "error-slot");
    const sourcesSlot = el("div");
    const traceSlot = el("div");
    const feedbackRow = el("div", "feedback-row hidden");
    const metaEl = el("div", "message-meta");
    metaEl.textContent = "正在生成…";
    bubble.append(body, errorSlot, sourcesSlot, traceSlot, feedbackRow);
    wrap.append(bubble, metaEl);
    message.appendChild(wrap);
    els.messages.appendChild(message);

    const buffer = {
      id: messageId,
      el: message,
      bubble,
      body,
      errorSlot,
      sourcesSlot,
      traceSlot,
      feedbackRow,
      metaEl,
      text: "",
      sources: new Map(),
      traceItems: [],
      traceOpen: false,
      finalized: false,
      localError: false,
      renderQueued: false,
      feedback: null,
    };
    buffers.set(messageId, buffer);

    ["1", "-1"].forEach((rating) => {
      const button = el(
        "button",
        "feedback-btn",
        rating === "1" ? "👍 有帮助" : "👎 没帮助"
      );
      button.type = "button";
      button.dataset.rating = rating;
      button.addEventListener("click", () => sendFeedback(buffer, Number(rating)));
      feedbackRow.appendChild(button);
    });
    feedbackRow.appendChild(el("span", "feedback-note", "反馈会用于改进检索与回答"));
    scrollToBottom();
    return buffer;
  }

  function scheduleRender(buffer) {
    if (buffer.renderQueued) return;
    buffer.renderQueued = true;
    requestAnimationFrame(() => {
      buffer.renderQueued = false;
      renderMarkdown(buffer.body, buffer.text, buffer.sources);
      scrollToBottom();
    });
  }

  function renderNow(buffer) {
    buffer.renderQueued = false;
    renderMarkdown(buffer.body, buffer.text, buffer.sources);
  }

  function mergeSources(buffer, list) {
    let changed = false;
    (list || []).forEach((source) => {
      if (!source || !source.id) return;
      if (!buffer.sources.has(source.id)) {
        buffer.sources.set(source.id, source);
        changed = true;
      } else {
        const existing = buffer.sources.get(source.id);
        if (existing.chunk_id !== source.chunk_id) {
          buffer.sources.set(source.id, source);
          changed = true;
        }
      }
    });
    if (changed) {
      renderSources(buffer);
      renderNow(buffer);
    }
  }

  function sortedSources(buffer) {
    return Array.from(buffer.sources.values()).sort((a, b) => {
      return Number(a.id.slice(1)) - Number(b.id.slice(1));
    });
  }

  function renderSources(buffer) {
    buffer.sourcesSlot.replaceChildren();
    const sources = sortedSources(buffer);
    if (!sources.length) return;
    const block = el("details", "sources-block");
    block.appendChild(el("summary", "sources-head", `参考来源 · ${sources.length}`));
    const list = el("div", "source-list");
    sources.forEach((source) => {
      const card = el("div", "source-card");
      card.dataset.sourceCard = source.id;
      const top = el("div", "source-top");
      top.appendChild(el("span", "cite", source.id));
      top.appendChild(el("span", "source-title", source.title || "未命名文档"));
      const section = (source.section || "")
        .split(" > ")
        .filter((part) => part && part !== source.title)
        .join(" · ");
      if (section) {
        top.appendChild(el("span", "source-section", section));
      }
      card.appendChild(top);
      const snippet = el("div", "source-snippet", source.snippet || "");
      snippet.title = source.snippet || "";
      card.appendChild(snippet);
      list.appendChild(card);
    });
    block.appendChild(list);
    buffer.sourcesSlot.appendChild(block);
  }

  function handleTrace(buffer, data) {
    if (!data || !data.tool) return;
    if (data.phase === "start") {
      buffer.traceItems.push({
        tool: data.tool,
        step: data.step,
        callId: data.call_id || "",
        args: data.args || {},
        status: "running",
        summary: "",
        duration: null,
      });
    } else {
      let item = null;
      if (data.call_id) {
        item = buffer.traceItems.find(
          (candidate) => candidate.callId === data.call_id
        ) || null;
      }
      if (!item) {
        for (let i = buffer.traceItems.length - 1; i >= 0; i -= 1) {
          const candidate = buffer.traceItems[i];
          if (
            candidate.status === "running" &&
            candidate.tool === data.tool &&
            candidate.step === data.step
          ) {
            item = candidate;
            break;
          }
        }
      }
      if (!item) {
        item = {
          tool: data.tool,
          step: data.step,
          callId: data.call_id || "",
          args: data.args || {},
          status: "running",
          summary: "",
          duration: null,
        };
        buffer.traceItems.push(item);
      }
      item.status = data.ok === false ? "error" : "done";
      item.summary = data.summary || "";
      item.duration = data.duration_ms;
    }
    renderTrace(buffer);
  }

  function renderTrace(buffer) {
    buffer.traceSlot.replaceChildren();
    if (!buffer.traceItems.length) return;
    const running = buffer.traceItems.filter((item) => item.status === "running");
    const totalMs = buffer.traceItems.reduce(
      (sum, item) => sum + (Number(item.duration) || 0),
      0
    );
    const details = el("details", "trace");
    details.open = buffer.traceOpen;
    details.addEventListener("toggle", () => {
      buffer.traceOpen = details.open;
    });
    const summary = el("summary");
    const dot = el("span", `trace-dot ${running.length ? "running" : ""}`);
    summary.appendChild(dot);
    let label;
    if (running.length) {
      label = `正在调用${TOOL_LABELS[running[0].tool] || running[0].tool}…`;
    } else {
      label = `工具调用 · ${buffer.traceItems.length} 次`;
      if (totalMs > 0) label += ` · ${Math.round(totalMs)} ms`;
    }
    summary.appendChild(el("span", "", label));
    details.appendChild(summary);

    const list = el("div", "trace-list");
    buffer.traceItems.forEach((item, index) => {
      const row = el("div", `trace-item ${item.status === "error" ? "error" : ""} ${item.status === "running" ? "running" : ""}`);
      row.appendChild(el("span", "trace-index", String(index + 1)));
      const body = el("div", "trace-body");
      const line = el("div", "trace-line");
      line.appendChild(
        el("span", "trace-name", TOOL_LABELS[item.tool] || item.tool)
      );
      const argsText = JSON.stringify(item.args || {});
      line.appendChild(
        el("span", "trace-args", argsText.length > 110 ? `${argsText.slice(0, 110)}…` : argsText)
      );
      if (item.duration !== null && item.duration !== undefined) {
        line.appendChild(el("span", "trace-duration", `${Math.round(item.duration)} ms`));
      }
      body.appendChild(line);
      if (item.summary) {
        body.appendChild(el("div", "trace-summary", item.summary));
      } else if (item.status === "running") {
        body.appendChild(el("div", "trace-summary", "执行中…"));
      }
      row.appendChild(body);
      list.appendChild(row);
    });
    details.appendChild(list);
    buffer.traceSlot.appendChild(details);
  }

  function normalizeTrace(items) {
    return (items || []).map((item) => ({
      tool: item.tool,
      step: item.step,
      callId: item.call_id || "",
      args: item.args || {},
      status: item.ok === false ? "error" : "done",
      summary: item.summary || "",
      duration: item.duration_ms ?? null,
    }));
  }

  function renderMeta(buffer, meta) {
    buffer.metaEl.replaceChildren();
    if (!meta) return;
    const parts = [];
    if (meta.latency_ms) {
      const ms = Number(meta.latency_ms);
      parts.push(ms < 1000 ? `耗时 ${Math.round(ms)} ms` : `耗时 ${(ms / 1000).toFixed(1)}s`);
    }
    if (meta.usage?.total_tokens) {
      parts.push(`${formatNumber(meta.usage.total_tokens)} tokens`);
    }
    if (meta.model) parts.push(meta.model);
    if (meta.finish_reason === "max_steps") parts.push("达到步骤上限");
    if (meta.interrupted) parts.push("已中断");
    if (!parts.length) return;
    parts.forEach((part) => buffer.metaEl.appendChild(el("span", "", part)));
  }

  function setFeedbackActive(buffer, rating) {
    buffer.feedback = rating;
    $$(".feedback-btn", buffer.feedbackRow).forEach((button) => {
      button.classList.toggle("active", Number(button.dataset.rating) === rating);
    });
  }

  async function sendFeedback(buffer, rating) {
    if (!buffer.finalized || buffer.localError) return;
    try {
      await api("/api/feedback", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message_id: buffer.id, rating }),
      });
      setFeedbackActive(buffer, rating);
      toast("感谢反馈，这会用于改进检索与回答。");
      loadStats();
    } catch (error) {
      toast(error.message);
    }
  }

  function finalizeAssistant(buffer, { done = null, error = null, interrupted = false } = {}) {
    if (!buffer || buffer.finalized) return;
    buffer.finalized = true;
    buffer.body.classList.remove("streaming");
    buffer.metaEl.replaceChildren();
    if (error) {
      buffer.error = error;
      if (!buffer.text.trim()) {
        buffer.text = "抱歉，本次回答生成失败，请稍后重试。";
      }
      const note = el("div", "error-note", error);
      buffer.errorSlot.appendChild(note);
      buffer.feedbackRow.classList.add("hidden");
    } else if (interrupted) {
      buffer.errorSlot.appendChild(el("div", "error-note", "已停止生成。"));
    } else {
      buffer.feedbackRow.classList.remove("hidden");
    }
    renderNow(buffer);
    if (done) {
      renderMeta(buffer, {
        latency_ms: done.latency_ms,
        usage: done.usage,
        finish_reason: done.finish_reason,
        interrupted: done.interrupted,
        model: state.stats?.model,
      });
      if (done.invalid_citations?.length) {
        buffer.errorSlot.appendChild(
          el(
            "div",
            "error-note",
            `有 ${done.invalid_citations.length} 个引用没有对应的检索来源，已标记。`
          )
        );
      }
    } else if (buffer.meta && !error) {
      renderMeta(buffer, buffer.meta);
    }
    scrollToBottom();
  }

  function renderHistoryMessage(message) {
    if (message.role === "user") {
      appendUserMessage(message.content, message.created_at);
      return;
    }
    if (message.role !== "assistant") return;
    const buffer = createAssistantMessage(message.id);
    buffer.text = message.content || "";
    const meta = message.meta || {};
    buffer.meta = meta;
    mergeSources(buffer, meta.sources || []);
    buffer.traceItems = normalizeTrace(meta.trace || []);
    if (buffer.traceItems.length) renderTrace(buffer);
    if (meta.status === "streaming" && !buffer.text) {
      buffer.text = "（该回答未生成完成）";
    }
    if (meta.src && false) {
      // reserved for future metadata versions
    }
    finalizeAssistant(buffer, {
      error: meta.error ? "本次回答生成失败，请重试。" : null,
      interrupted: Boolean(meta.interrupted),
    });
    if (!meta.error) {
      renderMeta(buffer, meta);
    }
    if (message.feedback?.rating) {
      setFeedbackActive(buffer, message.feedback.rating);
    }
  }

  // ------------------------------------------------------------------
  // SSE chat
  // ------------------------------------------------------------------
  async function consumeSSE(response, handlers) {
    if (!response.body) throw new Error("浏览器不支持流式响应");
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    const feed = (frame) => {
      if (!frame.trim()) return;
      let event = "message";
      const dataLines = [];
      frame.split("\n").forEach((line) => {
        if (line.startsWith(":")) return;
        if (line.startsWith("event:")) {
          event = line.slice(6).trim();
        } else if (line.startsWith("data:")) {
          dataLines.push(line.slice(5).replace(/^\s/, ""));
        }
      });
      if (!dataLines.length) return;
      let data;
      try {
        data = JSON.parse(dataLines.join("\n"));
      } catch {
        return;
      }
      const handler = handlers[event];
      if (handler) handler(data);
    };
    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      buffer = buffer.replace(/\r\n/g, "\n");
      let boundary = buffer.indexOf("\n\n");
      while (boundary !== -1) {
        const frame = buffer.slice(0, boundary);
        buffer = buffer.slice(boundary + 2);
        feed(frame);
        boundary = buffer.indexOf("\n\n");
      }
    }
    buffer += decoder.decode();
    if (buffer.trim()) feed(buffer.replace(/\r\n/g, "\n"));
  }

  function getLiveBuffer() {
    if (state.liveMessageId && buffers.has(state.liveMessageId)) {
      return buffers.get(state.liveMessageId);
    }
    return null;
  }

  function stopStreaming() {
    if (state.abortController) state.abortController.abort();
  }

  function setStreaming(value) {
    state.streaming = value;
    els.sendBtn.classList.toggle("stopping", value);
    els.sendIcon.classList.toggle("hidden", value);
    els.stopIcon.classList.toggle("hidden", !value);
    els.sendBtn.setAttribute("aria-label", value ? "停止生成" : "发送");
  }

  async function sendMessage(rawText) {
    if (state.streaming) return;
    const text = String(rawText || "").trim();
    if (!text) return;
    const limit = state.meta?.limits?.message_chars || 2000;
    if (text.length > limit) {
      toast(`消息最多 ${limit} 个字符`);
      return;
    }
    hideEmptyState();
    state.userScrolledUp = false;
    appendUserMessage(text);
    els.messageInput.value = "";
    autoGrow();
    setStreaming(true);
    state.abortController = new AbortController();
    let sawTerminal = false;
    let liveBuffer = null;

    const handlers = {
      session: (data) => {
        state.currentSessionId = data.session_id;
        state.liveMessageId = data.message_id;
        liveBuffer = createAssistantMessage(data.message_id);
        loadSessions();
        renderSessions();
      },
      token: (data) => {
        const buffer = liveBuffer || getLiveBuffer();
        if (!buffer || !data.delta) return;
        buffer.text += data.delta;
        scheduleRender(buffer);
      },
      trace: (data) => {
        const buffer = liveBuffer || getLiveBuffer();
        if (buffer) handleTrace(buffer, data);
      },
      sources: (data) => {
        const buffer = liveBuffer || getLiveBuffer();
        if (buffer) mergeSources(buffer, data.sources);
      },
      done: (data) => {
        sawTerminal = true;
        const buffer = liveBuffer || getLiveBuffer();
        if (buffer) finalizeAssistant(buffer, { done: data });
        state.liveMessageId = null;
      },
      error: (data) => {
        sawTerminal = true;
        const buffer = liveBuffer || getLiveBuffer();
        if (buffer) {
          finalizeAssistant(buffer, { error: data.message || "生成失败" });
        } else {
          showLocalError(data.message || "生成失败");
        }
        state.liveMessageId = null;
      },
    };

    try {
      const body = { message: text };
      if (state.currentSessionId) body.session_id = state.currentSessionId;
      const response = await fetch("/api/chat/stream", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
        signal: state.abortController.signal,
      });
      if (!response.ok) {
        let detail = `请求失败（HTTP ${response.status}）`;
        try {
          const data = await response.json();
          detail = data.detail || detail;
        } catch {
          /* ignore */
        }
        showLocalError(detail);
        return;
      }
      await consumeSSE(response, handlers);
      if (!sawTerminal) {
        const buffer = liveBuffer || getLiveBuffer();
        if (buffer && !buffer.finalized) {
          finalizeAssistant(buffer, { error: "连接意外中断，请重试。" });
        }
      }
    } catch (error) {
      if (error.name === "AbortError") {
        const buffer = liveBuffer || getLiveBuffer();
        if (buffer && !buffer.finalized) {
          finalizeAssistant(buffer, { interrupted: true });
        }
      } else {
        const buffer = liveBuffer || getLiveBuffer();
        if (buffer && !buffer.finalized) {
          finalizeAssistant(buffer, { error: `网络异常：${error.message}` });
        } else {
          showLocalError(`网络异常：${error.message}`);
        }
      }
    } finally {
      setStreaming(false);
      state.abortController = null;
      state.liveMessageId = null;
      loadStats();
    }
  }

  function showLocalError(message) {
    const buffer = createAssistantMessage(`local-${Date.now()}-${Math.random().toString(16).slice(2)}`);
    buffer.localError = true;
    buffer.text = "";
    finalizeAssistant(buffer, { error: message });
    buffer.feedbackRow.classList.add("hidden");
  }

  // ------------------------------------------------------------------
  // Layout, input & events
  // ------------------------------------------------------------------
  function autoGrow() {
    els.messageInput.style.height = "auto";
    els.messageInput.style.height = `${Math.min(els.messageInput.scrollHeight, 160)}px`;
  }

  function openSidebar() {
    els.sidebar.classList.add("open");
    els.sidebarBackdrop.classList.add("show");
  }

  function closeSidebar() {
    els.sidebar.classList.remove("open");
    els.sidebarBackdrop.classList.remove("show");
  }

  function newChat() {
    if (state.streaming) {
      toast("请先停止当前回答。");
      return;
    }
    state.currentSessionId = null;
    state.liveMessageId = null;
    els.messages.replaceChildren();
    buffers.clear();
    els.emptyState.classList.remove("hidden");
    renderSessions();
    closeSidebar();
    els.messageInput.focus();
  }

  function bindEvents() {
    els.composer.addEventListener("submit", (event) => {
      event.preventDefault();
      if (state.streaming) {
        stopStreaming();
        return;
      }
      sendMessage(els.messageInput.value);
    });

    els.messageInput.addEventListener("input", autoGrow);
    els.messageInput.addEventListener("keydown", (event) => {
      if (
        event.key === "Enter" &&
        !event.shiftKey &&
        !event.isComposing &&
        event.keyCode !== 229
      ) {
        event.preventDefault();
        els.composer.requestSubmit();
      }
    });

    els.chatScroll.addEventListener("scroll", () => {
      state.userScrolledUp = !isNearBottom();
    });

    els.newChatBtn.addEventListener("click", newChat);
    els.menuBtn.addEventListener("click", openSidebar);
    els.sidebarBackdrop.addEventListener("click", closeSidebar);

    els.uploadBtn.addEventListener("click", openUploadModal);
    els.reindexBtn.addEventListener("click", reindexDocuments);
    els.tokenConfirm.addEventListener("click", () => {
      const value = els.tokenInput.value.trim();
      if (!value) {
        toast("请输入管理员令牌");
        return;
      }
      const resolver = state.tokenResolver;
      state.tokenResolver = null;
      state.adminToken = value;
      localStorage.setItem("smartdesk_admin_token", value);
      closeModal("tokenModal");
      if (resolver) resolver(value);
    });

    $$("[data-close]").forEach((button) => {
      button.addEventListener("click", () => closeModal(button.dataset.close));
    });
    $$(".modal-backdrop").forEach((backdrop) => {
      backdrop.addEventListener("click", (event) => {
        if (event.target === backdrop) closeModal(backdrop.id);
      });
    });

    els.dropzone.addEventListener("click", () => els.fileInput.click());
    els.dropzone.addEventListener("keydown", (event) => {
      if (event.key === "Enter" || event.key === " ") {
        event.preventDefault();
        els.fileInput.click();
      }
    });
    els.fileInput.addEventListener("change", () => pickFile(els.fileInput.files[0]));
    els.dropzone.addEventListener("dragover", (event) => {
      event.preventDefault();
      els.dropzone.classList.add("dragover");
    });
    els.dropzone.addEventListener("dragleave", () => {
      els.dropzone.classList.remove("dragover");
    });
    els.dropzone.addEventListener("drop", (event) => {
      event.preventDefault();
      els.dropzone.classList.remove("dragover");
      pickFile(event.dataTransfer.files[0]);
    });
    els.uploadConfirm.addEventListener("click", uploadFile);

    els.messages.addEventListener("click", (event) => {
      const cite = event.target.closest(".cite");
      if (cite && cite.dataset.sourceId) {
        const message = cite.closest(".msg");
        const details = message?.querySelector(".sources-block");
        if (details) details.open = true;
        const card = message?.querySelector(
          `[data-source-card="${cite.dataset.sourceId}"]`
        );
        if (card) {
          card.classList.add("highlight");
          card.scrollIntoView({ behavior: "smooth", block: "center" });
          setTimeout(() => card.classList.remove("highlight"), 1800);
        } else {
          toast("该引用没有对应的检索来源。");
        }
      }
    });

    document.addEventListener("keydown", (event) => {
      if (event.key !== "Escape") return;
      ["uploadModal", "tokenModal"].forEach((id) => {
        if (!document.getElementById(id).classList.contains("hidden")) {
          closeModal(id);
        }
      });
      closeSidebar();
    });
  }

  async function init() {
    bindEvents();
    await Promise.allSettled([loadMeta(), loadSessions(), loadDocuments(), loadStats()]);
    els.messageInput.focus();
  }

  init();
})();
