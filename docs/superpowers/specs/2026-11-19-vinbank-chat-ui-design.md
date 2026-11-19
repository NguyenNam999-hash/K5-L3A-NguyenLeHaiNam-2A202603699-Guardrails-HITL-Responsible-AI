# VinBank Chat UI — Design

Date: 2026-11-19
Status: approved approach (A), design pending approval

## Intent

Give VinBank customers a browser chat against the **Blue** agent, where each
reply also shows the guardrail verdict (ALLOW/BLOCK, layer, reason). The UI is
for demo and real use of the lab's defense stack; it is additive and must not
change any graded artifact.

## Goals

- Chat in the browser against the real Blue agent and real plugins.
- Show per-turn verdict: status, layer, reason, one expandable detail row.
- Run with zero new dependencies: `python src/webui/server.py`.
- Keep `OPENROUTER_API_KEY` server-side only.
- No writes to `outputs/`; no change to graded code paths.

## Non-goals

- Accounts, login, persistence, database.
- Streaming tokens.
- Chatting with Red / Red Advance.
- Deployment, public hosting, TLS.
- Editing guardrail logic or `data/protected/vinbank_secrets.json`.

## Constraints

- Blue model/provider are hard-locked in `core/config.py`; reuse
  `create_blue_agent` and `build_production_plugins`.
- Guardrail plugins return replacement text only; they do not report which
  layer fired. Verdict detail must come from a tracing wrapper, not from
  rewriting graded files.
- `pytest tests/smoke` must keep passing with no API key.
- Commands run from repo root; imports are `from core...` with `src/` on
  `sys.path`.

## Approaches

- **A. stdlib `http.server` + vanilla HTML/JS (chosen).** No new packages, full
  control of the verdict panel, reuses the real plugin path.
- B. Gradio/Streamlit. Rejected: heavy dependency, boxed styling, weaker
  per-message trace control.
- C. Browser-only OpenRouter call. Rejected: exposes the API key and skips the
  Python guardrails entirely.

## Design

### Files

```
src/webui/__init__.py
src/webui/server.py       # ThreadingHTTPServer, routes, session store
src/webui/tracing.py      # verdict-recording plugin wrappers
src/webui/static/index.html  # layout, inline CSS, inline JS
tests/smoke/test_webui_trace.py  # wrapper logic, no API key, no network
```

### Request flow

1. `GET /` → `index.html`.
2. `POST /api/chat` with `{"session_id": "...", "message": "..."}`.
3. Server appends user turn to session history (max 20 turns).
4. `create_blue_agent(plugins)` with wrapped plugins:
   - `_TraceRateLimit` → real `RateLimitPlugin`
   - `_TraceInput` → real `InputGuardrailPlugin`
   - `_TraceOutput` → real `OutputGuardrailPlugin`
5. `chat_with_agent(agent, runner, message)` runs the real path. The runner's
   optional history argument is used for multi-turn context (see below).
6. Response: `{"reply", "verdict": {"status", "layers": [...]}, "usage": {...}}`.

### Verdict shape

```json
{
  "status": "BLOCK",
  "layers": [
    {"name": "rate_limit", "verdict": "ALLOW"},
    {"name": "input", "verdict": "BLOCK", "reason": "injection: ignore previous instructions"},
    {"name": "output", "verdict": "SKIP"}
  ]
}
```

- Rate limit reason: wait seconds from the plugin's block text.
- Input reason: run `detect_injection` and `topic_filter` on the same text,
  then delegate to the real plugin. The real plugin's return value is the
  source of truth for ALLOW/BLOCK; the pure functions only name the reason.
- Output reason: `content_filter(text)["issues"]`; after delegation, report
  `redacted` when the plugin rewrote the text.
- Model/network failure: `{"name": "model", "verdict": "ERROR", "reason": ...}`,
  HTTP 502.

### Multi-turn context

`OpenAIRunner.chat` currently sends only the current user message. Add an
optional keyword `history: list[dict] | None = None` (default keeps today's
behavior) and pass prior sessions turns as `messages` before the new one. This
is additive; no existing caller changes. Rejected alternative: bind history in
the web layer, which would duplicate the model call and drift from the runner.

### UI

Single page, Vietnamese labels (target users are VinBank customers), no build
step, no CDN. Blue brand accents, light/dark by `prefers-color-scheme`.
Layout:

- Header: VinBank Assistant, model badge, "new chat" button.
- Message list: user right, assistant left.
- Verdict row under every assistant reply: green `ALLOW`, red `BLOCK`, amber
  `REDACTED`/`ERROR`; click toggles per-layer details and reason.
- Composer: textarea, Enter sends, Shift+Enter newline, send disabled while
  pending.
- Errors shown as an inline system bubble, never a silent failure.
- Rate-limit and block replies come from the real plugins, so they read exactly
  as the lab produces them.

### Server behavior

- Bind `127.0.0.1:8000` by default; `--host`/`--port` flags to override.
- `ThreadingHTTPServer`, request body capped at 8 KB, message capped at 4000
  chars, JSON-only, no CORS headers.
- Session store: in-memory dict, 20-turn cap, idle expiry 1 hour.
- Missing `OPENROUTER_API_KEY`: server starts, `/api/chat` returns 503 with a
  clear message; UI shows it.
- No logging of message text unless `--debug`; nothing written to disk.

## Left to the builder

- Exact CSS palette and typography.
- Whether the verdict row starts collapsed (current call: collapsed, one-line
  summary visible).
- Test depth for the tracing wrapper.
