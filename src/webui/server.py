"""VinBank chat server — stdlib only, guardrailed Blue agent.

Run from the repo root:

    python src/webui/server.py
"""
from __future__ import annotations

import argparse
import asyncio
import json
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Callable

_STATIC_DIR = Path(__file__).resolve().parent / "static"
_MAX_BODY = 8192
_MAX_MESSAGE = 4000
_SESSION_TTL = 3600.0


class MissingApiKey(RuntimeError):
    """Raised when the Blue OpenRouter key is not configured."""


def _default_factory(max_requests: int, window_seconds: int) -> Callable:
    def factory(recorder):
        from core.config import get_openrouter_api_key

        if not get_openrouter_api_key():
            raise MissingApiKey(
                "OPENROUTER_API_KEY is not set. Add it to .env and restart."
            )

        from assignment.pipeline import build_production_plugins
        from agents.agent import create_blue_agent
        from webui.tracing import TracedInput, TracedOutput, TracedRateLimit

        rate, input_plugin, output_plugin = build_production_plugins(
            max_requests=max_requests, window_seconds=window_seconds
        )
        plugins = [
            TracedRateLimit(rate, recorder),
            TracedInput(input_plugin, recorder),
            TracedOutput(output_plugin, recorder),
        ]
        return create_blue_agent(plugins)

    return factory


class ChatApp:
    def __init__(
        self,
        *,
        pair_factory: Callable | None = None,
        max_turns: int = 20,
        max_requests: int = 10,
        window_seconds: int = 60,
    ) -> None:
        self.pair_factory = pair_factory or _default_factory(
            max_requests, window_seconds
        )
        self.max_turns = max_turns
        self._pair = None
        self._lock = threading.Lock()
        self._sessions: dict[str, tuple[list[dict], float]] = {}

    def _build_pair(self, recorder):
        if self._pair is None:
            self._pair = self.pair_factory(recorder)
        return self._pair

    def _prune_sessions(self, now: float) -> None:
        expired = [
            key
            for key, (_, seen) in self._sessions.items()
            if now - seen > _SESSION_TTL
        ]
        for key in expired:
            del self._sessions[key]

    def _history_for(self, session_id: str, now: float) -> list[dict]:
        entry = self._sessions.get(session_id)
        return list(entry[0]) if entry else []

    def _save_history(self, session_id: str, history: list[dict], now: float) -> None:
        limit = self.max_turns * 2
        if len(history) > limit:
            history = history[-limit:]
        self._sessions[session_id] = (history, now)

    @staticmethod
    def _error_payload(recorder, message: str, status: int = 502) -> tuple[dict, int]:
        if not any(layer["name"] == "output" for layer in recorder.layers):
            recorder.record("output", "SKIP")
        recorder.record("model", "ERROR", message)
        return {"error": message, "verdict": recorder.verdict()}, status

    def handle_chat(self, session_id: str, message: str) -> tuple[dict, int]:
        from webui.tracing import VerdictRecorder

        if not isinstance(session_id, str) or not session_id:
            session_id = uuid.uuid4().hex
        if not isinstance(message, str):
            return {"error": "message is required"}, 400
        message = message.strip()
        if not message:
            return {"error": "message is required"}, 400
        if len(message) > _MAX_MESSAGE:
            return {"error": f"message too long (max {_MAX_MESSAGE} chars)"}, 400

        with self._lock:
            recorder = VerdictRecorder()
            recorder.reset()
            now = time.monotonic()
            self._prune_sessions(now)
            history = self._history_for(session_id, now)

            try:
                agent, runner = self._build_pair(recorder)
            except MissingApiKey as exc:
                return {"error": str(exc)}, 503
            except Exception as exc:
                return self._error_payload(recorder, f"model setup failed: {exc}")

            try:
                reply = asyncio.run(runner.chat(agent, message, history=history))
            except Exception as exc:
                return self._error_payload(recorder, f"model call failed: {exc}")

            if not any(layer["name"] == "output" for layer in recorder.layers):
                recorder.record("output", "SKIP")

            history.extend(
                [
                    {"role": "user", "content": message},
                    {"role": "assistant", "content": reply},
                ]
            )
            self._save_history(session_id, history, now)
            return {"reply": reply, "verdict": recorder.verdict()}, 200


def build_handler(app: ChatApp, debug: bool = False):
    class Handler(BaseHTTPRequestHandler):
        def _send(self, status: int, body: bytes, content_type: str) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _send_json(self, status: int, payload: dict) -> None:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self._send(status, body, "application/json; charset=utf-8")

        def do_GET(self) -> None:  # noqa: N802
            if self.path.split("?", 1)[0] not in ("/", "/index.html"):
                self._send_json(404, {"error": "not found"})
                return
            path = _STATIC_DIR / "index.html"
            try:
                body = path.read_bytes()
            except OSError:
                self._send_json(500, {"error": "chat page missing"})
                return
            self._send(200, body, "text/html; charset=utf-8")

        def do_POST(self) -> None:  # noqa: N802
            if self.path.split("?", 1)[0] != "/api/chat":
                self._send_json(404, {"error": "not found"})
                return
            try:
                length = int(self.headers.get("Content-Length") or "0")
            except ValueError:
                length = 0
            if length > _MAX_BODY:
                self._send_json(413, {"error": "request too large"})
                return
            raw = self.rfile.read(length) if length > 0 else b""
            try:
                data = json.loads(raw.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                self._send_json(400, {"error": "invalid JSON"})
                return
            if not isinstance(data, dict):
                self._send_json(400, {"error": "invalid JSON"})
                return

            payload, status = app.handle_chat(
                data.get("session_id", ""), data.get("message", "")
            )
            if debug:
                print(f"[chat] {status} {data.get('session_id', '')}")
            self._send_json(status, payload)

        def log_message(self, format: str, *args) -> None:
            if debug:
                super().log_message(format, *args)

    return Handler


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="VinBank chat UI")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--max-requests", type=int, default=10)
    parser.add_argument("--window-seconds", type=int, default=60)
    args = parser.parse_args(argv)

    app = ChatApp(
        max_requests=args.max_requests, window_seconds=args.window_seconds
    )
    server = ThreadingHTTPServer(
        (args.host, args.port), build_handler(app, debug=args.debug)
    )
    print(f"VinBank chat UI → http://{args.host}:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
