"""Tracing adapters that record guardrail verdicts without changing decisions.

Each wrapper delegates to the real plugin and records only what happened.
The real plugin's return value decides ALLOW/BLOCK; the pure guardrail
functions are used to name the reason.
"""
from __future__ import annotations

from guardrails.input_guardrails import detect_injection, topic_filter
from guardrails.output_guardrails import content_filter

_RANK = {"ALLOW": 0, "SKIP": 0, "REDACTED": 1, "BLOCK": 2, "ERROR": 3}


def content_text(content) -> str:
    """Extract plain text from a google.genai Content object."""
    parts = getattr(content, "parts", None) or []
    chunks = []
    for part in parts:
        text = getattr(part, "text", None)
        if text:
            chunks.append(text)
    return "".join(chunks)


class VerdictRecorder:
    """Collects one turn's layers and derives the overall status."""

    def __init__(self) -> None:
        self.layers: list[dict] = []

    def reset(self) -> None:
        self.layers = []

    def record(self, name: str, verdict: str, reason: str = "") -> None:
        self.layers.append({"name": name, "verdict": verdict, "reason": reason})

    def verdict(self) -> dict:
        status = "ALLOW"
        for layer in self.layers:
            if _RANK.get(layer["verdict"], 0) > _RANK.get(status, 0):
                status = layer["verdict"]
        return {"status": status, "layers": list(self.layers)}


class TracedRateLimit:
    def __init__(self, plugin, recorder: VerdictRecorder) -> None:
        self.plugin = plugin
        self.recorder = recorder

    async def on_user_message_callback(self, *, invocation_context, user_message):
        result = await self.plugin.on_user_message_callback(
            invocation_context=invocation_context, user_message=user_message
        )
        if result is None:
            self.recorder.record("rate_limit", "ALLOW")
        else:
            self.recorder.record(
                "rate_limit", "BLOCK", "rate limit: " + content_text(result)
            )
        return result


class TracedInput:
    def __init__(self, plugin, recorder: VerdictRecorder) -> None:
        self.plugin = plugin
        self.recorder = recorder

    async def on_user_message_callback(self, *, invocation_context, user_message):
        text = content_text(user_message)
        injection = detect_injection(text) == "BLOCK"
        topic = topic_filter(text) == "BLOCK"
        result = await self.plugin.on_user_message_callback(
            invocation_context=invocation_context, user_message=user_message
        )
        if result is None:
            self.recorder.record("input", "ALLOW")
        else:
            reason = (
                "injection detected"
                if injection
                else "off-topic or blocked topic"
                if topic
                else "blocked by input guardrail"
            )
            self.recorder.record("input", "BLOCK", reason)
        return result


class TracedOutput:
    def __init__(self, plugin, recorder: VerdictRecorder) -> None:
        self.plugin = plugin
        self.recorder = recorder

    async def after_model_callback(self, *, callback_context, llm_response):
        before = content_text(getattr(llm_response, "content", None))
        issues = content_filter(before)["issues"]
        out = await self.plugin.after_model_callback(
            callback_context=callback_context, llm_response=llm_response
        )
        after = content_text(getattr(out, "content", None))
        if issues or after != before:
            self.recorder.record("output", "REDACTED", "; ".join(issues) or "output redacted")
        else:
            self.recorder.record("output", "ALLOW")
        return out
