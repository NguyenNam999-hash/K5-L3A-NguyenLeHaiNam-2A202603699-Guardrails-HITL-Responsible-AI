import asyncio
from types import SimpleNamespace

from core.openai_runtime import OpenAIAgent, OpenAIRunner


class _FakeClient:
    def __init__(self):
        self.calls = []
        outer = self

        class Completions:
            def create(self, **kwargs):
                outer.calls.append(kwargs)
                msg = SimpleNamespace(content="ok")
                return SimpleNamespace(choices=[SimpleNamespace(message=msg)])

        self.chat = SimpleNamespace(completions=Completions())


def _runner_with_fake_client():
    runner = OpenAIRunner(app_name="t", model="m", client_kwargs={})
    fake = _FakeClient()
    runner._client = lambda: fake
    return runner, fake


def test_history_is_sent_between_system_and_new_message():
    runner, fake = _runner_with_fake_client()
    agent = OpenAIAgent(name="a", instruction="SYS")
    history = [
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "reply"},
    ]
    out = asyncio.run(runner.chat(agent, "now", history=history))
    assert out == "ok"
    messages = fake.calls[0]["messages"]
    assert [m["role"] for m in messages] == ["system", "user", "assistant", "user"]
    assert messages[1]["content"] == "first"
    assert messages[2]["content"] == "reply"
    assert messages[3]["content"] == "now"


def test_no_history_keeps_system_and_user_only():
    runner, fake = _runner_with_fake_client()
    agent = OpenAIAgent(name="a", instruction="SYS")
    asyncio.run(runner.chat(agent, "hello"))
    assert [m["role"] for m in fake.calls[0]["messages"]] == ["system", "user"]
