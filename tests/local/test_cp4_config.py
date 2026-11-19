"""CP4 provider plumbing: dedicated Red key/base URL."""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))


def test_red_client_uses_dedicated_key_and_base_url(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "openai-key")
    monkeypatch.setenv("RED_TEAM_API_KEY", "red-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://gateway.example/v1")

    from core.config import red_openai_client_kwargs

    kwargs = red_openai_client_kwargs()
    assert kwargs["api_key"] == "red-key"
    assert kwargs["base_url"] == "https://gateway.example/v1"


def test_nemo_openai_model_entry_honors_base_url(monkeypatch):
    monkeypatch.setenv("RED_TEAM_API_KEY", "red-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://gateway.example/v1")

    from agents.red_advance_nemo import build_rails_config

    config = build_rails_config(provider="openai", model="m-test")
    main = config.models[0]
    assert main.parameters.get("base_url") == "https://gateway.example/v1"
    assert main.api_key_env_var == "RED_TEAM_API_KEY"
