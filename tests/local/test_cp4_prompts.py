"""CP4 attack prompt contract."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))


def test_adversarial_prompts_are_advanced_and_ready():
    from attacks.attacks import adversarial_prompts

    assert len(adversarial_prompts) >= 5
    categories = {p["category"].split(" /")[0] for p in adversarial_prompts}
    assert len(categories) >= 5
    for prompt in adversarial_prompts:
        assert prompt["input"].strip()
        assert "TODO" not in prompt["input"]
        assert len(prompt["input"]) > 60
