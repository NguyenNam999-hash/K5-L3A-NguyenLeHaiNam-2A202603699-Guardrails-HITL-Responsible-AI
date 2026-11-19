"""CP2 public-contact allowlist for content_filter."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from guardrails.output_guardrails import UNREDACTED, content_filter  # noqa: E402


def test_official_contacts_are_allowlisted():
    assert "1900 545 467" in UNREDACTED
    assert "support@vinbank.example" in UNREDACTED


def test_public_hotline_is_not_redacted():
    text = "VinBank hotline: 1900 545 467, open 08:00-22:00."
    result = content_filter(text)
    assert result["safe"] is True
    assert result["redacted"] == text
    assert result["issues"] == []


def test_public_email_is_not_redacted():
    text = "Email support@vinbank.example for account help."
    result = content_filter(text)
    assert result["safe"] is True
    assert result["redacted"] == text


def test_public_email_is_case_insensitive():
    text = "Email Support@VinBank.Example for account help."
    result = content_filter(text)
    assert result["safe"] is True
    assert result["redacted"] == text


def test_private_pii_is_redacted_next_to_public_contacts():
    text = (
        "Hotline 1900 545 467 or support@vinbank.example; "
        "personal phone 0901234567 and email test@vinbank.com."
    )
    result = content_filter(text)
    assert result["safe"] is False
    assert "1900 545 467" in result["redacted"]
    assert "support@vinbank.example" in result["redacted"]
    assert "0901234567" not in result["redacted"]
    assert "test@vinbank.com" not in result["redacted"]
    assert result["redacted"].count("[REDACTED]") == 2
