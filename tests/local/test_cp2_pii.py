"""CP2 PII coverage for extended formats."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from guardrails.output_guardrails import content_filter  # noqa: E402


@pytest.mark.parametrize(
    "text,pattern_name",
    [
        ("Call me on +84 912 345 678 tomorrow.", "phone_intl"),
        ("Call me on +84912345678 tomorrow.", "phone_intl"),
        ("Call me on +84.912.345.678 tomorrow.", "phone_intl"),
        ("Call 0912 345 678 for the transfer.", "phone_spaced"),
        ("Call 0912.345.678 for the transfer.", "phone_spaced"),
        ("Call 090 123 4567 for the transfer.", "phone_spaced"),
        ("Customer name: Nguyen Van An", "name"),
        ("Họ tên: Trần Thị Bích", "name"),
        ("Address: 123 Nguyen Trai Street, District 5", "address"),
        ("Địa chỉ: 45 Lê Lợi, Phường Bến Nghé", "address"),
        ("Date of birth: 12/03/1990", "dob"),
        ("DOB: 1990-03-12", "dob"),
        ("Ngày sinh: 12.03.1990", "dob"),
        ("Account number: 1234567890123", "account_number"),
        ("Số tài khoản: 1234567890", "account_number"),
        ("Card number: 4111 1111 1111 1111", "card_number"),
        ("CCCD 079 123 456 789", "national_id_spaced"),
    ],
)
def test_extended_pii_is_redacted(text, pattern_name):
    result = content_filter(text)
    assert result["safe"] is False
    assert "[REDACTED]" in result["redacted"]
    assert any(issue.startswith(f"{pattern_name}:") for issue in result["issues"])


@pytest.mark.parametrize(
    "text",
    [
        "The 12-month savings rate is 4.25% per year.",
        "Transfer 500000 VND on 2026-11-19.",
        "Your loan is approved for 36 months.",
        "The API documentation is at https://api.vinbank.example/docs.",
    ],
)
def test_benign_banking_text_is_not_flagged(text):
    assert content_filter(text)["safe"] is True
