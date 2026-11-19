"""Local tests for Checkpoint 3 pipeline primitives."""
from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import pytest


class _Content:
    def __init__(self, text):
        self.parts = [SimpleNamespace(text=text)]


def _ctx(user_id="student"):
    return SimpleNamespace(user_id=user_id)


# --- rate limit -----------------------------------------------------------

def test_rate_limit_allows_capacity_then_blocks():
    from assignment.rate_limiter import RateLimitPlugin

    plugin = RateLimitPlugin(max_requests=2, window_seconds=60)

    async def run():
        return [
            await plugin.on_user_message_callback(
                invocation_context=_ctx(), user_message=_Content("bank balance")
            )
            for _ in range(3)
        ]

    results = asyncio.run(run())
    assert results[0] is None
    assert results[1] is None
    assert results[2] is not None
    assert plugin.blocked_count == 1
    assert plugin.total_count == 3


# --- audit ----------------------------------------------------------------

def test_audit_records_input_output_latency_and_exports(tmp_path):
    from assignment.audit_log import AuditLogPlugin

    audit = AuditLogPlugin()
    audit.record_input(user_id="u1", text="balance?", request_id="r1")
    audit.record_output(
        user_id="u1",
        text="your balance is 100",
        blocked=False,
        layer=None,
        request_id="r1",
    )

    assert len(audit.logs) == 1
    entry = audit.logs[0]
    assert entry["user_id"] == "u1"
    assert entry["input"] == "balance?"
    assert entry["output"] == "your balance is 100"
    assert entry["blocked"] is False
    assert entry["latency_ms"] >= 0

    path = tmp_path / "audit.json"
    audit.export_json(str(path))
    assert json.loads(path.read_text(encoding="utf-8"))[0]["request_id"] == "r1"


# --- monitoring -----------------------------------------------------------

def test_monitoring_alerts_on_high_block_rate_and_exports(tmp_path):
    from assignment.monitoring import MonitoringAlert

    monitor = MonitoringAlert(block_rate_threshold=0.5)
    monitor.total_requests = 10
    monitor.blocked_requests = 6

    alerts = monitor.check_metrics()
    assert [a.metric for a in alerts] == ["block_rate"]

    path = tmp_path / "metrics.json"
    monitor.export_json(str(path))
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["block_rate"] == pytest.approx(0.6)
    assert payload["alerts"][0]["metric"] == "block_rate"


# --- egress ---------------------------------------------------------------

@pytest.mark.parametrize(
    "destination,payload,expected",
    [
        ("https://api.vinbank.example/v1/transfers", "approved transfer amount 500000", True),
        ("http://api.vinbank.example/v1/transfers", "approved transfer", False),
        ("https://api.vinbank.example.evil.com/v1/transfers", "approved transfer", False),
        ("https://evil.example/collect", "customer account 123456", False),
        ("https://api.vinbank.example/v1/transfers", "password=admin123", False),
        ("https://api.vinbank.example/v1/transfers", "email a@b.com", False),
        ("https://api.vinbank.example/v1/transfers", "call 0901234567", False),
        ("https://api.vinbank.example/v1/transfers", "db.vinbank.internal:5432", False),
    ],
)
def test_egress_allowlist(destination, payload, expected):
    from assignment.pipeline import is_egress_allowed

    assert is_egress_allowed(destination, payload) is expected


# --- suite ----------------------------------------------------------------

def test_assignment_suite_writes_schema_valid_artifacts():
    import jsonschema

    from assignment.pipeline import (
        build_observability,
        build_production_plugins,
        run_assignment_suite,
    )

    pipeline = {
        "plugins": build_production_plugins(),
        "audit": build_observability()[0],
        "monitor": build_observability()[1],
    }
    result = asyncio.run(run_assignment_suite(pipeline))

    root = __import__("pathlib").Path(__file__).resolve().parents[2]
    results_path = root / "outputs" / "results.json"
    assert results_path.is_file()
    assert (root / "outputs" / "audit_log.json").is_file()
    assert (root / "outputs" / "metrics.json").is_file()

    data = json.loads(results_path.read_text(encoding="utf-8"))
    schema = json.loads(
        (root / "schemas" / "results.schema.json").read_text(encoding="utf-8")
    )
    jsonschema.validate(instance=data, schema=schema)
    assert data == result
    assert all(q["blocked"] is False for q in data["safe_queries"])
    assert sum(1 for q in data["attack_queries"] if q["blocked"]) >= 5
    assert data["rate_limit"]["blocked"] >= 1
    assert (
        data["rate_limit"]["passed"] + data["rate_limit"]["blocked"]
        == data["rate_limit"]["sent"]
    )
