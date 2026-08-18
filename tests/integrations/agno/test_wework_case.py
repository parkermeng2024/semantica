"""
Self-consistency tests for the WeWork IPO backtest fixture: real disclosed
figures must trigger the expected policy violations, rules must parse, and
the fixture must flow through both offline demo paths.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from examples.agno_29_investment_demo import (
    DemoValidationError,
    _build_decision_data,
    execute_demo,
    load_case,
)
from integrations.agno import AgnoDecisionKit

WEWORK_CASE_PATH = (
    Path(__file__).parents[3] / "examples" / "data" / "wework_ipo_case.json"
)


def test_fixture_has_sources_and_expected_outcome():
    case = load_case(WEWORK_CASE_PATH)
    assert case["metadata"]["expected_outcome"] == "rejected"
    sources = case["metadata"]["sources"]
    assert len(sources) >= 5
    for source in sources:
        assert source["claim"]
        assert source["url"].startswith("https://")


def test_real_figures_match_s1_disclosures():
    case = load_case(WEWORK_CASE_PATH)
    wework = next(item for item in case["entities"] if item["name"] == "WeWork")
    props = wework["properties"]
    assert props["revenue_2018_usd"] == 1_820_000_000
    assert props["net_loss_2018_usd"] == 1_610_000_000
    # 1.61B / 1.82B ≈ 0.885 — loss equal to ~88% of revenue
    assert props["net_loss_to_revenue"] == pytest.approx(
        props["net_loss_2018_usd"] / props["revenue_2018_usd"], abs=0.01
    )
    assert props["governance_red_flags"] >= 2


def test_policy_rules_trigger_expected_violations():
    case = load_case(WEWORK_CASE_PATH)
    kit = AgnoDecisionKit()
    decision_data = _build_decision_data(case, outcome="approved", confidence=0.8)
    result = json.loads(
        kit.check_policy(
            decision_data=json.dumps(decision_data),
            policy_rules=json.dumps(case["policy_rules"]),
        )
    )
    assert result["compliant"] is False
    assert "Rule violated: net_loss_to_revenue <= 0.5" in result["violations"]
    assert "Rule violated: governance_red_flags <= 1" in result["violations"]
    # Every rule must be evaluable against the data — no "could not evaluate"
    # warnings, which would mean a rule field is missing or null.
    assert result["warnings"] == []


def test_build_decision_data_rejects_fixture_missing_rule_field():
    case = load_case(WEWORK_CASE_PATH)
    case["policy_rules"] = ["nonexistent_metric <= 1"]
    with pytest.raises(DemoValidationError, match="nonexistent_metric"):
        _build_decision_data(case, outcome="approved", confidence=0.8)


def test_offline_single_agent_run_passes_for_wework():
    case_data, run_output, validation = execute_demo(case_path=WEWORK_CASE_PATH)
    assert case_data["project"] == "WeWork"
    assert validation.ok is True, validation.errors
    assert validation.audit_record["metadata"]["outcome"] == "rejected"
    assert len(validation.policy_result["violations"]) == 2


def test_offline_committee_run_passes_for_wework():
    pytest.importorskip("agno.team.team", reason="committee path requires real agno Team")
    case_data, run_output, validation = execute_demo(
        committee=True, case_path=WEWORK_CASE_PATH
    )
    assert validation.ok is True, validation.errors
    assert validation.audit_record["metadata"]["outcome"] == "rejected"
    member_names = {m.agent_name for m in run_output.member_responses}
    assert member_names == {"analyst", "compliance"}
