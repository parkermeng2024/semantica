"""
Offline tests for the Agno 2.9 investment demo (fixture, seeding, validation,
rendering, CLI contract).  No network access and no DeepSeek key required.
"""

from __future__ import annotations

import json
from io import StringIO
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from examples.agno_29_investment_demo import (
    DEFAULT_CASE_PATH,
    DemoValidationError,
    build_context,
    load_case,
    seed_demo_data,
)
from integrations.agno import AgnoDecisionKit, AgnoKGToolkit


def test_fixture_contains_material_policy_violations():
    case = load_case(DEFAULT_CASE_PATH)
    project = next(item for item in case["entities"] if item["name"] == "Project Aurora")
    assert project["properties"]["customer_concentration"] == 0.46
    assert project["properties"]["regulatory_clearance"] is False
    assert len(case["policy_rules"]) == 3


def test_fixture_contains_due_diligence_documents():
    case = load_case(DEFAULT_CASE_PATH)
    assert len(case["documents"]) >= 2
    assert all(isinstance(doc, str) and doc for doc in case["documents"])


def test_seed_demo_data_round_trips_properties_and_precedents():
    context = build_context()
    kg = AgnoKGToolkit(context=context)
    decisions = AgnoDecisionKit(context=context)
    summary = seed_demo_data(load_case(DEFAULT_CASE_PATH), kg, decisions)
    assert summary.nodes_added == 4
    assert summary.edges_added == 3
    assert len(summary.precedent_ids) == 2
    query = json.loads(kg.query_graph("Project Aurora"))
    assert query["results"][0]["properties"]["arr_usd"] == 12_000_000


def test_seed_demo_data_rejects_tool_errors():
    class BrokenKG:
        def add_to_graph(self, **kwargs):
            return json.dumps({"nodes_added": 0, "edges_added": 0, "error": "write failed"})

    with pytest.raises(DemoValidationError, match="write failed"):
        seed_demo_data(load_case(DEFAULT_CASE_PATH), BrokenKG(), object())


# ---------------------------------------------------------------------------
# validate_run / render_execution
# ---------------------------------------------------------------------------
from examples.agno_29_investment_demo import render_execution, validate_run  # noqa: E402


def _tool(name, result, args=None, error=False):
    return SimpleNamespace(
        tool_name=name,
        tool_args=args or {},
        tool_call_error=error,
        result=json.dumps(result),
    )


def _successful_run(decision_id):
    return SimpleNamespace(
        content="建议拒绝，并在监管许可完成后重新评估。",
        tools=[
            _tool("query_graph", {"results": [{"id": "Project Aurora"}], "count": 1}),
            _tool("find_related", {"related": ["Acme Enterprise"]}),
            _tool("find_precedents", {"precedents": [{"outcome": "rejected"}], "count": 1}),
            _tool(
                "check_policy",
                {
                    "compliant": False,
                    "violations": [
                        "Rule violated: customer_concentration <= 0.35",
                        "Rule violated: regulatory_clearance == true",
                    ],
                    "warnings": [],
                },
            ),
            _tool("record_decision", {"decision_id": decision_id, "status": "recorded"}),
        ],
    )


def _record_decision(context, outcome="rejected"):
    return context.record_decision(
        category="investment_approval",
        scenario="Project Aurora investment review",
        reasoning="Customer concentration and clearance violate policy.",
        outcome=outcome,
        confidence=0.91,
        entities=["Project Aurora"],
    )


def test_validate_run_accepts_complete_governed_trace():
    context = build_context()
    decision_id = _record_decision(context)
    result = validate_run(_successful_run(decision_id), context)
    assert result.ok is True
    assert result.decision_id == decision_id
    assert result.audit_record["metadata"]["outcome"] == "rejected"


def test_validate_run_rejects_missing_and_duplicate_required_calls():
    context = build_context()
    run = SimpleNamespace(content="bad", tools=[])
    result = validate_run(run, context)
    assert result.ok is False
    assert "missing required tool call: query_graph" in result.errors
    assert "record_decision must be called exactly once; observed 0" in result.errors


def test_validate_run_rejects_policy_contradiction():
    context = build_context()
    decision_id = _record_decision(context, outcome="approved")
    result = validate_run(_successful_run(decision_id), context)
    assert result.ok is False
    assert "outcome 'approved' contradicts policy" in result.errors


def test_validate_run_rejects_tool_errors():
    context = build_context()
    decision_id = _record_decision(context)
    run = _successful_run(decision_id)
    run.tools[0] = _tool("query_graph", {}, error=True)
    result = validate_run(run, context)
    assert result.ok is False
    assert "tool call failed: query_graph" in result.errors


def test_render_execution_prints_model_and_audit_sections():
    context = build_context()
    decision_id = _record_decision(context)
    run = _successful_run(decision_id)
    validation = validate_run(run, context)
    output = StringIO()
    render_execution(
        load_case(DEFAULT_CASE_PATH),
        run,
        validation,
        stream=output,
        debug=True,
    )
    text = output.getvalue()
    assert "Agno 工具调用轨迹" in text
    assert "DeepSeek 投资建议" in text
    assert "Semantica 审计记录" in text
    assert "Sanitized tool details" in text
    assert decision_id in text
