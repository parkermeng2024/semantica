"""
Offline tests for the investment-committee (Agno Team) path of the Agno 2.9
demo: scripted deterministic models drive a real Team run — zero network.
"""

from __future__ import annotations

import json
from io import StringIO
from types import SimpleNamespace

import pytest

pytest.importorskip("agno.team.team", reason="committee tests require the real agno Team")

from examples.agno_29_investment_demo import (  # noqa: E402
    DEFAULT_CASE_PATH,
    build_context,
    execute_demo,
    load_case,
    main,
    render_execution,
    validate_team_run,
)


def _tool(name, result, args=None, error=False):
    return SimpleNamespace(
        tool_name=name,
        tool_args=args or {},
        tool_call_error=error,
        result=json.dumps(result) if not isinstance(result, str) else result,
    )


def _policy_tool():
    return _tool(
        "check_policy",
        {
            "compliant": False,
            "violations": [
                "Rule violated: customer_concentration <= 0.35",
                "Rule violated: regulatory_clearance == true",
            ],
            "warnings": [],
        },
    )


def _team_run(decision_id, outcome="rejected"):
    del outcome  # the recorded outcome lives in the audit node, not the trace
    return SimpleNamespace(
        content="主席收口：拒绝投资。",
        tools=[
            _tool("delegate_task_to_member", "分析师结论……"),
            _tool("delegate_task_to_member", "合规结论……"),
            _tool("record_decision", {"decision_id": decision_id, "status": "recorded"}),
        ],
        member_responses=[
            SimpleNamespace(
                agent_name="analyst",
                tools=[
                    _tool("query_graph", {"results": [{"id": "Project Aurora"}], "count": 1}),
                    _tool("find_related", {"related": ["Acme Enterprise"]}),
                    _tool("find_precedents", {"precedents": [], "count": 0}),
                ],
            ),
            SimpleNamespace(agent_name="compliance", tools=[_policy_tool()]),
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


def test_offline_committee_run_passes_team_validation():
    case_data, run_output, validation = execute_demo(committee=True)
    assert case_data["project"] == "Project Aurora"
    assert validation.ok is True, validation.errors
    assert validation.audit_record["metadata"]["outcome"] in {
        "rejected",
        "deferred_with_conditions",
    }
    member_names = {m.agent_name for m in run_output.member_responses}
    assert member_names == {"analyst", "compliance"}
    aggregated = [t.tool_name for t in (run_output.tools or [])]
    for member in run_output.member_responses:
        aggregated.extend(t.tool_name for t in (member.tools or []))
    assert aggregated.count("record_decision") == 1
    assert aggregated.count("check_policy") == 1


def test_main_committee_returns_0_offline_without_api_key(monkeypatch):
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    assert main(["--committee"]) == 0


def test_main_committee_live_returns_2_without_api_key(monkeypatch, capsys):
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    assert main(["--committee", "--live"]) == 2
    assert "DEEPSEEK_API_KEY" in capsys.readouterr().err


def test_validate_team_run_accepts_governed_trace():
    context = build_context()
    decision_id = _record_decision(context)
    result = validate_team_run(_team_run(decision_id), context)
    assert result.ok is True
    assert result.decision_id == decision_id
    assert result.audit_record["metadata"]["outcome"] == "rejected"


def test_validate_team_run_attributes_missing_calls_to_role():
    context = build_context()
    decision_id = _record_decision(context)
    run = _team_run(decision_id)
    run.member_responses = [
        m for m in run.member_responses if m.agent_name != "compliance"
    ]
    result = validate_team_run(run, context)
    assert result.ok is False
    assert "compliance: check_policy must be called exactly once; observed 0" in result.errors


def test_validate_team_run_rejects_duplicate_record_decision():
    context = build_context()
    decision_id = _record_decision(context)
    run = _team_run(decision_id)
    run.tools.append(
        _tool("record_decision", {"decision_id": decision_id, "status": "recorded"})
    )
    result = validate_team_run(run, context)
    assert result.ok is False
    assert "chair: record_decision must be called exactly once; observed 2" in result.errors


def test_validate_team_run_rejects_policy_contradiction():
    context = build_context()
    decision_id = _record_decision(context, outcome="approved")
    result = validate_team_run(_team_run(decision_id), context)
    assert result.ok is False
    assert "outcome 'approved' contradicts policy" in result.errors


def test_render_execution_groups_committee_trace_by_role():
    context = build_context()
    decision_id = _record_decision(context)
    run = _team_run(decision_id)
    validation = validate_team_run(run, context)
    output = StringIO()
    render_execution(load_case(DEFAULT_CASE_PATH), run, validation, stream=output)
    text = output.getvalue()
    assert "分析师 Agent（analyst）" in text
    assert "合规 Agent（compliance）" in text
    assert "主席收口（chair）" in text
    assert decision_id in text


def test_validate_team_run_ignores_builtin_tool_results():
    context = build_context()
    decision_id = _record_decision(context)
    run = _team_run(decision_id)
    run.tools.insert(
        0,
        _tool("search_knowledge_base", "prose, not json"),
    )
    result = validate_team_run(run, context)
    assert result.ok is True, result.errors
