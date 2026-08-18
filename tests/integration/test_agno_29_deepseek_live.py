"""
Opt-in paid live smoke test: DeepSeek ``deepseek-v4-pro`` drives the full
governed investment demo.  Skipped unless ``DEEPSEEK_API_KEY`` is set; never
runs in default offline CI.
"""

from __future__ import annotations

import os

import pytest

from examples.agno_29_investment_demo import execute_demo

pytestmark = pytest.mark.integration


@pytest.mark.skipif(
    not os.getenv("DEEPSEEK_API_KEY"),
    reason="DEEPSEEK_API_KEY is required for the paid live smoke test",
)
def test_deepseek_v4_pro_completes_governed_investment_demo():
    case_data, run_output, validation = execute_demo(debug=False, live=True)
    assert case_data["project"] == "Project Aurora"
    assert validation.ok is True, validation.errors
    assert validation.decision_id
    assert validation.audit_record is not None
    assert validation.audit_record["metadata"]["outcome"] in {
        "rejected",
        "deferred_with_conditions",
    }
    names = [tool.tool_name for tool in run_output.tools or []]
    assert names.count("check_policy") == 1
    assert names.count("record_decision") == 1


@pytest.mark.skipif(
    not os.getenv("DEEPSEEK_API_KEY"),
    reason="DEEPSEEK_API_KEY is required for the paid live smoke test",
)
def test_deepseek_v4_pro_completes_governed_committee_demo():
    case_data, run_output, validation = execute_demo(
        debug=False, live=True, committee=True
    )
    assert case_data["project"] == "Project Aurora"
    assert validation.ok is True, validation.errors
    assert validation.decision_id
    assert validation.audit_record is not None
    assert validation.audit_record["metadata"]["outcome"] in {
        "rejected",
        "deferred_with_conditions",
    }
    member_names = {m.agent_name for m in run_output.member_responses or []}
    assert member_names == {"analyst", "compliance"}
    aggregated = [tool.tool_name for tool in run_output.tools or []]
    for member in run_output.member_responses or []:
        aggregated.extend(tool.tool_name for tool in member.tools or [])
    assert aggregated.count("check_policy") == 1
    assert aggregated.count("record_decision") == 1
