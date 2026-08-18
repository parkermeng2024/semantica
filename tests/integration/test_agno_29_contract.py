"""
Offline contract test against the *real* Agno 2.9 package — zero API calls.

Verifies that the Semantica toolkits subclass the real Agno ``Toolkit``,
expose the expected tool counts with valid JSON schemas, and can be attached
to a real Agno ``Agent`` (constructed but never run, so no network traffic).
"""

from __future__ import annotations

from importlib.metadata import version

import pytest

agno = pytest.importorskip("agno", reason="real agno 2.9 package required")

from packaging.specifiers import SpecifierSet  # noqa: E402

from agno.agent import Agent  # noqa: E402
from agno.models.deepseek import DeepSeek  # noqa: E402
from agno.tools import Toolkit  # noqa: E402

from integrations.agno import AgnoDecisionKit, AgnoKGToolkit  # noqa: E402
from semantica.context import AgentContext, ContextGraph  # noqa: E402
from semantica.vector_store import VectorStore  # noqa: E402


def test_real_agno_29_toolkit_contract_without_model_call():
    assert version("agno") in SpecifierSet(">=2.9,<3")
    context = AgentContext(
        vector_store=VectorStore(backend="inmemory"),
        knowledge_graph=ContextGraph(advanced_analytics=False),
        decision_tracking=True,
        advanced_analytics=False,
        kg_algorithms=False,
    )
    kg = AgnoKGToolkit(context=context)
    decisions = AgnoDecisionKit(context=context)
    assert isinstance(kg, Toolkit)
    assert isinstance(decisions, Toolkit)
    assert len(kg.functions) == 7
    assert len(decisions.functions) == 6
    for function in [*kg.functions.values(), *decisions.functions.values()]:
        assert function.parameters["type"] == "object"
        assert isinstance(function.parameters["properties"], dict)
    agent = Agent(
        model=DeepSeek(id="deepseek-v4-pro", api_key="contract-test-key"),
        tools=[kg, decisions],
    )
    assert agent.model.id == "deepseek-v4-pro"


def test_real_agno_29_team_member_tool_traces_enumerable():
    """
    Committee contract: a scripted-model Team run (zero network) must expose
    every member's tool trajectory via ``TeamRunOutput.member_responses`` —
    the foundation of team-level governance invariants.
    """
    from examples.agno_29_investment_demo import (
        DEFAULT_CASE_PATH,
        build_committee,
        build_context,
        load_case,
        seed_demo_data,
        validate_team_run,
    )
    from integrations.agno import AgnoSharedContext

    case_data = load_case(DEFAULT_CASE_PATH)
    context = build_context()
    seed_demo_data(case_data, AgnoKGToolkit(context=context), AgnoDecisionKit(context=context))
    shared = AgnoSharedContext(
        vector_store=context.vector_store,
        knowledge_graph=context.knowledge_graph,
        decision_tracking=True,
        advanced_analytics=False,
        kg_algorithms=False,
    )
    team = build_committee(context, shared, case_data, live=False)
    run_output = team.run(case_data["request_zh"], stream=False)

    member_names = {m.agent_name for m in run_output.member_responses}
    assert member_names == {"analyst", "compliance"}
    aggregated = [t.tool_name for t in (run_output.tools or [])]
    for member in run_output.member_responses:
        aggregated.extend(t.tool_name for t in (member.tools or []))
    assert aggregated.count("record_decision") == 1
    assert aggregated.count("check_policy") == 1

    validation = validate_team_run(run_output, context)
    assert validation.ok is True, validation.errors
