"""
Agno 2.9 full-stack investment committee demo (single-Agent act).

An Agno ``Agent`` powered by ``deepseek-v4-pro`` evaluates the Project Aurora
investment case under governance constraints:

- ``tools=[AgnoKGToolkit, AgnoDecisionKit]`` — graph evidence + decision tools
- ``db=AgnoContextStore`` — agent memory backed by the Semantica context graph
- ``knowledge=AgnoKnowledgeGraph`` — due-diligence documents as GraphRAG

The default CLI path is fully offline: without ``--live`` a deterministic
simulated run is driven through the same ``validate_run`` / ``render_execution``
pipeline, so the whole governance chain can be demonstrated without an API
key.  ``--live`` builds the real DeepSeek-backed Agent (requires
``DEEPSEEK_API_KEY``).

Exit codes: ``0`` governed PASS, ``2`` configuration error, ``3`` model
error, ``4`` governance/validation failure.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, TextIO, Tuple

from semantica.utils.logging import get_logger

if TYPE_CHECKING:
    from semantica.context import AgentContext

logger = get_logger(__name__)

DEFAULT_CASE_PATH = Path(__file__).parent / "data" / "agno_investment_case.json"

MODEL_ID = "deepseek-v4-pro"


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------
class DemoValidationError(RuntimeError):
    """Raised when fixture loading, seeding, or run validation fails."""


class DemoConfigurationError(DemoValidationError):
    """Raised when runtime configuration (key, agno install) is missing."""


class DemoModelError(RuntimeError):
    """Raised when the DeepSeek request itself fails."""


# ---------------------------------------------------------------------------
# Fixture loading and deterministic seeding
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class SeedSummary:
    nodes_added: int
    edges_added: int
    precedent_ids: Tuple[str, ...]


def load_case(path: Path) -> Dict[str, Any]:
    """Load and structurally validate the investment case fixture."""
    data = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "project",
        "category",
        "entities",
        "relations",
        "precedents",
        "policy_rules",
        "documents",
        "request_zh",
    }
    missing = sorted(required.difference(data))
    if missing:
        raise DemoValidationError(f"case fixture is missing keys: {', '.join(missing)}")
    return data


def build_context() -> AgentContext:
    """Build the in-memory AgentContext shared by both toolkits."""
    from semantica.context import AgentContext, ContextGraph
    from semantica.vector_store import VectorStore

    graph = ContextGraph(advanced_analytics=False)
    vectors = VectorStore(backend="inmemory")
    return AgentContext(
        vector_store=vectors,
        knowledge_graph=graph,
        decision_tracking=True,
        advanced_analytics=False,
        kg_algorithms=False,
    )


def _parse_tool_json(raw: str, operation: str) -> Dict[str, Any]:
    """Parse a toolkit JSON result, rejecting malformed or errored payloads."""
    try:
        result = json.loads(raw)
    except (TypeError, json.JSONDecodeError) as exc:
        raise DemoValidationError(f"{operation} returned invalid JSON") from exc
    if not isinstance(result, dict):
        raise DemoValidationError(f"{operation} returned a non-object result")
    if result.get("error"):
        raise DemoValidationError(f"{operation} failed: {result['error']}")
    return result


def seed_demo_data(
    case_data: Dict[str, Any],
    kg_toolkit: Any,
    decision_toolkit: Any,
) -> SeedSummary:
    """
    Deterministically seed graph evidence and historical precedents.

    Every toolkit response is validated; any tool error or property loss
    aborts the demo before the model ever runs.
    """
    add_result = _parse_tool_json(
        kg_toolkit.add_to_graph(
            entities=json.dumps(case_data["entities"]),
            relations=json.dumps(case_data["relations"]),
        ),
        "add_to_graph",
    )
    nodes_added = add_result.get("nodes_added", 0)
    edges_added = add_result.get("edges_added", 0)
    if nodes_added != len(case_data["entities"]):
        raise DemoValidationError(
            f"expected {len(case_data['entities'])} nodes, got {nodes_added}"
        )
    if edges_added != len(case_data["relations"]):
        raise DemoValidationError(
            f"expected {len(case_data['relations'])} edges, got {edges_added}"
        )

    # Property round-trip: the graph must return the fixture properties.
    query_result = _parse_tool_json(
        kg_toolkit.query_graph(case_data["project"]), "query_graph"
    )
    project = next(
        item for item in case_data["entities"] if item["name"] == case_data["project"]
    )
    returned = (query_result.get("results") or [{}])[0].get("properties", {})
    for key, value in project["properties"].items():
        if returned.get(key) != value:
            raise DemoValidationError(
                f"property round-trip failed for {key!r}: "
                f"expected {value!r}, got {returned.get(key)!r}"
            )

    precedent_ids: List[str] = []
    for precedent in case_data["precedents"]:
        record_result = _parse_tool_json(
            decision_toolkit.record_decision(**precedent), "record_decision"
        )
        if record_result.get("status") != "recorded":
            raise DemoValidationError(
                f"record_decision returned unexpected status: {record_result}"
            )
        precedent_ids.append(record_result["decision_id"])

    return SeedSummary(
        nodes_added=nodes_added,
        edges_added=edges_added,
        precedent_ids=tuple(precedent_ids),
    )


# ---------------------------------------------------------------------------
# Run validation — governance checks independent of the model's prose
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ValidationResult:
    ok: bool
    errors: Tuple[str, ...]
    decision_id: Optional[str]
    policy_result: Dict[str, Any]
    audit_record: Optional[Dict[str, Any]]


def validate_run(run_output: Any, context: AgentContext) -> ValidationResult:
    """
    Validate an Agno run against the governance contract.

    Checks the tool trace (``RunOutput.tools``), the policy evaluation, the
    single final decision, and the Semantica audit read-back — none of which
    depend on the model's generated prose.
    """
    required_minimum = {
        "query_graph": 1,
        "find_related": 1,
        "find_precedents": 1,
    }
    required_exact = {"check_policy": 1, "record_decision": 1}

    errors: List[str] = []
    counts, policy_result, decision_result = _collect_tool_results(
        getattr(run_output, "tools", None) or [], errors
    )

    for name, minimum in required_minimum.items():
        if counts.get(name, 0) < minimum:
            errors.append(f"missing required tool call: {name}")
    for name, exactly in required_exact.items():
        if counts.get(name, 0) != exactly:
            errors.append(
                f"{name} must be called exactly once; observed {counts.get(name, 0)}"
            )

    decision_id, audit_record = _audit_decision(
        decision_result, policy_result, context, errors
    )

    return ValidationResult(
        ok=not errors,
        errors=tuple(errors),
        decision_id=decision_id,
        policy_result=policy_result,
        audit_record=audit_record,
    )


#: Governance evidence tools — the seven ``AgnoKGToolkit`` tools plus the six
#: ``AgnoDecisionKit`` tools.  Only these are required to return JSON objects
#: without an ``error`` field; anything else in the trace (Agno built-ins such
#: as ``search_knowledge_base`` or ``delegate_task_to_member``) is
#: orchestration plumbing whose results may be prose, so it is only checked
#: for ``tool_call_error``.
GOVERNANCE_TOOLS = frozenset(
    {
        # AgnoKGToolkit
        "extract_entities",
        "extract_relations",
        "add_to_graph",
        "query_graph",
        "find_related",
        "infer_facts",
        "export_subgraph",
        # AgnoDecisionKit
        "record_decision",
        "find_precedents",
        "trace_causal_chain",
        "analyze_impact",
        "check_policy",
        "get_decision_summary",
    }
)


def _collect_tool_results(
    tools: List[Any],
    errors: List[str],
    prefix: str = "",
) -> Tuple[Dict[str, int], Dict[str, Any], Dict[str, Any]]:
    """
    Count tool calls and harvest the policy / decision results from a tool
    trace.  ``prefix`` scopes error messages to a committee role (empty for
    single-Agent runs).  Only ``GOVERNANCE_TOOLS`` are JSON-validated.
    """
    counts: Dict[str, int] = {}
    policy_result: Dict[str, Any] = {}
    decision_result: Dict[str, Any] = {}

    for tool in tools:
        name = getattr(tool, "tool_name", "") or ""
        counts[name] = counts.get(name, 0) + 1
        if getattr(tool, "tool_call_error", False):
            errors.append(f"{prefix}tool call failed: {name}")
            continue
        if name not in GOVERNANCE_TOOLS:
            # Agno built-in / orchestration plumbing, not governance
            # evidence — its result may be prose, not toolkit JSON.
            continue
        try:
            result = json.loads(getattr(tool, "result", "") or "")
        except (TypeError, json.JSONDecodeError):
            errors.append(f"{prefix}tool '{name}' returned invalid JSON")
            continue
        if not isinstance(result, dict):
            errors.append(f"{prefix}tool '{name}' returned a non-object result")
            continue
        if result.get("error"):
            errors.append(f"{prefix}tool '{name}' failed: {result['error']}")
            continue
        if name == "check_policy":
            policy_result = result
        elif name == "record_decision":
            decision_result = result

    return counts, policy_result, decision_result


def _audit_decision(
    decision_result: Dict[str, Any],
    policy_result: Dict[str, Any],
    context: AgentContext,
    errors: List[str],
) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
    """
    Audit read-back: locate the recorded decision node, verify its type, and
    enforce the governed outcome set / policy-contradiction rule.
    """
    accepted_outcomes = {"rejected", "deferred_with_conditions"}

    decision_id: Optional[str] = decision_result.get("decision_id")
    audit_record: Optional[Dict[str, Any]] = None

    if decision_result and not decision_id:
        errors.append("record_decision result carries no decision_id")

    if decision_id:
        audit_record = context.knowledge_graph.find_node(decision_id)
        if audit_record is None:
            errors.append(f"audit read-back failed: decision {decision_id} not found")
        else:
            node_type = str(audit_record.get("type", "")).lower()
            if node_type != "decision":
                errors.append(
                    f"audit read-back returned node type {node_type!r}, expected 'decision'"
                )
            outcome = (audit_record.get("metadata") or {}).get("outcome")
            compliant = policy_result.get("compliant", True)
            if outcome == "approved" and not compliant:
                errors.append("outcome 'approved' contradicts policy")
            elif outcome not in accepted_outcomes:
                errors.append(
                    f"recorded outcome {outcome!r} is outside the governed set "
                    f"{sorted(accepted_outcomes)}"
                )

    return decision_id, audit_record


# ---------------------------------------------------------------------------
# Committee (Team) validation — team-level invariants over the aggregated
# trace: leader tools + every member's tools, attributed by role
# ---------------------------------------------------------------------------
#: Default per-role call expectations for the investment committee.
#: Values are ``(minimum, exact)`` — ``exact=None`` means "at least minimum".
COMMITTEE_ROLE_EXPECTATIONS: Dict[str, Dict[str, Tuple[int, Optional[int]]]] = {
    "analyst": {
        "query_graph": (1, None),
        "find_related": (1, None),
        "find_precedents": (1, None),
    },
    "compliance": {"check_policy": (1, 1)},
    "chair": {"record_decision": (1, 1)},
}


def _committee_traces(run_output: Any) -> Dict[str, List[Any]]:
    """
    Split a ``TeamRunOutput`` into per-role tool traces.  The leader's own
    tools are keyed ``"chair"``; each member response is keyed by its
    ``agent_name``.
    """
    traces: Dict[str, List[Any]] = {
        "chair": list(getattr(run_output, "tools", None) or [])
    }
    for member in getattr(run_output, "member_responses", None) or []:
        role = getattr(member, "agent_name", None) or "unknown"
        traces[role] = list(getattr(member, "tools", None) or [])
    return traces


def validate_team_run(
    run_output: Any,
    context: AgentContext,
    role_expectations: Optional[Dict[str, Dict[str, Tuple[int, Optional[int]]]]] = None,
) -> ValidationResult:
    """
    Validate a committee (Agno ``Team``) run against team-level invariants.

    Each role's trace is checked against its expectations — ``record_decision``
    exactly once team-wide (chair), ``check_policy`` exactly once (compliance),
    graph/precedent evidence at least once (analyst) — with error messages
    attributing failures to the responsible role.  Policy-contradiction and
    audit read-back rules are identical to the single-Agent path.
    """
    expectations = role_expectations or COMMITTEE_ROLE_EXPECTATIONS
    traces = _committee_traces(run_output)

    errors: List[str] = []
    policy_result: Dict[str, Any] = {}
    decision_result: Dict[str, Any] = {}

    for role, expected in expectations.items():
        role_tools = traces.get(role, [])
        counts, role_policy, role_decision = _collect_tool_results(
            role_tools, errors, prefix=f"{role}: "
        )
        if role_policy:
            policy_result = role_policy
        if role_decision:
            decision_result = role_decision
        for name, (minimum, exact) in expected.items():
            observed = counts.get(name, 0)
            if exact is not None:
                if observed != exact:
                    errors.append(
                        f"{role}: {name} must be called exactly once; "
                        f"observed {observed}"
                    )
            elif observed < minimum:
                errors.append(f"{role}: missing required tool call: {name}")

    decision_id, audit_record = _audit_decision(
        decision_result, policy_result, context, errors
    )

    return ValidationResult(
        ok=not errors,
        errors=tuple(errors),
        decision_id=decision_id,
        policy_result=policy_result,
        audit_record=audit_record,
    )


# ---------------------------------------------------------------------------
# Deterministic terminal rendering — no secrets, ever
# ---------------------------------------------------------------------------
def _trace_groups(run_output: Any) -> List[Tuple[str, List[Any]]]:
    """
    Group a run's tool trace for rendering.  Committee runs yield one group
    per member (labelled by role) plus the chair group; single-Agent runs
    yield a single unlabelled group.
    """
    members = list(getattr(run_output, "member_responses", None) or [])
    if not members:
        return [("", list(getattr(run_output, "tools", None) or []))]

    role_labels = {"analyst": "分析师 Agent", "compliance": "合规 Agent"}
    groups: List[Tuple[str, List[Any]]] = []
    for member in members:
        role = getattr(member, "agent_name", None) or "unknown"
        label = role_labels.get(role, f"成员 {role}")
        groups.append((f"{label}（{role}）", list(getattr(member, "tools", None) or [])))
    groups.append(("主席收口（chair）", list(getattr(run_output, "tools", None) or [])))
    return groups


def render_execution(
    case_data: Dict[str, Any],
    run_output: Any,
    validation: ValidationResult,
    stream: TextIO,
    debug: bool = False,
) -> None:
    """
    Render the case summary, tool trace, model recommendation, and audit
    record to ``stream``.  The debug block contains only tool names,
    arguments, and results — never client configuration, headers, or
    exception objects that may retain request data.
    """
    print("\n=== 投资案例摘要 ===", file=stream)
    print(f"项目: {case_data['project']}", file=stream)
    print(f"类别: {case_data['category']}", file=stream)
    print("政策规则:", file=stream)
    for rule in case_data["policy_rules"]:
        print(f"- {rule}", file=stream)

    groups = _trace_groups(run_output)
    print("\n=== Agno 工具调用轨迹 ===", file=stream)
    for label, tools in groups:
        if label:
            print(f"--- {label} ---", file=stream)
        for tool in tools:
            state = "ERROR" if getattr(tool, "tool_call_error", False) else "OK"
            print(f"[{state}] {tool.tool_name}", file=stream)
    if debug:
        print("\n=== Sanitized tool details ===", file=stream)
        for label, tools in groups:
            for tool in tools:
                details = {
                    "group": label or None,
                    "name": tool.tool_name,
                    "args": getattr(tool, "tool_args", None) or {},
                    "result": getattr(tool, "result", None),
                }
                print(json.dumps(details, ensure_ascii=False, indent=2), file=stream)

    print("\n=== DeepSeek 投资建议 ===", file=stream)
    print(str(getattr(run_output, "content", "")), file=stream)

    print("\n=== Semantica 审计记录 ===", file=stream)
    print(json.dumps(validation.audit_record, ensure_ascii=False, indent=2), file=stream)
    print(
        json.dumps(validation.policy_result, ensure_ascii=False, indent=2),
        file=stream,
    )

    print(f"\nDemo status: {'PASS' if validation.ok else 'FAIL'}", file=stream)
    for error in validation.errors:
        print(f"- {error}", file=stream)


# ---------------------------------------------------------------------------
# Agent construction (live mode) and deterministic offline simulation
# ---------------------------------------------------------------------------
def build_instructions(case_data: Dict[str, Any]) -> List[str]:
    """Semi-constrained instructions that enforce the governed tool sequence."""
    return [
        "Respond in Chinese, but record the machine outcome in English.",
        "Call query_graph for Project Aurora before making any recommendation. "
        "query_graph accepts plain natural-language keywords only — never "
        "pass Cypher (no MATCH clauses); Cypher is not supported here.",
        "Call find_related for Project Aurora with hops=2.",
        "Call find_precedents with category investment_approval.",
        "Create a candidate decision_data object containing category, outcome, "
        "confidence, customer_concentration, and regulatory_clearance.",
        f"Call check_policy exactly once with these rules: {json.dumps(case_data['policy_rules'])}.",
        "If policy is not compliant, do not record outcome approved.",
        "Call record_decision exactly once after policy evaluation.",
        "The recorded outcome must be rejected or deferred_with_conditions.",
        "Include the decision ID, confidence, graph evidence, precedent "
        "comparison, and policy violations in the final answer.",
    ]


def _require_runtime() -> None:
    """Validate the live-runtime prerequisites (API key + agno 2.9.x)."""
    import os
    from importlib.metadata import PackageNotFoundError, version

    from packaging.version import Version

    if not os.getenv("DEEPSEEK_API_KEY"):
        raise DemoConfigurationError(
            "DEEPSEEK_API_KEY is required for --live; export it before running the demo"
        )
    try:
        installed = Version(version("agno"))
    except PackageNotFoundError as exc:
        raise DemoConfigurationError("Agno is not installed") from exc
    if installed.release[:2] != (2, 9):
        raise DemoConfigurationError(
            f"Agno 2.9.x is required; installed version is {installed}"
        )


def build_agent(
    kg_toolkit: Any,
    decision_toolkit: Any,
    context: AgentContext,
    case_data: Dict[str, Any],
    debug: bool = False,
) -> Any:
    """
    Build the full-stack Agno 2.9 Agent: both Toolkits as ``tools``,
    ``AgnoContextStore`` as ``db``, and ``AgnoKnowledgeGraph`` as
    ``knowledge`` with the fixture's due-diligence documents ingested.
    """
    from agno.agent import Agent
    from agno.models.deepseek import DeepSeek

    from integrations.agno import AgnoContextStore, AgnoKnowledgeGraph

    knowledge = AgnoKnowledgeGraph(context_graph=context.knowledge_graph)
    knowledge.load(texts=list(case_data["documents"]))

    model = DeepSeek(
        id=MODEL_ID,
        use_thinking=True,
        max_retries=1,
        timeout=60.0,
    )
    return Agent(
        name="Semantica Investment Committee",
        model=model,
        tools=[kg_toolkit, decision_toolkit],
        db=AgnoContextStore(knowledge_graph=context.knowledge_graph),
        knowledge=knowledge,
        instructions=build_instructions(case_data),
        markdown=True,
        debug_mode=debug,
    )


class _SimulatedTool:
    """Deterministic stand-in for an Agno ``ToolExecution`` record."""

    __slots__ = ("tool_name", "tool_args", "tool_call_error", "result")

    def __init__(self, tool_name: str, tool_args: Dict[str, Any], result: str) -> None:
        self.tool_name = tool_name
        self.tool_args = tool_args
        self.tool_call_error = False
        self.result = result


def _simulate_run(
    case_data: Dict[str, Any],
    kg_toolkit: Any,
    decision_toolkit: Any,
) -> Any:
    """
    Offline deterministic run: drives the real toolkits through the governed
    sequence without a model, producing a ``RunOutput``-shaped object that
    flows through the same ``validate_run`` / ``render_execution`` pipeline.
    """
    project = case_data["project"]
    project_entity = next(
        item for item in case_data["entities"] if item["name"] == project
    )
    props = project_entity["properties"]

    tools: List[_SimulatedTool] = []

    graph_args = {"query": project}
    tools.append(_SimulatedTool("query_graph", graph_args, kg_toolkit.query_graph(**graph_args)))

    related_args = {"entity": project, "hops": 2}
    tools.append(_SimulatedTool("find_related", related_args, kg_toolkit.find_related(**related_args)))

    precedent_args = {
        "scenario": f"{project} investment review",
        "category": case_data["category"],
    }
    tools.append(
        _SimulatedTool(
            "find_precedents",
            precedent_args,
            decision_toolkit.find_precedents(**precedent_args),
        )
    )

    decision_data = {
        "category": case_data["category"],
        "outcome": "rejected",
        "confidence": 0.9,
        "customer_concentration": props["customer_concentration"],
        "regulatory_clearance": props["regulatory_clearance"],
    }
    policy_args = {
        "decision_data": json.dumps(decision_data),
        "policy_rules": json.dumps(case_data["policy_rules"]),
    }
    policy_raw = decision_toolkit.check_policy(**policy_args)
    tools.append(_SimulatedTool("check_policy", policy_args, policy_raw))

    policy_result = json.loads(policy_raw)
    outcome = "rejected" if not policy_result.get("compliant", True) else "approved"
    record_args = {
        "category": case_data["category"],
        "scenario": f"{project} investment review",
        "reasoning": (
            "Deterministic offline run: customer_concentration="
            f"{props['customer_concentration']}, regulatory_clearance="
            f"{props['regulatory_clearance']} → policy compliant="
            f"{policy_result.get('compliant')}."
        ),
        "outcome": outcome,
        "confidence": 0.9,
        "entities": project,
    }
    tools.append(
        _SimulatedTool(
            "record_decision",
            record_args,
            decision_toolkit.record_decision(**record_args),
        )
    )

    content = (
        f"【离线确定性模拟运行】基于图谱证据与先例对比，{project} 客户集中度 "
        f"{props['customer_concentration']:.0%}（上限 35%），监管许可状态为"
        f"「{'已获批' if props['regulatory_clearance'] else '待批'}」，违反两条政策红线。"
        f"建议：{('拒绝投资，待监管许可完成且客户集中度下降后重新评估。' if outcome == 'rejected' else '批准投资。')}"
        "本段为无模型的确定性输出，不经过 DeepSeek 生成。"
    )
    return SimpleNamespaceShim(content=content, tools=tools)


class SimpleNamespaceShim:
    """Minimal RunOutput-shaped object (content + tools)."""

    __slots__ = ("content", "tools")

    def __init__(self, content: str, tools: List[_SimulatedTool]) -> None:
        self.content = content
        self.tools = tools


# ---------------------------------------------------------------------------
# Committee (Team) construction — analyst + compliance members, chair leader
# ---------------------------------------------------------------------------
def _make_scripted_model(steps: List[Tuple[str, Any]], model_id: str) -> Any:
    """
    Build a deterministic offline ``agno.models.base.Model`` subclass whose
    ``invoke`` replays ``steps``: ``("tool", name, args_dict)`` entries emit a
    tool call, ``("text", content)`` entries emit final assistant text.  Zero
    network — the committee path stays testable in CI.

    NOTE: instance attributes must not shadow ``Model._tool_name`` (a method
    on the base class).
    """
    from agno.models.base import Model
    from agno.models.response import ModelResponse

    class ScriptedModel(Model):
        def __init__(self) -> None:
            super().__init__(id=model_id, provider="scripted")
            self._script = list(steps)
            self._step_idx = 0

        def invoke(self, *args: Any, **kwargs: Any) -> Any:
            step = self._script[min(self._step_idx, len(self._script) - 1)]
            self._step_idx += 1
            if step[0] == "tool":
                return ModelResponse(
                    role="assistant",
                    tool_calls=[
                        {
                            "id": f"call_{self._step_idx}",
                            "type": "function",
                            "function": {
                                "name": step[1],
                                "arguments": json.dumps(step[2]),
                            },
                        }
                    ],
                )
            return ModelResponse(role="assistant", content=step[1])

        async def ainvoke(self, *args: Any, **kwargs: Any) -> Any:
            return self.invoke(*args, **kwargs)

        def invoke_stream(self, *args: Any, **kwargs: Any) -> Any:
            yield self.invoke(*args, **kwargs)

        async def ainvoke_stream(self, *args: Any, **kwargs: Any) -> Any:
            yield self.invoke(*args, **kwargs)

        def _parse_provider_response(self, response: Any, **kwargs: Any) -> Any:
            return ModelResponse(role="assistant", content="parsed")

        def _parse_provider_response_delta(self, response: Any) -> Any:
            return ModelResponse(role="assistant", content="delta")

    return ScriptedModel()


def _prune_tools(toolkit: Any, keep: set) -> Any:
    """Restrict a toolkit to the named tools (role separation in the Team)."""
    toolkit.functions = {k: v for k, v in toolkit.functions.items() if k in keep}
    toolkit._tools = [fn for fn in toolkit._tools if fn.__name__ in keep]
    return toolkit


def _committee_instructions(role: str, case_data: Dict[str, Any]) -> List[str]:
    """Live-mode role instructions for the investment committee."""
    if role == "analyst":
        return [
            "You are the analyst on an investment committee. Respond in Chinese.",
            f"Call query_graph for {case_data['project']}.",
            f"Call find_related for {case_data['project']} with hops=2.",
            f"Call find_precedents with category {case_data['category']}.",
            "Summarise the graph evidence and precedent comparison; do not "
            "record any decision.",
        ]
    if role == "compliance":
        return [
            "You are the compliance officer on an investment committee. "
            "Respond in Chinese.",
            "Build a decision_data object containing category, outcome, "
            "confidence, customer_concentration, and regulatory_clearance.",
            f"Call check_policy exactly once with these rules: {json.dumps(case_data['policy_rules'])}.",
            "Report the violations; do not record any decision.",
        ]
    return [
        "You are the chair of an investment committee. Respond in Chinese, "
        "but record the machine outcome in English.",
        "Delegate the evidence review to member_id 'analyst' and the policy "
        "review to member_id 'compliance' via delegate_task_to_member.",
        "If policy is not compliant, do not record outcome approved.",
        "Call record_decision exactly once after both members reported.",
        "The recorded outcome must be rejected or deferred_with_conditions.",
        "Close with a Chinese synthesis citing the decision ID, graph "
        "evidence, precedent comparison, and policy violations.",
    ]


def build_committee(
    shared: Any,
    case_data: Dict[str, Any],
    live: bool = False,
    debug: bool = False,
) -> Any:
    """
    Build the investment-committee Agno ``Team`` (coordinate mode).

    - analyst: ``query_graph`` / ``find_related`` / ``find_precedents``
    - compliance: ``check_policy``
    - chair (Team leader): ``record_decision`` exactly once, after delegating
      to both members

    All toolkits attach directly to the ``AgnoSharedContext`` (the chair's
    final decision stays untagged); both members additionally run with
    ``db=shared.bind_agent(role)`` — role-scoped views over the one shared
    ``ContextGraph``.  Offline mode drives the real Team with scripted
    deterministic models (zero network); live mode uses ``deepseek-v4-pro``
    for all three roles.
    """
    from agno.agent import Agent
    from agno.team.team import Team

    from integrations.agno import AgnoDecisionKit, AgnoKGToolkit

    kg = _prune_tools(AgnoKGToolkit(context=shared), {"query_graph", "find_related"})
    analyst_decisions = _prune_tools(
        AgnoDecisionKit(context=shared), {"find_precedents"}
    )
    compliance_kit = _prune_tools(AgnoDecisionKit(context=shared), {"check_policy"})
    chair_kit = _prune_tools(AgnoDecisionKit(context=shared), {"record_decision"})

    props = next(
        item for item in case_data["entities"] if item["name"] == case_data["project"]
    )["properties"]

    if live:
        from agno.models.deepseek import DeepSeek

        def _model() -> Any:
            return DeepSeek(id=MODEL_ID, use_thinking=True, max_retries=1, timeout=60.0)

        analyst_model = _model()
        compliance_model = _model()
        chair_model = _model()
        analyst_kwargs: Dict[str, Any] = {
            "instructions": _committee_instructions("analyst", case_data)
        }
        compliance_kwargs: Dict[str, Any] = {
            "instructions": _committee_instructions("compliance", case_data)
        }
        chair_kwargs: Dict[str, Any] = {
            "instructions": _committee_instructions("chair", case_data)
        }
    else:
        decision_data = {
            "category": case_data["category"],
            "outcome": "rejected",
            "confidence": 0.9,
            "customer_concentration": props["customer_concentration"],
            "regulatory_clearance": props["regulatory_clearance"],
        }
        analyst_model = _make_scripted_model(
            [
                ("tool", "query_graph", {"query": case_data["project"]}),
                ("tool", "find_related", {"entity": case_data["project"], "hops": 2}),
                (
                    "tool",
                    "find_precedents",
                    {
                        "scenario": f"{case_data['project']} investment review",
                        "category": case_data["category"],
                    },
                ),
                ("text", "分析师结论：图谱证据显示客户集中度 46% 超标，先例 Project Atlas 曾被拒。"),
            ],
            "scripted-analyst",
        )
        compliance_model = _make_scripted_model(
            [
                (
                    "tool",
                    "check_policy",
                    {
                        "decision_data": json.dumps(decision_data),
                        "policy_rules": json.dumps(case_data["policy_rules"]),
                    },
                ),
                ("text", "合规结论：违反客户集中度与监管许可两条政策红线。"),
            ],
            "scripted-compliance",
        )
        chair_model = _make_scripted_model(
            [
                (
                    "tool",
                    "delegate_task_to_member",
                    {"member_id": "analyst", "task": "评估图谱证据与历史先例"},
                ),
                (
                    "tool",
                    "delegate_task_to_member",
                    {"member_id": "compliance", "task": "执行政策闸门评估"},
                ),
                (
                    "tool",
                    "record_decision",
                    {
                        "category": case_data["category"],
                        "scenario": f"{case_data['project']} investment review",
                        "reasoning": (
                            "客户集中度 46% 超过 35% 上限且监管许可待批，"
                            "与先例 Project Atlas 一致。"
                        ),
                        "outcome": "rejected",
                        "confidence": 0.9,
                        "entities": case_data["project"],
                    },
                ),
                (
                    "text",
                    "【离线确定性模拟运行】主席收口：综合分析师图谱证据与合规政策评估，"
                    "拒绝 Project Aurora 的 500 万美元投资，待监管许可完成且客户集中度"
                    "下降后重新评估。本段为无模型的确定性输出，不经过 DeepSeek 生成。",
                ),
            ],
            "scripted-chair",
        )
        analyst_kwargs = {}
        compliance_kwargs = {}
        chair_kwargs = {}

    analyst = Agent(
        name="analyst",
        model=analyst_model,
        tools=[kg, analyst_decisions],
        db=shared.bind_agent("analyst"),
        markdown=True,
        debug_mode=debug,
        **analyst_kwargs,
    )
    compliance = Agent(
        name="compliance",
        model=compliance_model,
        tools=[compliance_kit],
        db=shared.bind_agent("compliance"),
        markdown=True,
        debug_mode=debug,
        **compliance_kwargs,
    )
    return Team(
        name="investment_committee",
        mode="coordinate",
        model=chair_model,
        members=[analyst, compliance],
        tools=[chair_kit],
        debug_mode=debug,
        **chair_kwargs,
    )


# ---------------------------------------------------------------------------
# Orchestration and CLI
# ---------------------------------------------------------------------------
def execute_demo(
    debug: bool = False,
    live: bool = False,
    committee: bool = False,
) -> Tuple[Dict[str, Any], Any, ValidationResult]:
    """
    Seed the demo data, run (live DeepSeek or offline simulation), and
    validate the governed trace.  With ``committee=True`` the run is executed
    by the investment-committee ``Team`` and checked against team-level
    invariants.  Returns the case data, the run output, and the validation
    result.
    """
    from integrations.agno import (
        AGNO_TOOLKITS_AVAILABLE,
        AgnoDecisionKit,
        AgnoKGToolkit,
    )

    case_data = load_case(DEFAULT_CASE_PATH)
    context = build_context()
    kg_toolkit = AgnoKGToolkit(context=context)
    decision_toolkit = AgnoDecisionKit(context=context)
    seed_demo_data(case_data, kg_toolkit, decision_toolkit)

    if live:
        _require_runtime()
        if not AGNO_TOOLKITS_AVAILABLE:
            raise DemoConfigurationError("Agno 2.9 Toolkit integration is unavailable")

    if committee:
        from integrations.agno import AgnoSharedContext

        shared = AgnoSharedContext(
            vector_store=context.vector_store,
            knowledge_graph=context.knowledge_graph,
            decision_tracking=True,
            advanced_analytics=False,
            kg_algorithms=False,
        )
        team = build_committee(shared, case_data, live=live, debug=debug)
        try:
            run_output = team.run(case_data["request_zh"], stream=False)
        except Exception as exc:
            raise DemoModelError(
                f"DeepSeek request failed: {type(exc).__name__}"
            ) from exc
        validation = validate_team_run(run_output, context)
        return case_data, run_output, validation

    if live:
        agent = build_agent(kg_toolkit, decision_toolkit, context, case_data, debug=debug)
        try:
            run_output = agent.run(case_data["request_zh"], stream=False)
        except Exception as exc:
            raise DemoModelError(
                f"DeepSeek request failed: {type(exc).__name__}"
            ) from exc
    else:
        run_output = _simulate_run(case_data, kg_toolkit, decision_toolkit)

    validation = validate_run(run_output, context)
    return case_data, run_output, validation


def main(argv: Optional[List[str]] = None) -> int:
    """CLI entry point.  Exit codes: 0 pass, 2 config, 3 model, 4 governance."""
    parser = argparse.ArgumentParser(
        description="Agno 2.9 + Semantica investment committee demo",
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="run against the real DeepSeek API (requires DEEPSEEK_API_KEY)",
    )
    parser.add_argument(
        "--committee",
        action="store_true",
        help="run the multi-Agent investment committee Team instead of one Agent",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="print sanitized tool details and tracebacks",
    )
    args = parser.parse_args(argv)

    try:
        case_data, run_output, validation = execute_demo(
            debug=args.debug, live=args.live, committee=args.committee
        )
    except DemoConfigurationError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        if args.debug:
            import traceback

            traceback.print_exc()
        return 2
    except DemoModelError as exc:
        print(f"Model error: {exc}", file=sys.stderr)
        if args.debug:
            import traceback

            traceback.print_exc()
        return 3
    except DemoValidationError as exc:
        print(f"Validation error: {exc}", file=sys.stderr)
        if args.debug:
            import traceback

            traceback.print_exc()
        return 4

    render_execution(case_data, run_output, validation, stream=sys.stdout, debug=args.debug)
    if not validation.ok:
        return 4
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
