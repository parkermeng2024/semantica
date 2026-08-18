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
from typing import Any, Dict, List, Optional, TextIO, Tuple

from semantica.context import AgentContext, ContextGraph
from semantica.utils.logging import get_logger
from semantica.vector_store import VectorStore

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
    accepted_outcomes = {"rejected", "deferred_with_conditions"}

    errors: List[str] = []
    counts: Dict[str, int] = {}
    policy_result: Dict[str, Any] = {}
    decision_result: Dict[str, Any] = {}

    for tool in getattr(run_output, "tools", None) or []:
        name = getattr(tool, "tool_name", "") or ""
        counts[name] = counts.get(name, 0) + 1
        if getattr(tool, "tool_call_error", False):
            errors.append(f"tool call failed: {name}")
            continue
        try:
            result = json.loads(getattr(tool, "result", "") or "")
        except (TypeError, json.JSONDecodeError):
            errors.append(f"tool '{name}' returned invalid JSON")
            continue
        if not isinstance(result, dict):
            errors.append(f"tool '{name}' returned a non-object result")
            continue
        if result.get("error"):
            errors.append(f"tool '{name}' failed: {result['error']}")
            continue
        if name == "check_policy":
            policy_result = result
        elif name == "record_decision":
            decision_result = result

    for name, minimum in required_minimum.items():
        if counts.get(name, 0) < minimum:
            errors.append(f"missing required tool call: {name}")
    for name, exactly in required_exact.items():
        if counts.get(name, 0) != exactly:
            errors.append(
                f"{name} must be called exactly once; observed {counts.get(name, 0)}"
            )

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

    print("\n=== Agno 工具调用轨迹 ===", file=stream)
    for tool in getattr(run_output, "tools", None) or []:
        state = "ERROR" if getattr(tool, "tool_call_error", False) else "OK"
        print(f"[{state}] {tool.tool_name}", file=stream)
    if debug:
        print("\n=== Sanitized tool details ===", file=stream)
        for tool in getattr(run_output, "tools", None) or []:
            details = {
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
        "Call query_graph for Project Aurora before making any recommendation.",
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
# Orchestration and CLI
# ---------------------------------------------------------------------------
def execute_demo(
    debug: bool = False,
    live: bool = False,
) -> Tuple[Dict[str, Any], Any, ValidationResult]:
    """
    Seed the demo data, run (live DeepSeek or offline simulation), and
    validate the governed trace.  Returns the case data, the run output,
    and the validation result.
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
        "--debug",
        action="store_true",
        help="print sanitized tool details and tracebacks",
    )
    args = parser.parse_args(argv)

    try:
        case_data, run_output, validation = execute_demo(debug=args.debug, live=args.live)
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
