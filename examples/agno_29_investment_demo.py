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
