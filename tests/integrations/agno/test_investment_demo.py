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
