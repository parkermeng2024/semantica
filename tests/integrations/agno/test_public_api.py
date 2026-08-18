"""
Tests for the public API surface of ``integrations.agno``.
"""

from __future__ import annotations

from integrations.agno import AGNO_TOOLKITS_AVAILABLE
from integrations.agno.decision_kit import AGNO_AVAILABLE as DECISION_KIT_AVAILABLE
from integrations.agno.kg_toolkit import AGNO_AVAILABLE as KG_TOOLKIT_AVAILABLE


def test_toolkit_availability_is_exported_separately():
    assert AGNO_TOOLKITS_AVAILABLE is (
        DECISION_KIT_AVAILABLE and KG_TOOLKIT_AVAILABLE
    )
