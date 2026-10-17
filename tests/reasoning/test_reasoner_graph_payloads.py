"""Knowledge-graph payloads are expanded into reasoner facts."""

import unittest

from semantica.context import ContextGraph
from semantica.reasoning.reasoner import Reasoner


RULE = "IF founded_by(?x, ?y) THEN has_founder(?x, ?y)"


class TestReasonerGraphPayloads(unittest.TestCase):
    def test_graph_builder_payload_becomes_facts(self):
        kg = {
            "entities": [
                {"id": "Apple Inc.", "name": "Apple Inc.", "type": "ORG"},
                {"id": "Steve Jobs", "name": "Steve Jobs", "type": "PERSON"},
            ],
            "relationships": [
                {"source": "Apple Inc.", "target": "Steve Jobs", "type": "founded_by"},
            ],
            "metadata": {},
        }
        reasoner = Reasoner()
        reasoner.add_rule(RULE)

        inferred = reasoner.infer_facts(kg)

        self.assertIn("ORG(Apple Inc.)", reasoner.facts)
        self.assertIn("founded_by(Apple Inc., Steve Jobs)", reasoner.facts)
        self.assertEqual(inferred, ["has_founder(Apple Inc., Steve Jobs)"])

    def test_entities_only_payload_is_accepted(self):
        reasoner = Reasoner()
        reasoner.add_fact({"entities": [{"id": "Ada", "type": "PERSON"}]})
        self.assertEqual(reasoner.facts, {"PERSON(Ada)"})

    def test_context_graph_object_is_accepted(self):
        graph = ContextGraph(extract_entities=False, extract_relationships=False)
        graph.add_node(node_id="alice", node_type="Person", content="A")
        graph.add_node(node_id="bob", node_type="Person", content="B")
        graph.add_edge(source_id="alice", target_id="bob", edge_type="parent")

        reasoner = Reasoner()
        reasoner.add_fact(graph)

        self.assertEqual(
            reasoner.facts,
            {"Person(alice)", "Person(bob)", "parent(alice, bob)"},
        )

    def test_context_graph_dict_is_accepted(self):
        payload = {
            "nodes": {"n1": {"node_id": "n1", "node_type": "Host"}},
            "edges": [
                {"source_id": "n1", "target_id": "n2", "edge_type": "talks_to"},
            ],
        }
        reasoner = Reasoner()
        reasoner.add_fact(payload)

        self.assertEqual(reasoner.facts, {"Host(n1)", "talks_to(n1, n2)"})

    def test_single_records_still_work(self):
        reasoner = Reasoner()
        reasoner.add_fact("Person(ada)")
        reasoner.add_fact({"type": "Manager", "name": "Alice"})
        reasoner.add_fact({"source_name": "alice", "target_name": "bob", "type": "parent"})
        reasoner.add_fact({"source_id": "bob", "target_id": "charlie", "type": "parent"})

        self.assertEqual(
            reasoner.facts,
            {"Person(ada)", "Manager(Alice)", "parent(alice, bob)", "parent(bob, charlie)"},
        )

    def test_unrecognised_payload_is_ignored(self):
        reasoner = Reasoner()
        reasoner.add_fact({"unrelated": "value"})
        reasoner.add_fact(42)
        self.assertEqual(reasoner.facts, set())


if __name__ == "__main__":
    unittest.main()
