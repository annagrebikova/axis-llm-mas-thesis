"""DOT / Mermaid / TikZ renderers share graph semantics and escape content."""

from __future__ import annotations

import unittest

from mas_langgraph.tests.test_explanation_graph import LONG_CHUNK, SPECIAL, make_exported_case
from mas_langgraph.visualization import (
    build_explanation_graph,
    render_dot,
    render_mermaid,
    render_tikz,
)


class TestGraphRenderers(unittest.TestCase):
    def setUp(self):
        self.case = make_exported_case()
        self.graph = build_explanation_graph(self.case, "axis", "essential")

    def test_renderers_share_node_and_edge_ids(self):
        node_ids = set(self.graph.node_ids())
        edge_ids = set(self.graph.edge_ids())
        self.assertEqual(node_ids, {n.node_id for n in self.graph.nodes})
        self.assertEqual(edge_ids, {e.edge_id for e in self.graph.edges})
        dot = render_dot(self.graph)
        mermaid = render_mermaid(self.graph)
        tikz = render_tikz(self.graph)
        self.assertTrue(dot.startswith("digraph "))
        self.assertTrue(mermaid.startswith("flowchart LR"))
        self.assertIn(r"\begin{tikzpicture}", tikz)
        self.assertIn(r"\end{tikzpicture}", tikz)
        # Canonical ids are encoded, not dropped.
        self.assertGreater(dot.count("->"), 0)
        self.assertGreater(mermaid.count("-->") + mermaid.count("-.->"), 0)
        self.assertIn(r"\draw", tikz)

    def test_renderers_do_not_mutate_details(self):
        chunk = self.graph.node_map().get("artifact::E1:v1")
        if chunk is None:
            g = build_explanation_graph(self.case, "trace_provenance", "full")
            chunk = g.node_map()["artifact::E1:v1"]
            graph = g
        else:
            graph = self.graph
        original = chunk.details["content"]["content"]
        render_dot(graph)
        render_mermaid(graph)
        render_tikz(graph)
        self.assertEqual(
            graph.node_map()[chunk.node_id].details["content"]["content"],
            original,
        )
        self.assertEqual(original, LONG_CHUNK)

    def test_escaping_special_characters(self):
        g = build_explanation_graph(self.case, "trace_provenance", "full")
        chunk = g.node_map()["artifact::E1:v1"]
        self.assertIn('"', chunk.preview + chunk.details["content"]["content"])
        dot = render_dot(g)
        mermaid = render_mermaid(g)
        tikz = render_tikz(g)
        # DOT: quotes inside labels are escaped, not raw-injected.
        self.assertNotIn('\nnewline', dot.split("label=")[0])
        self.assertIn('\\"', dot)
        # Mermaid entity escaping keeps meaning without raw control chars.
        self.assertIn("#quot;", mermaid)
        self.assertIn("#91;", mermaid)
        self.assertIn("#123;", mermaid)
        self.assertIn("#124;", mermaid)
        self.assertNotIn('pipes} | underscores', mermaid)
        # TikZ escapes TeX specials.
        self.assertIn(r"\%", tikz)
        self.assertIn(r"\&", tikz)
        self.assertIn(r"\#", tikz)
        self.assertIn(r"\_", tikz)
        self.assertIn(r"\{", tikz)
        self.assertIn(r"\}", tikz)
        # Meaningful content is not deleted.
        self.assertIn("quotes", chunk.details["content"]["content"])
        self.assertIn("quotes", dot + mermaid + tikz)

    def test_previews_not_full_chunk_in_renderers(self):
        g = build_explanation_graph(self.case, "trace_provenance", "full")
        full = g.node_map()["artifact::E1:v1"].details["content"]["content"]
        preview = g.node_map()["artifact::E1:v1"].preview
        self.assertLess(len(preview), len(full))
        dot = render_dot(g)
        self.assertNotIn(full.strip(), dot)
        self.assertIn(preview.split()[0], dot)

    def test_tikz_positions_are_deterministic(self):
        a = render_tikz(self.graph)
        b = render_tikz(self.graph)
        self.assertEqual(a, b)
        self.assertIn(" at (", a)


if __name__ == "__main__":
    unittest.main()
