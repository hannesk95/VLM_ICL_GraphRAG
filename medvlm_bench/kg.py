"""Knowledge-graph RAG (GraphRAG design 1) and its comparators.

Pipeline for one case
---------------------
1. *Findings*: which imaging features of the KG vocabulary are present? Either derived from the
   segmentation (``findings_from_features``), predicted by the VLM itself (yes/no per feature,
   see ``run.extract_findings_vlm``), or both.
2. *Linking*: present findings map 1:1 onto KG ``feature`` nodes.
3. *Retrieval*: BFS from the linked nodes (``hops`` hops along out-edges) -> sub-graph.
4. *Serialisation*: edges as short evidence sentences + a ranking of candidate diagnoses by net
   supporting evidence (strong=3, moderate=2, weak=1) + diagnosis notes -> text in the prompt.

Comparators (same knowledge, different retrieval)
-------------------------------------------------
``vector``       TF-IDF top-n over flat edge sentences (no graph structure)
``full``         the whole KG dumped in the prompt (no retrieval)
``random_graph`` degree-preserving rewiring of edge targets (negative control)
"""
from __future__ import annotations

import json
from pathlib import Path

import networkx as nx
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

W = {"strong": 3, "moderate": 2, "weak": 1}
REL_TEXT = {
    "suggests": "suggests", "argues_against": "argues against", "graded_as": "can be graded as",
    "defined_by": "is defined by", "upgrades_to": "upgrades the tumour to",
}
DEFAULT_KG = Path(__file__).parent / "data" / "kg_seed.json"


class KnowledgeGraph:
    def __init__(self, nodes: list[dict], edges: list[dict]):
        self.nodes = {n["id"]: n for n in nodes}
        self.edges = list(edges)
        self.g = nx.MultiDiGraph()
        for n in nodes:
            self.g.add_node(n["id"], **n)
        for e in edges:
            self.g.add_edge(e["src"], e["dst"], **e)

    @classmethod
    def from_json(cls, path=DEFAULT_KG):
        d = json.loads(Path(path).read_text())
        return cls(d["nodes"], d["edges"])

    # ---- vocabulary
    def feature_nodes(self) -> list[dict]:
        return [n for n in self.nodes.values() if n["type"] == "feature"]

    def label(self, nid: str) -> str:
        return self.nodes[nid]["label"]

    # ---- sentences
    def edge_sentence(self, e: dict) -> str:
        s = f"{self.label(e['src'])} {REL_TEXT[e['rel']]} {self.label(e['dst'])}"
        extra = [f"{e['strength']}"] + ([e["note"]] if e.get("note") else [])
        return f"{s} ({'; '.join(extra)})"

    # ---- retrieval
    def subgraph_edges(self, feature_ids: list[str], hops: int = 2) -> list[dict]:
        seen_edges: list[dict] = []
        keyset = set()
        frontier = list(feature_ids)
        visited = set(feature_ids)
        for _ in range(hops):
            nxt = []
            for u in frontier:
                for _, v, d in self.g.out_edges(u, data=True):
                    k = (d["src"], d["rel"], d["dst"])
                    if k not in keyset:
                        keyset.add(k); seen_edges.append(d)
                    if v not in visited:
                        visited.add(v); nxt.append(v)
            frontier = nxt
        return seen_edges

    def rank_candidates(self, feature_ids: list[str]) -> list[tuple[str, int]]:
        """Net evidence per diagnosis node from direct feature->diagnosis edges."""
        sc: dict[str, int] = {}
        for f in feature_ids:
            for _, v, d in self.g.out_edges(f, data=True):
                if self.nodes[v]["type"] != "diagnosis":
                    continue
                s = W[d["strength"]] * (1 if d["rel"] == "suggests" else -1 if d["rel"] == "argues_against" else 0)
                sc[v] = sc.get(v, 0) + s
        return sorted(sc.items(), key=lambda t: -t[1])

    def serialize(self, present: list[str], absent: list[str] | None = None, hops: int = 2,
                  include_ranking: bool = True) -> str:
        lines = []
        lines.append("Imaging findings present: " + ("; ".join(self.label(f) for f in present) or "none of the listed findings") + ".")
        if absent:
            lines.append("Imaging findings not observed: " + "; ".join(self.label(f) for f in absent) + ".")
        edges = self.subgraph_edges(present, hops) if present else []
        # always give the grade rules of the diagnoses that are in the sub-graph
        if edges:
            lines.append("Relevant knowledge-graph evidence:")
            for e in edges:
                lines.append("- " + self.edge_sentence(e))
        if include_ranking and present:
            ranked = self.rank_candidates(present)
            if ranked:
                lines.append("Candidate diagnoses ranked by net supporting evidence: " + "; ".join(
                    f"{self.label(n)} ({s:+d})" for n, s in ranked) + ".")
        dx_in_graph = {e["dst"] for e in edges if self.nodes[e["dst"]]["type"] == "diagnosis"}
        for n in dx_in_graph:
            if self.nodes[n].get("description"):
                lines.append(f"Note on {self.label(n)}: {self.nodes[n]['description']}")
        lines.append("This is general background knowledge, not specific to this patient; imaging findings are not definitive.")
        return "\n".join(lines)

    # ---- comparators
    def edge_sentences(self) -> list[str]:
        return [self.edge_sentence(e) for e in self.edges]

    def full_text(self) -> str:
        return "Background knowledge (complete):\n" + "\n".join("- " + s for s in self.edge_sentences())

    def rewired(self, seed: int = 0) -> "KnowledgeGraph":
        """Negative control: permute edge targets within each relation type (keeps degree sequence per rel)."""
        rng = np.random.default_rng(seed)
        new = [dict(e) for e in self.edges]
        for rel in {e["rel"] for e in new}:
            idx = [i for i, e in enumerate(new) if e["rel"] == rel]
            dst = [new[i]["dst"] for i in idx]
            perm = rng.permutation(len(dst))
            for i, p in zip(idx, perm):
                new[i]["dst"] = dst[p]
        new = [e for e in new if e["src"] != e["dst"]]
        return KnowledgeGraph(list(self.nodes.values()), new)


class VectorRAG:
    """Flat TF-IDF retrieval over the KG's edge sentences (same knowledge, no graph structure)."""

    def __init__(self, kg: KnowledgeGraph, top_n: int = 8):
        self.kg, self.top_n = kg, top_n
        self.sent = kg.edge_sentences()
        self.vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True).fit(self.sent)
        self.M = self.vec.transform(self.sent)

    def serialize(self, present: list[str], absent: list[str] | None = None) -> str:
        head = "Imaging findings present: " + ("; ".join(self.kg.label(f) for f in present) or "none of the listed findings") + "."
        if not present:
            return head
        q = self.vec.transform([" ".join(self.kg.label(f) for f in present)])
        sims = (self.M @ q.T).toarray().ravel()
        order = [i for i in np.argsort(-sims)[: self.top_n] if sims[i] > 0]
        body = "\n".join("- " + self.sent[i] for i in order)
        return head + "\nRetrieved background knowledge:\n" + body
