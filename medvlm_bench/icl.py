"""In-context-learning exemplar selection.

Strategies
----------
random        k random train cases
balanced      class-balanced random
knn           k most similar train cases (cosine in embedding space)
knn_balanced  top-k/2 most similar per class
mmr           maximal marginal relevance (relevance vs. diversity, lambda=0.7)
graph         personalised-PageRank retrieval on the case graph (see casegraph.py)
graph_balanced  same, class-balanced

Ordering of the shots in the prompt: ``similar_last`` (most similar shot closest to the query;
default), ``similar_first`` or ``random``.
"""
from __future__ import annotations

import numpy as np

from .casegraph import CaseGraph

STRATEGIES = ("random", "balanced", "knn", "knn_balanced", "mmr", "graph", "graph_balanced")


class ExemplarSelector:
    def __init__(self, pool_cases, pool_emb: np.ndarray, strategy: str = "random", seed: int = 0,
                 order: str = "similar_last", casegraph: CaseGraph | None = None):
        assert strategy in STRATEGIES, strategy
        self.pool, self.E = list(pool_cases), np.asarray(pool_emb)
        self.y = np.array([c.y for c in self.pool])
        self.strategy, self.seed, self.order = strategy, seed, order
        self.graph = casegraph
        if strategy.startswith("graph") and casegraph is None:
            raise ValueError("graph strategies need a CaseGraph")

    def _rng(self, pid: str):
        return np.random.default_rng([self.seed, int(pid[-4:])])

    def _balanced(self, ranking: np.ndarray, k: int) -> list[int]:
        want = {1: (k + 1) // 2, 0: k // 2}
        out = []
        for i in ranking:
            if want[int(self.y[i])] > 0:
                out.append(int(i)); want[int(self.y[i])] -= 1
            if len(out) == k:
                break
        return out

    def select(self, query_case, q_emb: np.ndarray, k: int) -> list:
        """Return k exemplar cases (train pool only; the query is never part of the pool)."""
        if k <= 0:
            return []
        assert all(c.pid != query_case.pid for c in self.pool), "query case found in exemplar pool (leakage)"
        k = min(k, len(self.pool))
        rng = self._rng(query_case.pid)
        sims = self.E @ q_emb
        s = self.strategy
        if s == "random":
            idx = list(rng.choice(len(self.pool), k, replace=False))
        elif s == "balanced":
            perm = rng.permutation(len(self.pool))
            idx = self._balanced(perm, k)
        elif s == "knn":
            idx = list(np.argsort(-sims)[:k])
        elif s == "knn_balanced":
            idx = self._balanced(np.argsort(-sims), k)
        elif s == "mmr":
            lam, chosen = 0.7, []
            cand = set(range(len(self.pool)))
            for _ in range(k):
                best, best_sc = None, -np.inf
                for i in cand:
                    red = max((float(self.E[i] @ self.E[j]) for j in chosen), default=0.0)
                    sc = lam * sims[i] - (1 - lam) * red
                    if sc > best_sc:
                        best, best_sc = i, sc
                chosen.append(best); cand.discard(best)
            idx = chosen
        else:  # graph / graph_balanced
            idx = self.graph.retrieve(q_emb, k, balance=s.endswith("balanced"))
        idx = [int(i) for i in idx]
        if self.order == "similar_last":
            idx = sorted(idx, key=lambda i: sims[i])
        elif self.order == "similar_first":
            idx = sorted(idx, key=lambda i: -sims[i])
        else:
            idx = [int(i) for i in rng.permutation(idx)]
        return [self.pool[i] for i in idx]
