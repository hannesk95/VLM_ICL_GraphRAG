"""Case graph for graph-based exemplar retrieval (GraphRAG design 2).

Nodes are TRAIN cases; edges connect each case to its k nearest neighbours in embedding space
(symmetrised, weights exp(cos/tau)). For a query we seed a personalised PageRank (PPR) on the
query's nearest train cases and rank all train cases by their PPR mass. Compared with plain kNN,
PPR also surfaces cases that are only reachable through intermediate neighbours (2+ hops),
i.e. it respects the manifold / cluster structure of the pool.

The same PPR mass doubles as a non-VLM reference classifier ("label propagation"): the share of
PPR mass on grade-4 nodes is a probability-like score.
"""
from __future__ import annotations

import numpy as np


class CaseGraph:
    def __init__(self, pids, y, emb, k: int = 8, alpha: float = 0.8, tau_edge: float = 0.1):
        self.pids = list(pids)
        self.y = np.asarray(y, dtype=int)
        self.E = np.asarray(emb, dtype=np.float64)
        self.alpha = alpha
        n = len(self.pids)
        S = self.E @ self.E.T
        np.fill_diagonal(S, -np.inf)
        W = np.zeros((n, n))
        kk = min(k, n - 1)
        for i in range(n):
            nb = np.argpartition(-S[i], kk - 1)[:kk]
            W[i, nb] = np.exp(S[i, nb] / tau_edge)
        W = np.maximum(W, W.T)
        P = W / W.sum(1, keepdims=True)
        self.M = np.linalg.inv(np.eye(n) - alpha * P.T)
        self.W = W

    def ppr(self, q_emb, n_seeds: int = 10, tau_seed: float = 0.1) -> np.ndarray:
        sims = self.E @ np.asarray(q_emb, dtype=np.float64)
        seeds = np.argsort(-sims)[: min(n_seeds, len(sims))]
        p = np.zeros(len(sims))
        w = np.exp((sims[seeds] - sims[seeds].max()) / tau_seed)
        p[seeds] = w / w.sum()
        return (1 - self.alpha) * (self.M @ p)

    def retrieve(self, q_emb, k: int, balance: bool = False, n_seeds: int = 10) -> list[int]:
        score = self.ppr(q_emb, n_seeds)
        order = np.argsort(-score)
        if not balance:
            return [int(i) for i in order[:k]]
        want = {1: (k + 1) // 2, 0: k // 2}
        out = []
        for i in order:
            if want[int(self.y[i])] > 0:
                out.append(int(i))
                want[int(self.y[i])] -= 1
            if len(out) == k:
                break
        return out

    def prior(self, q_emb, n_seeds: int = 10) -> float:
        """Label-propagation score: PPR mass on grade-4 cases."""
        s = self.ppr(q_emb, n_seeds)
        return float(s[self.y == 1].sum() / max(s.sum(), 1e-12))
