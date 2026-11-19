"""Experiment runner: zero-shot baselines, ICL, KG-RAG / case-graph GraphRAG, reference classifiers.

Usage (from the VLM_ICL_GraphRAG folder)
-----------------------------------------
  python -m medvlm_bench.run list
  python -m medvlm_bench.run index --root data/brain --out runs/exp1
  python -m medvlm_bench.run run   --root data/brain --out runs/exp1 --backend medgemma \
         --split test --conditions "zs_v*" "icl_knn_k4" "rag_graph_seg"
  python -m medvlm_bench.run refs  --root data/brain --out runs/exp1 --split test
  python -m medvlm_bench.evaluate  --out runs/exp1
"""
from __future__ import annotations

import argparse
import datetime
import fnmatch
import json
import os
import sys
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np
import pandas as pd

from .casegraph import CaseGraph
from .data import Case, cases_from_df, index_dataset, summarize
from .embed import HandcraftedEmbedder
from .icl import STRATEGIES, ExemplarSelector
from .data import pid_key
from .imaging import RenderConfig, compute_case_features, features_to_text, findings_from_features, render_case
from .kg import KnowledgeGraph, VectorRAG
from .prompts import GLIOMA_GRADE, TaskSpec, build_messages, n_images, yes_no_messages
from .promptpdf import messages_to_pdf
from .tasks import TASKS, adapt_condition, condition_supported, feature_api, get_task, infer_task

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    def tqdm(x, **k):
        return x


# ------------------------------------------------------------------------------- conditions
@dataclass(frozen=True)
class Condition:
    name: str
    k: int = 0
    selector: str = "random"
    shot_format: str = "label"  # label | findings (adds quantitative seg findings to every turn)
    order: str = "similar_last"
    shuffle_labels: bool = False  # control: exemplar labels replaced by random labels
    rag: str = "none"  # none | graph | vector | random_graph | full
    findings_source: str = "seg"  # seg | vlm | both  (used by rag != none)
    variant: int = 0
    render: RenderConfig = field(default_factory=RenderConfig)
    single_turn: bool = False  # ICL: all examples inside one user message instead of multi-turn chat
    counterbalance: bool = True  # average over both A/B option orders
    seeds: tuple = (0,)


def build_registry() -> dict[str, Condition]:
    R: dict[str, Condition] = {}

    def add(c: Condition):
        R[c.name] = c

    for v in (0, 1, 2):
        add(Condition(f"zs_v{v}", variant=v))
    add(Condition("zs_single", render=RenderConfig(seqs=("T1c",), n_slices=1)))
    add(Condition("zs_multiimage", render=RenderConfig(layout="slices")))
    add(Condition("zs_uniform", render=RenderConfig(slice_mode="uniform")))
    add(Condition("zs_t1c_only", render=RenderConfig(seqs=("T1c",), n_slices=3)))
    
    for sel in STRATEGIES:
        for k in (1, 2, 3, 4, 5, 8, 10):
            add(Condition(f"icl_{sel}_k{k}", k=k, selector=sel,
                          seeds=(0, 1, 2) if sel in ("random", "balanced") else (0,)))
            add(Condition(f"icl_{sel}_k{k}_multiimage", k=k, selector=sel,
                          seeds=(0, 1, 2) if sel in ("random", "balanced") else (0,), render=RenderConfig(layout="slices")))
    add(Condition("icl_knn_k4_findings", k=4, selector="knn", shot_format="findings"))
    
    for k in (2, 4, 8):
        add(Condition(f"icl_knn_k{k}_single", k=k, selector="knn", single_turn=True))
    add(Condition("icl_random_k4_single", k=4, selector="random", single_turn=True, seeds=(0, 1, 2)))
    add(Condition("icl_knn_k4_randlabels", k=4, selector="knn", shuffle_labels=True, seeds=(0, 1, 2)))
    add(Condition("icl_knn_k4_orderrandom", k=4, selector="knn", order="random", seeds=(0, 1, 2)))
    
    for rag in ("graph", "vector", "random_graph", "full"):
        for src in ("seg", "vlm"):
            add(Condition(f"rag_{rag}_{src}", rag=rag, findings_source=src))
    add(Condition("rag_graph_both", rag="graph", findings_source="both"))
    # add(Condition("combo_graph_seg_icl_graph_k4", k=4, selector="graph", rag="graph", findings_source="seg"))
    # add(Condition("combo_graph_vlm_icl_graph_k4", k=4, selector="graph", rag="graph", findings_source="vlm"))

    for k in (1, 2, 4, 8):
        add(Condition(f"combo_graph_seg_icl_graph_k{k}", k=k, selector="graph",
                    rag="graph", findings_source="seg"))
        add(Condition(f"combo_graph_vlm_icl_graph_k{k}", k=k, selector="graph",
                    rag="graph", findings_source="vlm"))

    for k in (1, 2, 4, 8):
        add(Condition(f"combo_graph_seg_icl_graph_k{k}_multiimage", k=k, selector="graph",
                    rag="graph", findings_source="seg",
                    render=RenderConfig(layout="slices")))

    return R


def select_conditions(patterns, registry) -> list[Condition]:
    out, seen = [], set()
    for p in patterns:
        hits = fnmatch.filter(registry.keys(), p)
        if not hits:
            raise SystemExit(f"No condition matches {p!r}. Use `list` to see all.")
        for h in hits:
            if h not in seen:
                seen.add(h); out.append(registry[h])
    return out


# ------------------------------------------------------------------------------- context
class Context:
    def __init__(self, df: pd.DataFrame, scorer, out_dir: Path, embedder=None, task: TaskSpec = GLIOMA_GRADE,
                 kg_path=None, seed: int = 0, cache_dir=None, task_name: str = "brain"):
        self.df, self.scorer, self.out, self.task = df, scorer, Path(out_dir), task
        self.task_name = task_name  # brain | head-neck | breast | liver | kidney | soft-tissue (see tasks.py)
        self._feat_fn, _, _, self._feat_text = feature_api(task_name)
        self.cache = Path(cache_dir) if cache_dir else self.out / "cache"  # fold-independent -> shared across folds
        self.train = cases_from_df(df, "train")
        if not self.train:
            raise SystemExit("No complete TRAIN cases available yet - copy more data or relax --require.")
        self.embedder = (embedder or HandcraftedEmbedder(self.cache / "features", task=task_name)).fit(self.train)
        self._emb: dict[str, np.ndarray] = {}
        E = self._embed(self.train)
        self.train_emb = E
        self.graph = CaseGraph([c.pid for c in self.train], [c.y for c in self.train], E) if len(self.train) > 8 else None
        if task_name == "brain":  # the knowledge graph (WHO CNS5) exists for brain only
            self.kg = KnowledgeGraph.from_json(kg_path) if kg_path else KnowledgeGraph.from_json()
            self.kg_rewired = self.kg.rewired(seed)
            self.vec = VectorRAG(self.kg)
        else:
            self.kg = self.kg_rewired = self.vec = None
        self._vlm_find: dict = {}

    def _embed(self, cases) -> np.ndarray:
        miss = [c for c in cases if c.pid not in self._emb]
        if miss:
            for c, e in zip(miss, self.embedder.embed(miss)):
                self._emb[c.pid] = e
        return np.stack([self._emb[c.pid] for c in cases])

    def emb(self, case) -> np.ndarray:
        return self._embed([case])[0]

    def images(self, case, cfg: RenderConfig):
        return render_case(case, cfg, self.cache / "renders")

    def features(self, case) -> dict:
        return self._feat_fn(case, self.cache / "features")

    def features_text(self, case) -> str:
        return self._feat_text(self.features(case))

    def seg_findings(self, case):
        return findings_from_features(compute_case_features(case, self.cache / "features"))

    def vlm_findings(self, case, cfg: RenderConfig) -> dict[str, float]:
        """P(feature present) per KG feature, from yes/no probes of the VLM itself (cached)."""
        f = self.cache / "vlm_findings" / f"{case.pid}_{cfg.key()}.json"
        if f.exists():
            return json.loads(f.read_text())
        imgs, layout = self.images(case, cfg), cfg.describe()
        out = {}
        for n in self.kg.feature_nodes():
            lp = self.scorer.option_logprobs(yes_no_messages(layout, imgs, n["question"]))
            e = np.exp(lp - lp.max())
            out[n["id"]] = float(e[0] / e.sum())
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(json.dumps(out))
        return out

    def evidence(self, cond: Condition, case) -> str | None:
        if cond.rag == "none":
            return None
        present, absent = [], []
        if cond.findings_source in ("seg", "both"):
            sf = self.seg_findings(case)
            present += [k for k, v in sf.items() if v]
            absent += [k for k, v in sf.items() if not v]
        if cond.findings_source in ("vlm", "both"):
            vf = self.vlm_findings(case, cond.render)
            present += [k for k, p in vf.items() if p >= 0.5]
            absent += [k for k, p in vf.items() if p < 0.5]
        present = list(dict.fromkeys(present))
        absent = [a for a in dict.fromkeys(absent) if a not in present]
        if cond.rag == "graph":
            return self.kg.serialize(present, absent)
        if cond.rag == "random_graph":
            return self.kg_rewired.serialize(present, absent)
        if cond.rag == "vector":
            return self.vec.serialize(present, absent)
        if cond.rag == "full":
            head = "Imaging findings present: " + ("; ".join(self.kg.label(f) for f in present) or "none of the listed findings") + "."
            return head + "\n" + self.kg.full_text()
        raise ValueError(cond.rag)


# ------------------------------------------------------------------------------- inference
def predict_case(cond: Condition, case: Case, seed: int, ctx: Context, selector: ExemplarSelector | None, prompt_dir: Path) -> dict:
    cfg = cond.render
    q_imgs = ctx.images(case, cfg)
    layout = cfg.describe()
    shots, shot_cases, shot_ys = [], [], []
    if cond.k > 0:
        q_emb = ctx.emb(case)
        shot_cases = selector.select(case, q_emb, cond.k)
        rng = np.random.default_rng([seed, pid_key(case.pid), 7])
        ys = [int(rng.integers(0, 2)) if cond.shuffle_labels else sc.y for sc in shot_cases]
        for sc, y in zip(shot_cases, ys):
            ftxt = ctx.features_text(sc) if cond.shot_format == "findings" else ""
            shots.append((ctx.images(sc, cfg), y, ftxt))
        shot_ys = ys
    q_ftxt = ctx.features_text(case) if cond.shot_format == "findings" else ""
    evidence = ctx.evidence(cond, case)

    ps, mass, nimg = [], [], 0
    for swap in ((False, True) if cond.counterbalance else (False,)):
        msgs = build_messages(ctx.task, layout, q_imgs, shots, evidence, q_ftxt, cond.variant, swap, cond.single_turn)
        messages_to_pdf(msgs, os.path.join(prompt_dir, f"{cond.name}_prompt_{case.pid}.pdf"), title=f"{cond.name} | {case.pid} (true label y={case.y})")
        lp = ctx.scorer.option_logprobs(msgs, ("A", "B"))
        e = np.exp(lp - lp.max())
        pA = float(e[0] / e.sum())
        ps.append(1 - pA if swap else pA)
        mass.append(float(np.exp(lp).sum()))
        nimg = n_images(msgs)
    return dict(
        p_pos=float(np.mean(ps)), p_pos_o0=ps[0], p_pos_o1=ps[1] if len(ps) > 1 else np.nan,
        option_mass=float(np.mean(mass)), n_images=nimg, n_shots=len(shots),
        shot_pids="|".join(sc.pid for sc in shot_cases), shot_ys="|".join(map(str, shot_ys)),
    )


def run_condition(cond: Condition, split: str, ctx: Context, limit: int | None = None, only_seed: int | None = None):
    queries = cases_from_df(ctx.df, split)
    if limit:
        queries = queries[:limit]
    name = f"{cond.name}@{ctx.embedder.name}" if cond.k > 0 else cond.name  # exemplar retrieval depends on the embedder
    for seed in cond.seeds if only_seed is None else (only_seed,):
        f = ctx.out / "predictions" / split / f"{name}__s{seed}.csv"
        f.parent.mkdir(parents=True, exist_ok=True)

        prompt_dir = ctx.out / "prompts"
        # make it a absolute path
        prompt_dir = prompt_dir.resolve()
        prompt_dir.mkdir(parents=True, exist_ok=True)

        done = set(pd.read_csv(f).pid) if f.exists() else set()
        selector = None
        if cond.k > 0:
            selector = ExemplarSelector(ctx.train, ctx.train_emb, cond.selector, seed, cond.order, ctx.graph)
        todo = [c for c in queries if c.pid not in done]
        for c in tqdm(todo, desc=f"{name} s{seed} [{split}]"):
            row = dict(condition=name, seed=seed, split=split, pid=c.pid, y=c.y)
            row.update(predict_case(cond, c, seed, ctx, selector, prompt_dir))
            pd.DataFrame([row]).to_csv(f, mode="a", header=not f.exists(), index=False)  # incremental / resumable


# ------------------------------------------------------------------------------- references
def run_references(ctx: Context, split: str, limit: int | None = None):
    """Non-VLM references on the same embedding: logistic regression, kNN, case-graph label propagation."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.neighbors import KNeighborsClassifier

    ytr = np.array([c.y for c in ctx.train])
    queries = cases_from_df(ctx.df, split)[: limit or None]
    Eq = ctx._embed(queries)
    preds = {}
    lr = LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000).fit(ctx.train_emb, ytr)
    preds["ref_logreg"] = lr.predict_proba(Eq)[:, 1]
    if len(set(ytr)) == 2:
        kn = KNeighborsClassifier(n_neighbors=min(5, len(ytr)), metric="cosine", weights="distance").fit(ctx.train_emb, ytr)
        preds["ref_knn5"] = kn.predict_proba(Eq)[:, 1]
    if ctx.graph is not None:
        preds["ref_labelprop"] = np.array([ctx.graph.prior(e) for e in Eq])
    for name, p in preds.items():
        name = f"{name}@{ctx.embedder.name}"
        df = pd.DataFrame(dict(condition=name, seed=0, split=split, pid=[c.pid for c in queries],
                               y=[c.y for c in queries], p_pos=p, p_pos_o0=p, p_pos_o1=np.nan,
                               option_mass=1.0, n_images=0, n_shots=0, shot_pids="", shot_ys=""))
        f = ctx.out / "predictions" / split / f"{name}__s0.csv"
        f.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(f, index=False)
        print("wrote", f)


# ------------------------------------------------------------------------------- CLI
def _make_df(args, task: str = "brain") -> pd.DataFrame:
    req = tuple(args.require.split(",")) if args.require else ("T1c", "FLAIR", "T2")
    return index_dataset(args.root, required_seqs=req, require_seg=not args.no_seg_required, seed=args.split_seed,
                         n_folds=args.n_folds, fold=args.fold, val_fraction=args.val_fraction, task=task)


def _make_scorer(args):
    if args.backend == "mock":
        from .model import MockScorer
        return MockScorer(args.split_seed)
    from .model import MedGemmaScorer
    return MedGemmaScorer(args.model_id)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="medvlm_bench.run")
    sub = ap.add_subparsers(dest="cmd", required=True)
    pl = sub.add_parser("list", help="list all registered conditions")
    pl.add_argument("--task", default=None, choices=list(TASKS), help="only the conditions that exist for this task")
    for name in ("index", "run", "refs"):
        p = sub.add_parser(name)
        p.add_argument("--root", required=True,
                       help="dataset folder: brain = UCSF-PDGM-metadata*.xlsx + MRI/; others = data/<task>")
        p.add_argument("--task", default=None, choices=list(TASKS),
                       help="brain | head-neck | breast | liver | kidney | soft-tissue (default: from the --root folder name)")
        p.add_argument("--out", required=True)
        p.add_argument("--require", default="", help="comma-separated sequences every case must have (default T1c,FLAIR,T2)")
        p.add_argument("--no-seg-required", action="store_true")
        p.add_argument("--split-seed", type=int, default=0)
        p.add_argument("--n-folds", type=int, default=None,
                       help="stratified K-fold CV instead of the single 60/10/30 split; use with --fold")
        p.add_argument("--fold", type=int, default=0, help="test fold (0..n_folds-1); results go to <out>/fold<k>")
        p.add_argument("--val-fraction", type=float, default=0.15,
                       help="CV only: share of the non-test patients used as validation (threshold selection)")
        if name != "index":
            p.add_argument("--split", default="test", choices=["train", "val", "test"])
            p.add_argument("--limit", type=int, default=None)
            p.add_argument("--backend", default="medgemma", choices=["medgemma", "mock"])
            p.add_argument("--model-id", default="google/medgemma-1.5-4b-it")
            p.add_argument("--embedder", default="medsiglip", choices=["handcrafted", "medsiglip"])
        if name == "run":
            p.add_argument("--conditions", nargs="+", required=True, help="names or fnmatch patterns")
            p.add_argument("--seed", type=int, default=None, help="run a single seed of stochastic conditions")
    a = ap.parse_args(argv)

    reg = build_registry()
    if a.cmd == "list":
        for n, c in reg.items():
            if a.task and not condition_supported(c, a.task):
                continue
            print(f"{n:38s} k={c.k} sel={c.selector if c.k else '-':14s} rag={c.rag:12s} seeds={list(c.seeds)}")
        return

    # set a.out to current date and time
    # import datetime
    # a.out = datetime.datetime.now().strftime("runs/%Y-%m-%d_%H-%M-%S")

    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    cache = out / "cache"  # renders / features / VLM findings do not depend on the fold -> shared
    if a.n_folds:
        out = out / f"fold{a.fold}"; out.mkdir(parents=True, exist_ok=True)
    task = a.task or infer_task(a.root)
    tdef = get_task(task)
    print(f"task: {task}  (positive class y=1: {tdef.positive})")
    df = _make_df(a, task)
    df.to_csv(out / "index.csv", index=False)
    print(summarize(df).to_string())
    if a.cmd == "index":
        return
    scorer = _make_scorer(a)
    emb = None
    if a.embedder == "medsiglip":
        from .embed import MedSigLIPEmbedder
        emb = MedSigLIPEmbedder(tdef.base_render() if task != "brain" else RenderConfig(), cache_dir=cache / "renders")
    ctx = Context(df, scorer, out, emb, task=tdef.prompt, seed=a.split_seed, cache_dir=cache, task_name=task)
    if a.cmd == "refs":
        run_references(ctx, a.split, a.limit)
        return
    conds = select_conditions(a.conditions, reg)
    skipped = [c.name for c in conds if not condition_supported(c, task)]
    if skipped:
        print(f"skipping {len(skipped)} condition(s) that do not exist for task '{task}' (KG / brain-only): "
              + ", ".join(skipped[:6]) + (" ..." if len(skipped) > 6 else ""))
    conds = [adapt_condition(c, task) for c in conds if condition_supported(c, task)]
    if not conds:
        raise SystemExit(f"No selected condition is available for task '{task}'.")
    for cond in conds:
        run_condition(cond, a.split, ctx, a.limit, a.seed)


if __name__ == "__main__":
    main()
