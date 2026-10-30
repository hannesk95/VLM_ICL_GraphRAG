"""End-to-end pipeline test on synthetic data with the mock scorer (no GPU / model needed).

    pytest -q tests/
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from make_synthetic import make  # noqa: E402
from medvlm_bench import run as R  # noqa: E402
from medvlm_bench.data import index_dataset, norm_id  # noqa: E402
from medvlm_bench.evaluate import evaluate  # noqa: E402
from medvlm_bench.kg import KnowledgeGraph  # noqa: E402
from medvlm_bench.metrics import point_metrics  # noqa: E402


@pytest.fixture(scope="module")
def env(tmp_path_factory):
    root = make(tmp_path_factory.mktemp("syn"), n=60)
    out = tmp_path_factory.mktemp("out")
    df = index_dataset(root)
    return root, out, df


def test_ids_and_files():
    assert norm_id("UCSF-PDGM-004") == "UCSF-PDGM-0004"
    assert norm_id("UCSF-PDGM-0429_FU003d") is None  # follow-up metadata rows are dropped


def test_index_ignores_followups_partials_and_flags_incomplete(env):
    _, _, df = env
    assert len(df) == 60 and df.pid.is_unique
    assert set(df.split) == {"train", "val", "test"}
    assert (~df.complete).sum() == 6  # every 9th case misses FLAIR
    assert not df.path_T1c.isna().all()


def test_splits_are_independent_of_file_availability(env):
    root, _, df = env
    hidden = []
    for f in list((Path(root) / "MRI").glob("UCSF-PDGM-000[1-9]_T1c_bias.nii.gz")):
        h = f.with_name(f.name + ".hidden")
        f.rename(h); hidden.append((h, f))  # simulate a not-yet-copied file
    try:
        df2 = index_dataset(root)
        assert (df.set_index("pid").split == df2.set_index("pid").split).all()
        assert df2.complete.sum() < df.complete.sum()
    finally:
        for h, f in hidden:
            h.rename(f)


def test_kg_graph_and_rewire():
    kg = KnowledgeGraph.from_json()
    txt = kg.serialize(["central_necrosis", "ring_enhancement"], ["no_enhancement"])
    assert "Glioblastoma, IDH-wildtype" in txt and "WHO CNS grade 4" in txt
    rw = kg.rewired(0)
    assert len(rw.edges) <= len(kg.edges) and rw.serialize(["central_necrosis"]) != kg.serialize(["central_necrosis"])


def test_full_pipeline_mock(env):
    root, out, df = env
    from medvlm_bench.model import MockScorer
    ctx = R.Context(df, MockScorer(0), out)
    reg = R.build_registry()
    names = ["zs_v0", "icl_knn_k2", "icl_graph_k4", "icl_knn_k2_single", "rag_graph_seg", "rag_graph_vlm",
             "combo_graph_seg_icl_graph_k4", "icl_knn_k4_randlabels"]
    for split in ("val", "test"):
        for n in names:
            R.run_condition(reg[n], split, ctx, limit=8)
        R.run_references(ctx, split, limit=8)
    res, paired = evaluate(out, "zs_v0", n_boot=50)
    assert set(names) <= set(res.condition)
    assert (res.option_mass > 0.9).all()
    assert (out / "results.md").exists()

    # leakage: exemplars are train-only and never the query patient
    sp = dict(zip(df.pid, df.split))
    for f in (out / "predictions").glob("*/*.csv"):
        d = pd.read_csv(f).fillna("")
        for r in d.itertuples():
            for p in str(r.shot_pids).split("|"):
                if p:
                    assert sp[p] == "train" and p != r.pid


def test_metrics_basic():
    y = np.array([0, 0, 1, 1]); p = np.array([0.1, 0.4, 0.35, 0.8])
    m = point_metrics(y, p)
    assert abs(m["auroc"] - 0.75) < 1e-9
