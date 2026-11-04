# medvlm_bench – MedGemma 1.5 × in-context learning × GraphRAG on brain MRI

Benchmark code for **WHO CNS grade 4 vs. grade 2/3** classification on **UCSF-PDGM** with
`google/medgemma-1.5-4b-it`, comparing

| Block | What | Conditions (see `python -m medvlm_bench.run list`) |
|---|---|---|
| Baselines | zero-shot, 3 prompt wordings, image-representation ablations | `zs_v0/1/2`, `zs_single`, `zs_t1c_only`, `zs_multiimage`, `zs_uniform` |
| ICL | k ∈ {1,2,4,8}; exemplar selection random / balanced / kNN / kNN-balanced / MMR / **case-graph PPR**; findings-augmented shots; random-label and order controls; multi-turn vs single-turn | `icl_<selector>_k<k>`, `icl_knn_k4_findings`, `icl_knn_k4_randlabels`, `icl_knn_k4_orderrandom`, `icl_knn_k<k>_single` |
| GraphRAG (KG) | imaging findings → knowledge-graph sub-graph → evidence text in the prompt; comparators: flat TF-IDF RAG, whole KG in prompt, rewired (random) KG | `rag_{graph,vector,full,random_graph}_{seg,vlm}`, `rag_graph_both` |
| GraphRAG (cases) | personalised-PageRank retrieval on a kNN graph of training cases → ICL shots; combined with KG evidence | `icl_graph_k<k>`, `icl_graph_balanced_k<k>`, `combo_graph_{seg,vlm}_icl_graph_k4` |
| References (no VLM) | logistic regression, kNN, case-graph label propagation on the same case embedding | `ref_logreg`, `ref_knn5`, `ref_labelprop` |

## Quick start

```bash
pip install -r requirements.txt
huggingface-cli login            # accept the HAI-DEF terms for google/medgemma-1.5-4b-it (gated)

# 1) index the data + create the stable patient-level split (60/10/30 train/val/test)
python -m medvlm_bench.run index --root data/brain --out runs/exp1

# 2) zero-shot baselines on validation (for threshold selection) and test
for S in val test; do
  python -m medvlm_bench.run run --root data/brain --out runs/exp1 --backend medgemma --split $S \
      --conditions "zs_v*" "zs_single" "zs_multiimage"
done

# 3) ICL, GraphRAG, references
for S in val test; do
  python -m medvlm_bench.run run  --root data/brain --out runs/exp1 --split $S \
      --conditions "icl_*_k4" "icl_*_k8" "rag_*" "combo_*"
  python -m medvlm_bench.run refs --root data/brain --out runs/exp1 --split $S
done

# 4) tables: results_table.csv, paired_vs_ref.csv, results.md
python -m medvlm_bench.evaluate --out runs/exp1 --ref zs_v0
```
`--root` is the folder with `UCSF-PDGM-metadata_v5.xlsx` and `MRI/`. Runs are **resumable** (each case is
appended to its CSV; re-running skips finished cases). Use `--limit 10` for a dry run and `--backend mock`
to test without a GPU. `pytest -q tests/` runs the whole pipeline on synthetic data.
Conditions that depend on the case embedding (ICL with k > 0, references) are saved with the embedder
appended, e.g. `icl_knn_k4@handcrafted` or `ref_logreg@medsiglip` (`--embedder`), so runs with different embedders
can share one `--out` folder.

## Data handling (important while the copy is still running)

* `index` reports, per split, how many patients are **complete** (need `T1c`, `FLAIR`, `T2` and a segmentation by default;
  change with `--require T1c,FLAIR` or `--no-seg-required`). Incomplete cases are skipped, `*.filepart` files are ignored.
* **The split is computed on all 501 metadata patients, not on the files present**, so it will not change as more MRIs arrive
  (same `--split-seed` ⇒ same assignment). Re-run the same commands after the copy finished; finished cases are not recomputed
  (delete `cache/` if you change a rendering setting).
* Only baseline scans are used (follow-up `_FU…` files and metadata rows are ignored).
* Metadata IDs have 3 digits (`UCSF-PDGM-004`), file names 4 (`UCSF-PDGM-0004`); they are normalised.
* Label: `y = 1` ⇔ WHO CNS grade 4 (402/501 patients, ≈80 %). **Strong class imbalance** → report AUROC/AUPRC and balanced
  accuracy, not accuracy. Grade 4 is overwhelmingly IDH-wildtype GBM; that is a shortcut worth discussing in the paper.

## How the VLM is queried

* Input: one montage image (rows = T1c / FLAIR / T2, columns = 3 axial slices placed at cumulative-tumour-area quantiles),
  896×896 as MedGemma expects; or `layout="slices"` (one image per tile).
* **Forced choice A/B** – we read the next-token log-probabilities of the answer letters, giving continuous scores
  (AUROC, calibration). `option_mass` (probability mass on valid letters) is logged as a format-compliance check.
* **Counterbalancing**: every prediction is the mean over both option orders (A/B swapped); `order_sensitivity` in the results
  table quantifies the residual position bias.
* Exemplars come from the **train split only**; the code asserts the query is never in the pool; label-shuffled and
  random-order controls are included. Splits are patient-level, so no patient appears in both exemplar pool and test.

## GraphRAG design

1. **Knowledge graph** (`medvlm_bench/data/kg_seed.json`): imaging-feature nodes → diagnosis / grade nodes with relations
   (`suggests`, `argues_against`, `graded_as`, `defined_by`, `upgrades_to`) and strengths. It contains **no patient data**.
   Findings per case come from `seg` (thresholds on the segmentation, see `imaging.THRESH`), from the `vlm` itself (a yes/no
   probe per feature), or `both`. Retrieval = BFS (2 hops) from the present findings; the evidence is serialised with a
   net-evidence ranking of candidate diagnoses.
2. **Case graph** (`casegraph.py`): kNN graph over training cases (embedding = segmentation descriptors by default, or MedSigLIP
   via `--embedder medsiglip`); personalised PageRank from the query's nearest cases selects the ICL exemplars.

## Things you must decide / verify before publishing

1. **The seed KG is a draft.** Grading rules follow WHO CNS5, but the imaging associations are general knowledge flagged
   `needs_citation`; have a neuroradiologist/neuropathologist curate and cite every edge. `astro_idhwt_lg` mirrors the UCSF-PDGM
   label taxonomy (not a formal CNS5 type).
2. **Segmentation leakage of expertise.** `slice_mode="tumor"`, `seg`-based findings and the handcrafted embedding all use the
   (expert-corrected) tumour segmentation. This is realistic where tumours are delineated (e.g. radiotherapy), but report the
   segmentation-free variants too (`zs_uniform`, `rag_*_vlm`, `--embedder medsiglip`). The `ref_*` classifiers on segmentation
   descriptors will likely be strong – that is a legitimate comparison, not a bug.
3. **Thresholds** in `imaging.THRESH` (necrosis ≥15 % of the core, etc.) are a-priori heuristics; do not tune them on test.
4. **Data contamination**: UCSF-PDGM (and BraTS-2021 re-uses of it) is public; whether MedGemma saw it cannot be excluded.
   Consider an internal hold-out cohort as the primary test set.
5. **3D input.** `layout="slices"` approximates multi-slice input, but the model card says CT/MRI volumes "require some
   pre-processing" (see Google's `high_dimensional_ct_hugging_face` notebook) – align with that protocol if you want the
   official 3D pathway as a separate arm.
6. **Multi-turn ICL**: the model card notes MedGemma was not evaluated/optimised for multi-turn use, therefore single-turn ICL
   (`*_single`) is included as a factor.
7. Not yet implemented: chain-of-thought / rationale shots, LoRA fine-tuned upper reference, DeLong test (paired bootstrap
   ΔAUROC + exact McNemar are used), other tasks/datasets (add a `TaskSpec` in `prompts.py` and a loader in `data.py`).

## Verified vs. untested

Verified here on real UCSF-PDGM cases: loading/orientation, montage rendering, segmentation features. Verified on synthetic data
with a mock scorer: indexing, splits, all selectors, KG/graph retrieval, prompt construction, resume logic, evaluation, leakage
checks (`pytest`). **Not run in this environment:** the actual MedGemma and MedSigLIP back-ends (`model.py`, `embed.py:MedSigLIPEmbedder`)
– they follow the official model-card usage but need a first run on your GPU (try `--limit 5`).
