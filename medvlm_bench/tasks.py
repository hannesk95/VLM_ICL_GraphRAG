"""Task registry: one entry per dataset / classification task.

Everything that differs between the datasets lives here (and only here):

* ``load(root)``     -> case table (pid, y, grade, diagnosis, idh=stratification group, path_<seq>, path_seg, complete)
* ``prompt``         -> wording of the system prompt / question / answer options (``prompts.TaskSpec``)
* ``render``         -> how a case is turned into 2D images (``imaging.RenderConfig`` defaults of the task)
* ``kind``           -> "ct_hu" | "ct_u8" | "mri" (decides how mask-derived intensity features are expressed)
* ``kg``             -> True only for brain: the knowledge-graph conditions (rag_*, combo_*) exist for brain only

Brain is the original UCSF-PDGM task and goes through the unchanged code path in ``data.py`` / ``imaging.py``.
Positive class (y = 1) of every task is stated next to its loader and in ``TaskDef.positive``.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field, replace
from functools import partial
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from .prompts import GLIOMA_GRADE, TaskSpec


# --------------------------------------------------------------------------- helpers
def _find(root: Path, *patterns: str) -> Path:
    for pat in patterns:
        hits = sorted(root.glob(pat))
        if hits:
            return hits[-1]
    raise FileNotFoundError(f"None of {patterns} found in {root}")


def _clean_group(s) -> str:
    """Stratification group label. Must not contain '_' (data._strata splits on it)."""
    s = re.sub(r"[^A-Za-z0-9]+", "", str(s))
    return s or "all"


def _attach_files(meta: pd.DataFrame, root: Path, seq: str, img_fn: Callable[[str], Path], seg_fn: Callable[[str], Path],
                  what: str) -> pd.DataFrame:
    """Add path_<seq> / path_seg for every patient whose files exist and keep only those patients.

    Unlike the brain task (where the split is computed on the full metadata table because files were still being
    copied), the non-brain datasets are complete, so patients without image or mask are dropped *before* the split.
    """
    img = meta.pid.map(img_fn)
    seg = meta.pid.map(seg_fn)
    ok = np.array([a.exists() and b.exists() for a, b in zip(img, seg)])
    print(f"[{what}] {len(meta)} labelled patients in the clinical table, {int(ok.sum())} with image + mask "
          f"({int((~ok).sum())} dropped: file missing)")
    meta = meta[ok].copy().reset_index(drop=True)
    meta[f"path_{seq}"] = [str(p) for p in img[ok]]
    meta["path_seg"] = [str(p) for p in seg[ok]]
    meta["complete"] = True
    return meta


def _std_cols(df: pd.DataFrame, extra: tuple[str, ...] = ()) -> pd.DataFrame:
    cols = ["pid", "y", "grade", "diagnosis", "idh", *extra]
    return df[cols]


# --------------------------------------------------------------------------- loaders
def load_head_neck(root: Path) -> pd.DataFrame:
    """HNSCC of the oropharynx, planning CT. y = 1: HPV-positive, y = 0: HPV-negative.
    Strata group = source collection (OPC / HN / HNSCC file-name prefix)."""
    clin = pd.read_csv(_find(root, "clinical*.csv"))
    d = clin.rename(columns={"id": "pid"})
    d = d[d.hpv.astype(str).str.lower().isin(["positive", "negative"])].copy()
    d["y"] = (d.hpv.str.lower() == "positive").astype(int)
    d["grade"] = d["y"]
    d["diagnosis"] = d["histology"].astype(str) + " / HPV " + d["hpv"].astype(str)
    d["idh"] = d.pid.map(lambda p: _clean_group(re.match(r"[A-Za-z]+", p).group(0)))
    d = _std_cols(d.drop_duplicates("pid").reset_index(drop=True))
    ct = root / "CT"
    return _attach_files(d, root, "CT", lambda p: ct / f"{p}_ct.nii.gz", lambda p: ct / f"{p}_mask.nii.gz", "head-neck")


def load_breast(root: Path) -> pd.DataFrame:
    """Duke breast DCE-MRI, first post-contrast volume (<id>_0001). y = 1: Nottingham grade high,
    y = 0: intermediate / low. Patients without a recorded grade are excluded. Strata group = scanner manufacturer."""
    xl = _find(root, "clinical*.xlsx")
    d = pd.read_excel(xl, sheet_name="dataset_info")
    if "dataset" in d.columns:
        d = d[d.dataset.astype(str).str.upper() == "DUKE"]
    g = d.nottingham_grade.astype(str).str.lower().str.strip()
    d = d[g.isin(["high", "intermediate", "low"])].copy()
    g = g[d.index]
    d["pid"] = d.patient_id.astype(str)
    d["y"] = (g == "high").astype(int)
    d["grade"] = g.map({"low": 1, "intermediate": 2, "high": 3}).astype(int)
    d["diagnosis"] = d.tumor_subtype.astype(str)
    d["idh"] = d.manufacturer.map(_clean_group)
    d = _std_cols(d.drop_duplicates("pid").reset_index(drop=True))
    mri = root / "MRI"
    return _attach_files(d, root, "DCE", lambda p: mri / f"{p}_0001.nii.gz", lambda p: mri / f"{p}_segmentation.nii.gz",
                         "breast")


def load_liver(root: Path) -> pd.DataFrame:
    """Multi-phase liver CT, venous phase C2 only. y = 1: combined hepatocellular-cholangiocarcinoma (cHCC-CCA),
    y = 0: HCC. (ICC / CN rows of the clinical table are not part of this task.)"""
    d = pd.read_csv(_find(root, "clinical*.csv"))
    d = d[(d.phase == "C2") & d.cancer_type.isin(["HCC", "CHCC"])].copy()
    d["pid"] = d.patient_id.astype(str)
    d["y"] = (d.cancer_type == "CHCC").astype(int)
    d["grade"] = d["y"]
    d["diagnosis"] = d.cancer_type.astype(str)
    d["idh"] = "all"
    d = _std_cols(d.drop_duplicates("pid").reset_index(drop=True))
    ct = root / "CT"
    return _attach_files(d, root, "CT", lambda p: ct / f"{p}_ct_C2.nii.gz", lambda p: ct / f"{p}_mask_C2.nii.gz", "liver")


def load_kidney(root: Path) -> pd.DataFrame:
    """KiTS-style renal masses, arterial-phase CT. y = 1: high WHO/ISUP grade (3/4), y = 0: low grade (1/2).
    Masses without an ISUP grade (benign, unclassified) are excluded. Strata group = clear-cell vs other subtype.
    Mask = <id>_segmentation_tumor (binary tumour; the plain <id>_segmentation also contains the kidney)."""
    d = pd.read_csv(_find(root, "clinical*.csv"))
    d = d[d.tumor_isup_grade.notna()].copy()
    d["pid"] = d.patient_id.astype(str)
    d["grade"] = d.tumor_isup_grade.astype(int)
    d["y"] = (d.grade >= 3).astype(int)
    d["diagnosis"] = d.tumor_histologic_subtype.astype(str)
    d["idh"] = np.where(d.diagnosis == "clear_cell_rcc", "ccRCC", "other")
    d = _std_cols(d.drop_duplicates("pid").reset_index(drop=True))
    ct = root / "CT"
    return _attach_files(d, root, "CT", lambda p: ct / f"{p}_arterial.nii.gz",
                         lambda p: ct / f"{p}_segmentation_tumor.nii.gz", "kidney")


def load_soft_tissue(root: Path) -> pd.DataFrame:
    """Extremity / trunk sarcoma, fat-suppressed T2 (STIR) MRI. y = 1: high grade (G2/G3), y = 0: low grade (G1).
    Strata group = cohort, taken from the ID prefix ('Sar...' vs 'S...' = the two source cohorts)."""
    d = pd.read_csv(_find(root, "patient_metadata*.csv", "clinical*.csv"))
    d = d[d.Grading.isin([1, 2, 3])].copy()
    d["pid"] = d.ID.astype(str)
    d["grade"] = d.Grading.astype(int)
    d["y"] = (d.grade >= 2).astype(int)
    d["diagnosis"] = d.Histology.astype(str)
    d["idh"] = d.pid.map(lambda p: "Sar" if p.startswith("Sar") else "S")
    d = _std_cols(d.drop_duplicates("pid").reset_index(drop=True))
    mri = root / "MRI"
    return _attach_files(d, root, "T2FS", lambda p: mri / f"{p}STIR.nii.gz", lambda p: mri / f"{p}STIR-label.nii.gz",
                         "soft-tissue")


# --------------------------------------------------------------------------- task definitions
@dataclass(frozen=True)
class TaskDef:
    name: str
    prompt: TaskSpec
    kind: str  # ct_hu | ct_u8 | mri
    seq: str  # key of the image in Case.paths and in RenderConfig.seqs
    noun: str  # how the slices are described to the model
    positive: str  # documentation of y = 1
    load: Callable | None = None
    render: dict = field(default_factory=dict)  # RenderConfig kwargs
    kg: bool = False  # knowledge-graph conditions available?

    def base_render(self):
        from .imaging import RenderConfig
        return RenderConfig(task=self.name, seqs=(self.seq,), n_slices=9, slice_mode="tumor", layout="montage",
                            **self.render)


CT_SOFT = (-160.0, 240.0)  # soft-tissue window, level 40 / width 400 HU

TASKS: dict[str, TaskDef] = {
    "brain": TaskDef(
        name="brain", prompt=GLIOMA_GRADE, kind="mri", seq="T1c", noun="", positive="WHO CNS grade 4", kg=True),
    "head-neck": TaskDef(
        name="head-neck", kind="ct_hu", seq="CT", positive="HPV-positive", load=load_head_neck,
        noun="axial CT slices (soft-tissue window)",
        render=dict(plane="axial", fov=(100, 180), window=CT_SOFT),
        prompt=TaskSpec(
            name="hnscc_hpv", pos_text="HPV-positive oropharyngeal squamous cell carcinoma",
            neg_text="HPV-negative oropharyngeal squamous cell carcinoma",
            subject="CT scan of an adult patient with a histologically confirmed squamous cell carcinoma of the oropharynx",
            expert="head and neck radiologist", target="the HPV status of the oropharyngeal tumour",
            question="What is the HPV status of this oropharyngeal tumour?", question_short="Classify the HPV status.",
            v2_role="a careful head and neck oncology imaging assistant",
            v2_hint=("Consider the primary tumour site and size, its margins, necrosis or cystic change and the "
                     "appearance of any cervical lymph nodes before deciding."))),
    "breast": TaskDef(
        name="breast", kind="mri", seq="DCE", positive="Nottingham grade high", load=load_breast,
        noun="axial fat-suppressed T1-weighted dynamic contrast-enhanced breast MRI slices (first post-contrast phase)",
        render=dict(plane="axial", fov=(70, 140)),
        prompt=TaskSpec(
            name="breast_grade", pos_text="Nottingham grade 3 (high grade)",
            neg_text="Nottingham grade 1 or 2 (low or intermediate grade)",
            subject=("dynamic contrast-enhanced breast MRI (first post-contrast phase, fat-suppressed T1-weighted) "
                     "of an adult patient with biopsy-proven invasive breast cancer"),
            expert="breast radiologist", target="the Nottingham histologic grade of the tumour",
            question="What is the Nottingham grade of this breast tumour?", question_short="Classify the breast tumour grade.",
            v2_role="a careful breast imaging assistant",
            v2_hint=("Consider lesion shape, margin, size, internal enhancement pattern and surrounding "
                     "parenchymal enhancement before deciding."))),
    "liver": TaskDef(
        name="liver", kind="ct_u8", seq="CT", positive="cHCC-CCA", load=load_liver,
        noun="axial venous-phase contrast-enhanced CT slices of the liver",
        render=dict(plane="axial", fov=(120, 300)),
        prompt=TaskSpec(
            name="liver_type", pos_text="combined hepatocellular-cholangiocarcinoma (cHCC-CCA)",
            neg_text="hepatocellular carcinoma (HCC)",
            subject="venous-phase contrast-enhanced CT of the liver of an adult patient with a primary liver carcinoma",
            expert="abdominal radiologist", target="the histological type of the liver tumour",
            question="What is the histological type of this liver tumour?", question_short="Classify the liver tumour.",
            v2_role="a careful abdominal oncology imaging assistant",
            v2_hint=("Consider tumour margins, enhancement pattern, internal heterogeneity or necrosis, "
                     "capsule, hepatic contour changes and vascular involvement before deciding."))),
    "kidney": TaskDef(
        name="kidney", kind="ct_hu", seq="CT", positive="WHO/ISUP grade 3-4", load=load_kidney,
        noun="axial arterial-phase contrast-enhanced CT slices (soft-tissue window)",
        render=dict(plane="axial", fov=(100, 200), window=CT_SOFT),
        prompt=TaskSpec(
            name="kidney_grade", pos_text="WHO/ISUP grade 3 or 4 (high grade)",
            neg_text="WHO/ISUP grade 1 or 2 (low grade)",
            subject="arterial-phase contrast-enhanced CT of an adult patient with a renal cell carcinoma",
            expert="uroradiologist", target="the WHO/ISUP nuclear grade of the renal tumour",
            question="What is the WHO/ISUP grade of this renal tumour?", question_short="Classify the renal tumour grade.",
            v2_role="a careful uro-oncology imaging assistant",
            v2_hint=("Consider tumour size, margins, enhancement pattern, necrosis and extension beyond the "
                     "renal capsule or into the renal vein before deciding."))),
    "soft-tissue": TaskDef(
        name="soft-tissue", kind="mri", seq="T2FS", positive="FNCLCC grade 2-3", load=load_soft_tissue,
        noun="fat-suppressed T2-weighted MRI slices in the original acquisition plane",
        render=dict(plane="native", fov=(80, 300)),
        prompt=TaskSpec(
            name="sarcoma_grade", pos_text="high grade (grade 2 or 3)", neg_text="low grade (grade 1)",
            subject=("fat-suppressed T2-weighted MRI of an adult patient with a soft-tissue sarcoma of the "
                     "extremity or trunk"),
            expert="musculoskeletal radiologist", target="the histological grade of the sarcoma",
            question="What is the histological grade of this soft-tissue sarcoma?", question_short="Classify the sarcoma grade.",
            v2_role="a careful musculoskeletal oncology imaging assistant",
            v2_hint=("Consider tumour size, margins, signal heterogeneity, necrosis, peritumoral oedema and "
                     "infiltration of adjacent structures before deciding."))),
}
NON_BRAIN = tuple(n for n in TASKS if n != "brain")


def get_task(name: str) -> TaskDef:
    try:
        return TASKS[name]
    except KeyError:
        raise SystemExit(f"Unknown task {name!r}. Choose from: {', '.join(TASKS)}") from None


def infer_task(root) -> str:
    """Task from the data folder name (data/head-neck -> 'head-neck'); a folder with the UCSF-PDGM xlsx -> brain."""
    root = Path(root)
    n = root.name.lower()
    if n in TASKS:
        return n
    if any(root.glob("UCSF-PDGM-metadata*.xlsx")):
        return "brain"
    raise SystemExit(f"Cannot infer the task from --root {root}. Pass --task {{{','.join(TASKS)}}}.")


# --------------------------------------------------------------------------- layout text
def describe_layout(cfg) -> str:
    """Text that tells the model how the image(s) are laid out (used inside the system prompt)."""
    from .generic import GRID_COLS, order_text
    t = get_task(cfg.task)
    n = cfg.n_slices * len(cfg.seqs)
    if t.render.get("plane") == "native":
        order = "in order of slice position"
    else:
        order = "in order from the patient's inferior to superior side"
    where = "cropped around the tumour (radiological convention: patient right on the image left)"
    if cfg.layout == "montage":
        if n == 1:
            return f"The image shows one of the {t.noun}, {where}."
        cols = min(GRID_COLS, n)
        rows = -(-n // cols)
        return (f"Each image is a montage of {n} {t.noun}, {where}. The tiles form a {rows}x{cols} grid that is read "
                f"left to right, then top to bottom, {order}.")
    if n == 1:
        return f"The image shows one of the {t.noun}, {where}."
    return f"Each case is given as a series of {n} {t.noun}, {where}, {order}."


# --------------------------------------------------------------------------- features / conditions
def feature_api(task: str):
    """(feature_fn(case, cache_dir) -> dict, keys, log_keys, to_text(dict) -> str) for a task."""
    if task == "brain":
        from .embed import _LOG_KEYS
        from .imaging import FEATURE_KEYS, compute_case_features, features_to_text
        return compute_case_features, FEATURE_KEYS, _LOG_KEYS, features_to_text
    from .generic import GENERIC_KEYS, GENERIC_LOG_KEYS, compute_generic_features, generic_features_to_text
    t = get_task(task)
    return (partial(compute_generic_features, kind=t.kind, seq=t.seq), GENERIC_KEYS, GENERIC_LOG_KEYS,
            partial(generic_features_to_text, kind=t.kind))


def condition_supported(cond, task: str) -> bool:
    """KG conditions (rag_*, combo_*) and the brain-only T1c ablation exist for brain only."""
    if task == "brain":
        return True
    return cond.rag == "none" and cond.name != "zs_t1c_only"


def adapt_condition(cond, task: str):
    """Map a condition (defined for brain) onto another task: the condition's *deviations* from the default brain
    render config (layout, slice mode, number of slices, tile size) are applied to the task's own base config.
    Sequence choices are brain-specific and dropped (e.g. zs_single keeps its '1 slice')."""
    if task == "brain":
        return cond
    from .imaging import RenderConfig
    d, r = RenderConfig(), cond.render
    upd = {k: getattr(r, k) for k in ("layout", "slice_mode", "n_slices", "tile", "labels") if getattr(r, k) != getattr(d, k)}
    return replace(cond, render=replace(get_task(task).base_render(), **upd))
