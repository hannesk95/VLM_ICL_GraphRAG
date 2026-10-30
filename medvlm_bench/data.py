"""Dataset indexing, label construction and stable patient-level splits for UCSF-PDGM.

Design notes
------------
* Only *baseline* scans are used. Follow-up files (``..._FU001d_...``) are ignored, so a
  patient can never appear in two splits through a follow-up scan.
* Splits are computed on the **full metadata table (all patients)**, not on the files that
  happen to be present. The split therefore does not change while the data is still being
  copied; cases whose files are missing are simply reported as ``complete == False``.
* ``*.filepart`` / partially copied files never match the file patterns and are skipped.
* Label: ``y = 1`` for WHO CNS grade 4, ``y = 0`` for grade 2/3.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

SEQS = ("T1", "T1c", "T2", "FLAIR")
_IMG_RE = re.compile(r"^(UCSF-PDGM-\d{4})_(T1c|T1|T2|FLAIR)_bias\.nii\.gz$")
_SEG_RE = re.compile(r"^(UCSF-PDGM-\d{4})_tumor_segmentation(_merged)?\.nii\.gz$")
_ID_RE = re.compile(r"^UCSF-PDGM-(\d+)$")

COLMAP = {
    "ID": "pid_raw",
    "Sex": "sex",
    "Age at MRI": "age",
    "WHO CNS Grade": "grade",
    "Final pathologic diagnosis (WHO 2021)": "diagnosis",
    "MGMT status": "mgmt",
    "1p/19q": "codel_1p19q",
    "IDH": "idh_raw",
}


def norm_id(raw) -> str | None:
    """'UCSF-PDGM-004' -> 'UCSF-PDGM-0004'. Returns None for follow-up / malformed IDs."""
    m = _ID_RE.match(str(raw).strip())
    return None if m is None else f"UCSF-PDGM-{int(m.group(1)):04d}"


@dataclass
class Case:
    pid: str
    y: int
    split: str
    grade: int
    diagnosis: str
    idh: str
    paths: dict = field(default_factory=dict)

    def path(self, key: str):
        return self.paths.get(key)


def load_metadata(xlsx: str | Path) -> tuple[pd.DataFrame, list[str]]:
    df = pd.read_excel(xlsx).rename(columns=COLMAP)
    df["pid"] = df["pid_raw"].map(norm_id)
    dropped = df.loc[df.pid.isna(), "pid_raw"].astype(str).tolist()
    df = df[df.pid.notna()].copy()
    if not df.pid.is_unique:
        raise ValueError("Duplicate baseline patient IDs in metadata")
    df["grade"] = df["grade"].astype(int)
    df["y"] = (df["grade"] == 4).astype(int)
    df["idh"] = np.where(df["idh_raw"].astype(str).str.lower().str.contains("wild"), "wildtype", "mutant")
    cols = ["pid", "y", "grade", "diagnosis", "idh", "mgmt", "codel_1p19q", "age", "sex"]
    return df[cols].reset_index(drop=True), dropped


def scan_mri_dir(mri_dir: str | Path) -> tuple[dict, list[str]]:
    """Return {pid: {'T1c': path, ..., 'seg': path, 'seg_merged': path}} and ignored file names."""
    found: dict[str, dict] = {}
    ignored: list[str] = []
    for p in sorted(Path(mri_dir).iterdir()):
        n = p.name
        m = _IMG_RE.match(n)
        if m:
            found.setdefault(m.group(1), {})[m.group(2)] = str(p)
            continue
        m = _SEG_RE.match(n)
        if m:
            found.setdefault(m.group(1), {})["seg_merged" if m.group(2) else "seg"] = str(p)
            continue
        ignored.append(n)
    return found, ignored


def assign_splits(meta: pd.DataFrame, seed: int = 0, fractions=(0.6, 0.1, 0.3)) -> pd.Series:
    """Stratified (grade-4 x IDH status) patient-level train/val/test split on ALL patients."""
    tr, va, te = fractions
    assert abs(tr + va + te - 1) < 1e-6
    strat = meta["y"].astype(str) + "_" + meta["idh"]
    # tiny strata (<10 patients) are merged into the largest stratum of the same grade group
    vc = strat.value_counts()
    for s in vc.index[vc < 10]:
        same_y = [t for t in vc.index if t.split("_")[0] == s.split("_")[0] and vc[t] >= 10]
        if same_y:
            strat = strat.replace(s, same_y[0])
        else:
            strat = strat.replace(s, s.split("_")[0] + "_all")
    idx = np.arange(len(meta))
    idx_trval, idx_te = train_test_split(idx, test_size=te, stratify=strat, random_state=seed)
    idx_tr, idx_va = train_test_split(
        idx_trval, test_size=va / (tr + va), stratify=strat.iloc[idx_trval], random_state=seed
    )
    s = pd.Series("", index=meta.index)
    s.iloc[idx_tr], s.iloc[idx_va], s.iloc[idx_te] = "train", "val", "test"
    return s


def index_dataset(
    root: str | Path,
    required_seqs=("T1c", "FLAIR", "T2"),
    require_seg: bool = True,
    seed: int = 0,
    fractions=(0.6, 0.1, 0.3),
) -> pd.DataFrame:
    """Build the case table. ``root`` is the folder containing the metadata xlsx and ``MRI/``."""
    root = Path(root)
    xlsx = sorted(root.glob("UCSF-PDGM-metadata*.xlsx"))
    if not xlsx:
        raise FileNotFoundError(f"No UCSF-PDGM-metadata*.xlsx in {root}")
    meta, _ = load_metadata(xlsx[-1])
    meta["split"] = assign_splits(meta, seed=seed, fractions=fractions)
    files, _ = scan_mri_dir(root / "MRI")
    for key in (*SEQS, "seg", "seg_merged"):
        meta[f"path_{key}"] = meta.pid.map(lambda p, k=key: files.get(p, {}).get(k))
    has_img = np.all([meta[f"path_{s}"].notna() for s in required_seqs], axis=0)
    has_seg = meta["path_seg"].notna() | meta["path_seg_merged"].notna()
    meta["complete"] = has_img & (has_seg if require_seg else True)
    return meta


def cases_from_df(df: pd.DataFrame, split: str | None = None, only_complete: bool = True) -> list[Case]:
    d = df
    if split is not None:
        d = d[d.split == split]
    if only_complete:
        d = d[d.complete]
    out = []
    for r in d.itertuples(index=False):
        rd = r._asdict()
        paths = {k[5:]: rd[k] for k in rd if k.startswith("path_") and isinstance(rd[k], str)}
        out.append(Case(rd["pid"], int(rd["y"]), rd["split"], int(rd["grade"]), rd["diagnosis"], rd["idh"], paths))
    return out


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby("split").agg(
        n_total=("pid", "size"), n_grade4=("y", "sum"), n_complete=("complete", "sum")
    )
    g["n_complete_grade4"] = df[df.complete].groupby("split")["y"].sum()
    return g.fillna(0).astype(int)
