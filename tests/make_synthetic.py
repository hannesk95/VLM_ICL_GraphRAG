"""Create a small synthetic UCSF-PDGM-like dataset (for pipeline tests only; NOT real data).

Mimics: 3-digit metadata IDs, 4-digit file IDs, LPS orientation, BraTS seg labels (1/2/4), merged
masks, follow-up files, a '.filepart' file and incomplete cases (to test the completeness logic).
"""
from __future__ import annotations

import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd

SHAPE = (64, 64, 40)


def _vol(rng, centre, r, kind, grade4):
    X, Y, Z = np.indices(SHAPE)
    brain = (((X - 32) / 28) ** 2 + ((Y - 32) / 28) ** 2 + ((Z - 20) / 17) ** 2) < 1
    base = rng.normal(0.4, 0.05, SHAPE) * brain
    d = np.sqrt(((X - centre[0])) ** 2 + ((Y - centre[1])) ** 2 + ((Z - centre[2]) * 1.3) ** 2)
    seg = np.zeros(SHAPE, np.int16)
    seg[(d < r * 1.7) & brain] = 2
    seg[(d < r) & brain] = 4
    if grade4:
        seg[(d < r * 0.5) & brain] = 1
    return brain, base, seg


def make(root: str | Path, n: int = 60, seed: int = 0):
    root = Path(root); (root / "MRI").mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    aff = np.diag([-1.0, -1.0, 1.0, 1.0])  # LPS
    rows = []
    for i in range(1, n + 1):
        g4 = rng.random() < 0.78
        grade = 4 if g4 else int(rng.choice([2, 3]))
        idh = "wildtype" if (g4 and rng.random() < 0.93) or (not g4 and rng.random() < 0.3) else "IDH1 p.R132H"
        dx = "Glioblastoma, IDH-wildtype" if (g4 and idh == "wildtype") else "Astrocytoma, IDH-mutant"
        rows.append({"ID": f"UCSF-PDGM-{i:03d}", "Sex": "M", "Age at MRI": 60, "WHO CNS Grade": grade,
                     "Final pathologic diagnosis (WHO 2021)": dx, "MGMT status": "positive", "MGMT index": 1,
                     "1p/19q": "intact", "IDH": idh, "1-dead 0-alive": 1, "OS": 100, "EOR": "GTR",
                     "Biopsy prior to imaging": "No", "BraTS21 ID": np.nan,
                     "BraTS21 Segmentation Cohort": np.nan, "BraTS21 MGMT Cohort": np.nan})
        pid = f"UCSF-PDGM-{i:04d}"
        centre = rng.integers(18, 46, 3); centre[2] = rng.integers(12, 28)
        r = rng.uniform(5, 9) if g4 else rng.uniform(3, 6)
        brain, base, seg = _vol(rng, centre, r, "x", g4)
        t1c = base.copy(); t1c[seg == 4] += 0.5 if g4 else 0.15; t1c[seg == 1] -= 0.2
        flair = base.copy(); flair[seg > 0] += 0.4
        t2 = base.copy(); t2[seg > 0] += 0.35
        t1 = base.copy()
        vols = {"T1c": t1c, "FLAIR": flair, "T2": t2, "T1": t1}
        incomplete = i % 9 == 0  # every 9th case lacks FLAIR
        for s, v in vols.items():
            if s == "FLAIR" and incomplete:
                continue
            nib.save(nib.Nifti1Image((np.clip(v, 0, None) * 1000 * brain).astype(np.float32), aff), root / "MRI" / f"{pid}_{s}_bias.nii.gz")
        nib.save(nib.Nifti1Image(seg.astype(np.uint8), aff), root / "MRI" / f"{pid}_tumor_segmentation.nii.gz")
        nib.save(nib.Nifti1Image((seg > 0).astype(np.uint8), aff), root / "MRI" / f"{pid}_tumor_segmentation_merged.nii.gz")
    # follow-up / partial files / follow-up metadata row that must be ignored
    nib.save(nib.Nifti1Image(np.zeros(SHAPE, np.float32), aff), root / "MRI" / "UCSF-PDGM-0001_FU001d_T1c_bias.nii.gz")
    (root / "MRI" / "UCSF-PDGM-0002_FLAIR_bias.nii.gz.filepart").write_bytes(b"partial")
    rows.append({**rows[0], "ID": "UCSF-PDGM-0001_FU001d"})
    pd.DataFrame(rows).to_excel(root / "UCSF-PDGM-metadata_v5.xlsx", index=False)
    return root


if __name__ == "__main__":
    print(make(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 60))
