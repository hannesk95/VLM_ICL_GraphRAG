"""Task-agnostic rendering and mask-derived features for the non-brain datasets (CT / MRI, any slice plane).

The brain pipeline (``imaging.render_case`` / ``compute_case_features``) is left untouched; ``imaging.render_case``
dispatches to :func:`render_generic` whenever ``cfg.task != "brain"``.

Conventions
-----------
* Volumes are reoriented to RAS+ (``as_closest_canonical``). Axis 0 increases to the patient's right, axis 1 to
  anterior, axis 2 to superior. Voxel sizes are read from the affine (after reorientation).
* Slices are taken along one axis (``axial`` = axis 2, ``native`` = the through-plane axis of the acquisition,
  e.g. coronal STIR) and shown in radiological convention (patient right on the image left; anterior / superior at
  the top).
* The tumour mask (any label > 0) is used ONLY for (i) choosing the slices (quantiles of the tumour area along the
  slice axis) and (ii) centring / sizing a physically square field of view (FOV). It is never drawn into the image.
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import nibabel as nib
import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage as ndi

from .imaging import _font, _quantile_slices

GRID_COLS = 3


# --------------------------------------------------------------------------- loading
def _to3d(a: np.ndarray) -> np.ndarray:
    a = np.squeeze(a)
    while a.ndim > 3:  # e.g. a 4th singleton / first time point
        a = a[..., 0]
    if a.ndim != 3:
        raise ValueError(f"expected a 3D volume, got shape {a.shape}")
    return a


@lru_cache(maxsize=6)
def load_ras(path: str):
    img = nib.as_closest_canonical(nib.load(path))
    arr = _to3d(np.asarray(img.dataobj, dtype=np.float32))
    return arr, tuple(float(z) for z in nib.affines.voxel_sizes(img.affine)[:3])


@lru_cache(maxsize=6)
def load_mask_ras(path: str) -> np.ndarray:
    """Binary tumour mask (any label > 0; the datasets use 1 or 255)."""
    img = nib.as_closest_canonical(nib.load(path))
    return _to3d(np.asarray(img.dataobj)) > 0


@lru_cache(maxsize=64)
def _pct_limits(path: str) -> tuple[float, float]:
    v = load_ras(path)[0]
    nz = v[v > 0][::4]
    if nz.size < 100:
        return 0.0, float(max(v.max(), 1.0))
    lo, hi = np.percentile(nz, (1.0, 99.5))
    return float(lo), float(max(hi, lo + 1e-3))


def slice_axis(zooms, plane: str) -> int:
    """Axis along which slices are taken."""
    z = np.asarray(zooms, dtype=float)
    if plane == "native" and z.max() >= 1.5 * z.min():
        return int(np.argmax(z))  # through-plane axis of the original acquisition
    return 2


def order_text(axis: int) -> tuple[str, str]:
    """Direction in which the slice index increases in RAS+ (first, last)."""
    return {0: ("the patient's left", "the patient's right"), 1: ("posterior", "anterior"), 2: ("inferior", "superior")}[axis]


def _crop2d(sl: np.ndarray, a0: int, a1: int, b0: int, b1: int, fill: float) -> np.ndarray:
    """sl[a0:a1, b0:b1] where indices may fall outside the array (padded with `fill`)."""
    out = np.full((a1 - a0, b1 - b0), fill, dtype=np.float32)
    ia0, ia1 = max(a0, 0), min(a1, sl.shape[0])
    ib0, ib1 = max(b0, 0), min(b1, sl.shape[1])
    if ia1 > ia0 and ib1 > ib0:
        out[ia0 - a0:ia1 - a0, ib0 - b0:ib1 - b0] = sl[ia0:ia1, ib0:ib1]
    return out


def _span(c: float, n: int, size: int) -> tuple[int, int]:
    """Window of n voxels centred on c, shifted to stay inside [0, size) whenever possible."""
    s = int(round(c - n / 2))
    if n <= size:
        s = min(max(s, 0), size - n)
    return s, s + n


# --------------------------------------------------------------------------- rendering
def render_generic(case, cfg, cache_dir=None) -> list[Image.Image]:
    cache = Path(cache_dir) if cache_dir else None
    if cache:
        cache.mkdir(parents=True, exist_ok=True)
        hit = sorted(cache.glob(f"{case.pid}_{cfg.key()}_*.png"))
        if hit:
            return [Image.open(p).convert("RGB") for p in hit]

    seq0 = cfg.seqs[0]
    ref, zooms = load_ras(case.path(seq0))
    seg_path = case.path("seg")
    if not seg_path:
        raise ValueError(f"{case.pid}: no tumour mask")
    tumor = load_mask_ras(seg_path)
    if tumor.shape != ref.shape:
        raise ValueError(f"{case.pid}: mask shape {tumor.shape} != image shape {ref.shape}")
    if not tumor.any():
        raise ValueError(f"{case.pid}: empty tumour mask")

    ax = slice_axis(zooms, cfg.plane)
    ra, rb = [i for i in range(3) if i != ax]  # in-plane axes (increasing order)
    others = tuple(i for i in range(3) if i != ax)
    if cfg.slice_mode == "tumor":
        zs = _quantile_slices(tumor.sum(axis=others), cfg.n_slices)
    else:  # "uniform": evenly across the scanned region, no use of the mask
        thr = -500.0 if cfg.window and cfg.window[0] < 0 else 0.0
        area = (ref > thr).sum(axis=others)
        nz = np.where(area > 0.02 * area.max())[0]
        z0, z1 = nz.min(), nz.max()
        zs = [int(round(z)) for z in np.linspace(z0 + 0.15 * (z1 - z0), z1 - 0.15 * (z1 - z0), cfg.n_slices)]

    # tumour-centred, physically square FOV (side in mm, adaptive to the tumour size)
    idx = np.where(tumor)
    lo_, hi_ = [i.min() for i in idx], [i.max() for i in idx]
    ext_mm = max((hi_[ra] - lo_[ra] + 1) * zooms[ra], (hi_[rb] - lo_[rb] + 1) * zooms[rb])
    fmin, fmax = cfg.fov
    side = float(min(max(1.5 * ext_mm, fmin), fmax))
    na, nb = int(round(side / zooms[ra])), int(round(side / zooms[rb]))
    a0, a1 = _span((lo_[ra] + hi_[ra]) / 2.0, na, ref.shape[ra])
    b0, b1 = _span((lo_[rb] + hi_[rb]) / 2.0, nb, ref.shape[rb])

    n_seq, n_sl = len(cfg.seqs), len(zs)
    if n_seq == 1:
        cols = min(GRID_COLS, n_sl)
        rows = -(-n_sl // cols)
    else:  # rows = sequences, columns = slices (same as the brain montage)
        rows, cols = n_seq, n_sl
    tile = cfg.tile or (894 // max(rows, cols) if cfg.layout == "montage" else 448)
    font = _font(max(10, tile // 14))
    tiles = []
    for s in cfg.seqs:
        vol = ref if s == seq0 else load_ras(case.path(s))[0]
        lo, hi = cfg.window if cfg.window else _pct_limits(case.path(s))
        fill = lo if cfg.window else 0.0
        for k, z in enumerate(zs):
            sl = np.take(vol, z, axis=ax)
            sl = _crop2d(sl, a0, a1, b0, b1, fill)
            u8 = (np.clip((sl - lo) / (hi - lo), 0, 1) * 255).astype(np.uint8)
            u8 = u8.T[::-1, ::-1]  # rows: anterior/superior at the top; cols: patient right on image left
            im = Image.fromarray(u8).convert("RGB").resize((tile, tile), Image.BICUBIC)
            if cfg.labels and (n_seq > 1 or n_sl > 1):
                ImageDraw.Draw(im).text((4, 2), s if n_seq > 1 else str(k + 1), fill=(255, 255, 0), font=font)
            tiles.append(im)

    if cfg.layout == "slices":
        imgs = tiles
    else:
        side_px = max(cols, rows) * tile
        canvas = Image.new("RGB", (side_px, side_px))
        for i, im in enumerate(tiles):
            canvas.paste(im, ((i % cols) * tile, (i // cols) * tile))
        imgs = [canvas]
    if cache:
        for j, im in enumerate(imgs):
            im.save(cache / f"{case.pid}_{cfg.key()}_{j:02d}.png")
    return imgs


# --------------------------------------------------------------------------- mask-derived features
GENERIC_KEYS = [
    "tumor_ml", "eq_diam_mm", "ext_max_mm", "ext_mid_mm", "ext_min_mm", "elongation", "bbox_fill", "n_comp",
    "int_mean", "int_std", "int_cv", "int_p10", "int_p90", "rim_contrast", "cx", "cy", "cz",
]
GENERIC_LOG_KEYS = {"tumor_ml", "eq_diam_mm", "ext_max_mm", "ext_mid_mm", "ext_min_mm"}


def compute_generic_features(case, cache_dir=None, kind: str = "mri", seq: str | None = None) -> dict:
    """Quantitative, mask-derived descriptors (shape, size, intensity statistics, rim contrast).

    kind: "ct_hu" (raw Hounsfield units), "ct_u8" (already windowed 0..255 CT) or "mri" (intensities are divided by
    the median of the non-zero voxels, because MRI units are arbitrary).
    """
    cache = Path(cache_dir) / f"{case.pid}_feat.json" if cache_dir else None
    if cache and cache.exists():
        return json.loads(cache.read_text())
    f = {k: float("nan") for k in GENERIC_KEYS}
    seg_path = case.path("seg")
    img_path = case.path(seq) if seq else next((p for k, p in case.paths.items() if k != "seg"), None)
    if seg_path:
        m = load_mask_ras(seg_path)
        vol, zooms = load_ras(img_path) if img_path else (None, None)
        if zooms is None:
            import nibabel as _nib
            zooms = tuple(float(z) for z in _nib.affines.voxel_sizes(_nib.as_closest_canonical(_nib.load(seg_path)).affine)[:3])
        if m.any():
            vox = float(np.prod(zooms))
            n = int(m.sum())
            f["tumor_ml"] = n * vox / 1000.0
            f["eq_diam_mm"] = float((6.0 * n * vox / np.pi) ** (1.0 / 3.0))
            idx = np.where(m)
            ext = sorted(((i.max() - i.min() + 1) * z for i, z in zip(idx, zooms)), reverse=True)
            f["ext_max_mm"], f["ext_mid_mm"], f["ext_min_mm"] = (float(e) for e in ext)
            f["elongation"] = float(ext[0] / max(ext[2], 1e-6))
            bbox_vox = np.prod([(i.max() - i.min() + 1) for i in idx])
            f["bbox_fill"] = float(n / bbox_vox)
            lab, nc = ndi.label(m)
            if nc > 1:
                sizes = np.asarray(ndi.sum(m, lab, range(1, nc + 1)))
                nc = int((sizes >= 0.05 * sizes.max()).sum())
            f["n_comp"] = float(nc)
            cx, cy, cz = ndi.center_of_mass(m)
            f["cx"], f["cy"], f["cz"] = cx / m.shape[0], cy / m.shape[1], cz / m.shape[2]
            if vol is not None and vol.shape == m.shape:
                sl = tuple(slice(max(i.min() - 6, 0), i.max() + 7) for i in idx)  # work on the tumour neighbourhood only
                mm, vv = m[sl], vol[sl]
                shell = ndi.binary_dilation(mm, iterations=3) & ~mm
                scale = 1.0
                if kind == "mri":
                    pos = vol[vol > 0][::4]
                    scale = float(np.median(pos)) if pos.size else 1.0
                    scale = scale or 1.0
                x = vv[mm] / scale
                f["int_mean"], f["int_std"] = float(x.mean()), float(x.std())
                f["int_cv"] = float(x.std() / (abs(x.mean()) + 1e-6)) if kind == "mri" else float(x.std())
                f["int_p10"], f["int_p90"] = (float(q) for q in np.percentile(x, (10, 90)))
                if shell.any():
                    f["rim_contrast"] = float(x.mean() - (vv[shell] / scale).mean())
    if cache:
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(f))
    return f


def generic_features_to_text(f: dict, kind: str = "mri") -> str:
    """Compact text description of the mask-derived findings (used by the '*_findings' ICL format)."""
    def ok(k):
        return np.isfinite(f.get(k, float("nan")))
    parts = []
    if ok("tumor_ml"):
        parts.append(f"tumour volume {f['tumor_ml']:.1f} mL")
    if ok("ext_max_mm"):
        parts.append(f"largest extent {f['ext_max_mm']:.0f} mm")
    if ok("n_comp") and f["n_comp"] > 1:
        parts.append(f"{int(f['n_comp'])} separate lesion components")
    if ok("int_mean"):
        if kind == "ct_hu":
            parts.append(f"mean attenuation {f['int_mean']:.0f} HU (SD {f['int_std']:.0f})")
        elif kind == "ct_u8":
            parts.append(f"mean display intensity {f['int_mean']:.0f}/255 (SD {f['int_std']:.0f})")
        else:
            parts.append(f"mean signal {f['int_mean']:.2f}x the image median (CV {f['int_cv']:.2f})")
    s = "; ".join(parts)
    return f"Quantitative tumour findings (from segmentation): {s}." if s else ""
