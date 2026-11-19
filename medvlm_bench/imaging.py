"""Volume loading, 2D rendering for the VLM and segmentation-derived quantitative features.

Conventions
-----------
Volumes are reoriented to RAS+ (``as_closest_canonical``) so axis 2 is always axial
(inferior -> superior). Slices are displayed in radiological convention (patient right on the
image left, anterior at the top).

Segmentation labels follow BraTS: 1 = necrotic/non-enhancing core, 2 = peritumoral oedema,
4 = enhancing tumour. ``seg_merged`` is a binary whole-tumour mask (== seg > 0).
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import nibabel as nib
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage as ndi


@lru_cache(maxsize=6)
def load_vol(path: str):
    img = nib.as_closest_canonical(nib.load(path))
    arr = np.asarray(img.dataobj, dtype=np.float32)
    return arr, tuple(float(z) for z in img.header.get_zooms()[:3])


@lru_cache(maxsize=6)
def load_seg(path: str):
    img = nib.as_closest_canonical(nib.load(path))
    arr = np.rint(np.asarray(img.dataobj)).astype(np.int16)
    return arr, tuple(float(z) for z in img.header.get_zooms()[:3])


@lru_cache(maxsize=64)
def _norm_limits(path: str):
    v, _ = load_vol(path)
    nz = v[v > 0]
    if nz.size < 100:
        return 0.0, float(max(v.max(), 1.0))
    lo, hi = np.percentile(nz, (1.0, 99.5))
    return float(lo), float(max(hi, lo + 1e-3))


# --------------------------------------------------------------------------- rendering
@dataclass(frozen=True)
class RenderConfig:
    seqs: tuple = ("T1c", "FLAIR", "T2")
    n_slices: int = 3
    slice_mode: str = "tumor"  # "tumor": slices at cumulative-area quantiles of the tumour (needs seg)
    #                            "uniform": evenly across the brain (no segmentation needed)
    crop: str = "brain"  # "brain": brain bounding box; "tumor": fixed 128 mm FOV around the tumour centroid
    layout: str = "montage"  # "montage": one image (rows=sequences, cols=slices); "slices": one image per tile
    tile: int = 0  # 0 -> auto so that the montage is ~894 px (MedGemma works on 896x896)
    labels: bool = True
    # --- multi-dataset extension: the defaults below reproduce the brain behaviour exactly ---------------------------
    task: str = "brain"  # "brain" -> code path below; any other task -> generic.render_generic
    plane: str = "axial"  # generic: "axial" or "native" (slice along the through-plane axis of the acquisition)
    fov: tuple = (0, 0)  # generic: (min, max) side length in mm of the tumour-centred square crop
    window: tuple = ()  # generic: (lo, hi) display window in native units; () -> percentile normalisation

    def key(self) -> str:
        if self.task == "brain":  # legacy key (same string as before the extension) -> existing render caches stay valid
            legacy = (f"RenderConfig(seqs={self.seqs!r}, n_slices={self.n_slices!r}, slice_mode={self.slice_mode!r}, "
                      f"crop={self.crop!r}, layout={self.layout!r}, tile={self.tile!r}, labels={self.labels!r})")
            return hashlib.md5(legacy.encode()).hexdigest()[:10]
        return hashlib.md5(repr(self).encode()).hexdigest()[:10]

    def describe(self) -> str:
        if self.task != "brain":
            from .tasks import describe_layout
            return describe_layout(self)
        cols = f"{self.n_slices} axial slice{'s' if self.n_slices > 1 else ''}"
        order = " (left to right: inferior to superior)" if self.n_slices > 1 else ""
        seqs = ", ".join(self.seqs)
        if self.layout == "montage":
            return (
                f"Each image is a montage of axial brain MRI slices (radiological convention). "
                f"Rows from top to bottom: {seqs} (labelled in the corner of each tile). "
                f"Columns: {cols} through the tumour{order}."
            )
        return (
            f"Each case is given as a series of axial brain MRI slices (radiological convention), "
            f"ordered by sequence ({seqs}) and then by slice{order}."
        )


def _quantile_slices(area: np.ndarray, n: int) -> list[int]:
    if n == 1:
        return [int(area.argmax())]
    cum = np.cumsum(area) / area.sum()
    idx = [int(np.searchsorted(cum, q)) for q in (np.arange(n) + 0.5) / n]
    out: list[int] = []
    for i in idx:
        i = min(i, len(area) - 1)
        while i in out and i + 1 < len(area):
            i += 1
        out.append(i)
    return sorted(out)


def _square_box(cx, cy, side, X, Y):
    side = int(min(side, X, Y))
    x0 = int(min(max(round(cx - side / 2), 0), X - side))
    y0 = int(min(max(round(cy - side / 2), 0), Y - side))
    return x0, x0 + side, y0, y0 + side


def _font(size: int):
    try:
        return ImageFont.load_default(size=size)
    except TypeError:  # old Pillow
        return ImageFont.load_default()


def render_case(case, cfg: RenderConfig, cache_dir: str | Path | None = None) -> list[Image.Image]:
    """Return a list of RGB PIL images (1 for 'montage', rows*cols for 'slices')."""
    if cfg.task != "brain":
        from .generic import render_generic
        return render_generic(case, cfg, cache_dir)
    cache = Path(cache_dir) if cache_dir else None
    if cache:
        cache.mkdir(parents=True, exist_ok=True)
        hit = sorted(cache.glob(f"{case.pid}_{cfg.key()}_*.png"))
        if hit:
            return [Image.open(p).convert("RGB") for p in hit]

    vols = {s: load_vol(case.path(s))[0] for s in cfg.seqs}
    ref = vols[cfg.seqs[0]]
    X, Y, Z = ref.shape

    seg_path = case.path("seg") or case.path("seg_merged")
    tumor = None
    if seg_path:
        tumor = load_seg(seg_path)[0] > 0
    if cfg.slice_mode == "tumor":
        if tumor is None or tumor.sum() == 0:
            raise ValueError(f"{case.pid}: slice_mode='tumor' needs a non-empty segmentation")
        zs = _quantile_slices(tumor.sum(axis=(0, 1)), cfg.n_slices)
    else:
        area = (ref > 0).sum(axis=(0, 1))
        nz = np.where(area > 0.02 * area.max())[0]
        z0, z1 = nz.min(), nz.max()
        zs = [int(round(z)) for z in np.linspace(z0 + 0.15 * (z1 - z0), z1 - 0.15 * (z1 - z0), cfg.n_slices)]

    if cfg.crop == "tumor" and tumor is not None and tumor.sum() > 0:
        cx, cy, _ = ndi.center_of_mass(tumor)
        box = _square_box(cx, cy, 128, X, Y)
    else:
        m = ref > 0
        xs, ys = np.where(m.any(axis=2).any(axis=1))[0], np.where(m.any(axis=2).any(axis=0))[0]
        box = _square_box((xs.min() + xs.max()) / 2, (ys.min() + ys.max()) / 2, max(np.ptp(xs), np.ptp(ys)) + 8, X, Y)
    x0, x1, y0, y1 = box

    n_rows, n_cols = len(cfg.seqs), len(zs)
    tile = cfg.tile or (894 // max(n_rows, n_cols) if cfg.layout == "montage" else 448)
    font = _font(max(10, tile // 14))
    tiles = []
    for s in cfg.seqs:
        lo, hi = _norm_limits(case.path(s))
        for z in zs:
            sl = vols[s][x0:x1, y0:y1, z]
            u8 = (np.clip((sl - lo) / (hi - lo), 0, 1) * 255).astype(np.uint8)
            u8 = u8.T[::-1, ::-1]  # rows: anterior at top; cols: patient right on image left
            im = Image.fromarray(u8).convert("RGB").resize((tile, tile), Image.BICUBIC)
            if cfg.labels:
                ImageDraw.Draw(im).text((4, 2), s, fill=(255, 255, 0), font=font)
            tiles.append(im)

    if cfg.layout == "slices":
        imgs = tiles
    else:
        W, H = n_cols * tile, n_rows * tile
        side = max(W, H)
        canvas = Image.new("RGB", (side, side))
        for i, im in enumerate(tiles):
            canvas.paste(im, ((i % n_cols) * tile, (i // n_cols) * tile))
        imgs = [canvas]
    if cache:
        for j, im in enumerate(imgs):
            im.save(cache / f"{case.pid}_{cfg.key()}_{j:02d}.png")
    return imgs


# --------------------------------------------------------------------------- features
FEATURE_KEYS = [
    "tumor_ml", "necrosis_ml", "edema_ml", "enh_ml",
    "necrosis_frac_core", "enh_frac_core", "edema_to_core", "rim_frac",
    "frac_minor_side", "cx", "cy", "cz",
    "enh_t1c_ratio", "edema_flair_ratio", "core_t2_ratio",
]


def compute_case_features(case, cache_dir: str | Path | None = None) -> dict:
    """Quantitative, segmentation-derived descriptors (NaN where inputs are missing)."""
    cache = Path(cache_dir) / f"{case.pid}_feat.json" if cache_dir else None
    if cache and cache.exists():
        return json.loads(cache.read_text())
    f = {k: float("nan") for k in FEATURE_KEYS}
    seg_full = case.path("seg")
    seg_path = seg_full or case.path("seg_merged")
    if seg_path:
        seg, zooms = load_seg(seg_path)
        ml = float(np.prod(zooms)) / 1000.0
        whole = seg > 0
        f["tumor_ml"] = float(whole.sum() * ml)
        X, Y, Z = seg.shape
        if whole.any():
            cx, cy, cz = ndi.center_of_mass(whole)
            f["cx"], f["cy"], f["cz"] = cx / X, cy / Y, cz / Z
            mid = X / 2.0
            ref_path = next((case.path(s) for s in ("T1c", "FLAIR", "T2", "T1") if case.path(s)), None)
            if ref_path:
                m = load_vol(ref_path)[0] > 0
                xs = np.where(m.any(axis=(1, 2)))[0]
                mid = (xs.min() + xs.max()) / 2.0
            xi = np.where(whole)[0]
            left, right = (xi < mid).sum(), (xi >= mid).sum()
            f["frac_minor_side"] = float(min(left, right) / max(left + right, 1))
        if seg_full:
            nec, edm, enh = seg == 1, seg == 2, seg == 4
            f["necrosis_ml"], f["edema_ml"], f["enh_ml"] = (float(m.sum() * ml) for m in (nec, edm, enh))
            core = nec.sum() + enh.sum()
            f["necrosis_frac_core"] = float(nec.sum() / core) if core else 0.0
            f["enh_frac_core"] = float(enh.sum() / core) if core else 0.0
            f["edema_to_core"] = float(edm.sum() / (core + 1e-6)) if core else float(edm.sum() > 0) * 10.0
            if enh.any() and nec.any():
                dil = ndi.binary_dilation(nec, iterations=2)
                f["rim_frac"] = float((dil & enh).sum() / enh.sum())
            else:
                f["rim_frac"] = 0.0
            for key, seq, mask in (("enh_t1c_ratio", "T1c", enh), ("edema_flair_ratio", "FLAIR", edm),
                                   ("core_t2_ratio", "T2", nec | enh)):
                p = case.path(seq)
                if p and mask.any():
                    v = load_vol(p)[0]
                    med = float(np.median(v[v > 0])) or 1.0
                    f[key] = float(v[mask].mean() / med)
    if cache:
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(f))
    return f


# Heuristic, a-priori thresholds (NOT tuned on any labels). Adjust in one place if needed.
THRESH = dict(
    necrosis_frac_core=0.15, necrosis_ml=2.0, enh_ml_ring=3.0, rim_frac=0.5,
    enh_ml_avid=5.0, enh_ml_none=0.5, edema_ml_large=40.0, minor_side_midline=0.10,
)


def findings_from_features(f: dict) -> dict[str, bool]:
    """Map quantitative descriptors to KG imaging-feature ids (only those derivable from a mask)."""
    T = THRESH
    g = lambda k: f.get(k, float("nan"))  # noqa: E731
    ok = lambda *ks: all(np.isfinite(g(k)) for k in ks)  # noqa: E731
    out: dict[str, bool] = {}
    if ok("necrosis_frac_core", "necrosis_ml"):
        out["central_necrosis"] = g("necrosis_frac_core") >= T["necrosis_frac_core"] and g("necrosis_ml") >= T["necrosis_ml"]
    if ok("enh_ml", "necrosis_ml", "rim_frac"):
        out["ring_enhancement"] = g("enh_ml") >= T["enh_ml_ring"] and g("necrosis_ml") >= T["necrosis_ml"] and g("rim_frac") >= T["rim_frac"]
        out["avid_enhancement"] = g("enh_ml") >= T["enh_ml_avid"]
        out["no_enhancement"] = g("enh_ml") < T["enh_ml_none"]
    if ok("edema_ml"):
        out["large_edema"] = g("edema_ml") >= T["edema_ml_large"]
    if ok("frac_minor_side"):
        out["crosses_midline"] = g("frac_minor_side") >= T["minor_side_midline"]
    return out


def features_to_text(f: dict) -> str:
    """Compact text description of the quantitative findings (used for 'findings' ICL format)."""
    def fmt(k, unit, d=1):
        v = f.get(k, float("nan"))
        return f"{v:.{d}f} {unit}" if np.isfinite(v) else None
    parts = [
        ("whole tumour volume", fmt("tumor_ml", "mL")),
        ("enhancing tumour", fmt("enh_ml", "mL")),
        ("necrotic/non-enhancing core", fmt("necrosis_ml", "mL")),
        ("peritumoral oedema", fmt("edema_ml", "mL")),
    ]
    s = "; ".join(f"{n} {v}" for n, v in parts if v)
    return f"Quantitative tumour findings (from segmentation): {s}." if s else ""
