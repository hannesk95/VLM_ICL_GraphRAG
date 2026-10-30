"""Case embeddings used for exemplar retrieval (kNN / MMR / case graph) and reference classifiers.

* ``HandcraftedEmbedder`` - segmentation-derived descriptors, no model needed (always available).
* ``MedSigLIPEmbedder``   - image embeddings from google/medsiglip-448 (gated; needs HF access).

Both are fitted on the TRAIN split only (standardisation statistics), then applied to all cases.
"""
from __future__ import annotations

import numpy as np

from .imaging import FEATURE_KEYS, RenderConfig, compute_case_features, render_case

_LOG_KEYS = {"tumor_ml", "necrosis_ml", "edema_ml", "enh_ml", "edema_to_core"}


def _l2(x: np.ndarray) -> np.ndarray:
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-8)


class HandcraftedEmbedder:
    name = "handcrafted"

    def __init__(self, cache_dir=None):
        self.cache_dir = cache_dir
        self.mu = self.sd = self.fill = None

    def _raw(self, cases) -> np.ndarray:
        rows = []
        for c in cases:
            f = compute_case_features(c, self.cache_dir)
            rows.append([np.log1p(f[k]) if k in _LOG_KEYS and np.isfinite(f[k]) else f[k] for k in FEATURE_KEYS])
        return np.asarray(rows, dtype=np.float64)

    def fit(self, train_cases):
        X = self._raw(train_cases)
        self.fill = np.nanmedian(X, axis=0)
        self.fill = np.where(np.isfinite(self.fill), self.fill, 0.0)
        X = np.where(np.isfinite(X), X, self.fill)
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-6
        return self

    def embed(self, cases) -> np.ndarray:
        X = self._raw(cases)
        X = np.where(np.isfinite(X), X, self.fill)
        return _l2((X - self.mu) / self.sd)


class MedSigLIPEmbedder:
    """Mean-pooled MedSigLIP image embedding over the case's rendered images (L2-normalised)."""

    name = "medsiglip"

    def __init__(self, render_cfg: RenderConfig, model_id="google/medsiglip-448", cache_dir=None, device=None):
        import torch
        from transformers import AutoModel, AutoProcessor

        self.torch = torch
        self.cfg, self.cache_dir = render_cfg, cache_dir
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModel.from_pretrained(model_id).to(self.device).eval()

    def fit(self, train_cases):  # nothing to fit
        return self

    def embed(self, cases) -> np.ndarray:
        out = []
        with self.torch.inference_mode():
            for c in cases:
                imgs = render_case(c, self.cfg, self.cache_dir)
                px = self.processor(images=imgs, return_tensors="pt")["pixel_values"].to(self.device)
                e = self.model.get_image_features(pixel_values=px).float().mean(0).cpu().numpy()
                out.append(e)
        return _l2(np.stack(out))
