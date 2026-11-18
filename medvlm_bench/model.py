"""Model back-ends.

``MedGemmaScorer``  Hugging Face MedGemma (google/medgemma-1.5-4b-it; gated -> `huggingface-cli login`).
                    Classification is done by *forced choice*: we read the next-token log-probabilities of
                    the answer letters ("A"/"B") after the assistant-turn prompt. This yields continuous
                    scores (AUROC, calibration) instead of brittle free-text parsing, and also reports the
                    probability mass that falls on the valid options (a format-compliance diagnostic).
``MockScorer``      Deterministic pseudo-random scorer for testing the pipeline without a GPU/model.

Message format (both back-ends): list of {"role": "system"|"user"|"assistant",
"content": [{"type": "text", "text": ...} | {"type": "image", "image": PIL.Image}, ...]}
"""
from __future__ import annotations

import hashlib

import numpy as np


class MockScorer:
    name = "mock"

    def __init__(self, seed: int = 0):
        self.seed = seed

    def _h(self, messages) -> float:
        h = hashlib.md5()
        h.update(str(self.seed).encode())
        last_user = [m for m in messages if m["role"] == "user"][-1]
        for part in last_user["content"]:
            if part["type"] == "image":
                h.update(part["image"].resize((16, 16)).tobytes())
            else:
                h.update(part["text"].encode())
        return int(h.hexdigest()[:8], 16) / 0xFFFFFFFF

    def option_logprobs(self, messages, options=("A", "B")) -> np.ndarray:
        u = self._h(messages)
        l = np.log(np.array([u, 1 - u]) * 0.97 + 1e-6)
        return l[: len(options)]

    def generate(self, messages, max_new_tokens: int = 16) -> str:
        return "A" if self._h(messages) > 0.5 else "B"


class MedGemmaScorer:
    name = "medgemma"

    def __init__(self, model_id: str = "google/medgemma-1.5-4b-it", dtype: str = "bfloat16",
                 device_map: str = "auto"):
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        self.torch = torch
        self.model_id = model_id
        self.dtype = getattr(torch, dtype)
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_id, torch_dtype=self.dtype, device_map=device_map
        ).eval()
        self._opt_ids: dict[tuple, list[int]] = {}

    def _prep(self, messages):
        msgs = []
        for m in messages:
            content = []
            for part in m["content"]:
                if part["type"] == "image":
                    content.append({"type": "image", "image": part["image"].convert("RGB")})
                else:
                    content.append(part)
            msgs.append({"role": m["role"], "content": content})
        inputs = self.processor.apply_chat_template(
            msgs, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt"
        )
        return inputs.to(self.model.device, dtype=self.dtype)

    def _option_ids(self, options) -> list[int]:
        if options not in self._opt_ids:
            ids = []
            for o in options:
                t = self.processor.tokenizer.encode(o, add_special_tokens=False)
                if len(t) != 1:
                    raise ValueError(f"Option {o!r} is not a single token: {t}")
                ids.append(t[0])
            self._opt_ids[options] = ids
        return self._opt_ids[options]

    def option_logprobs(self, messages, options=("A", "B")) -> np.ndarray:
        """Full-vocabulary log-probabilities of the option tokens at the first answer position."""
        inputs = self._prep(messages)
        with self.torch.inference_mode():
            logits = self.model(**inputs, logits_to_keep=1).logits[0, -1].float()
        lp = self.torch.log_softmax(logits, dim=-1)
        return lp[self._option_ids(tuple(options))].cpu().numpy()

    def generate(self, messages, max_new_tokens: int = 256) -> str:
        inputs = self._prep(messages)
        n = inputs["input_ids"].shape[-1]
        with self.torch.inference_mode():
            out = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        return self.processor.decode(out[0][n:], skip_special_tokens=True)
