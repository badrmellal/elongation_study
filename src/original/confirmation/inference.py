"""Frozen ASR inference and text-defined attention queries for confirmation."""

from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import (
    Wav2Vec2ForCTC, Wav2Vec2Processor, WhisperForConditionalGeneration,
    WhisperProcessor, WhisperTokenizerFast,
)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "release 2" / "src"))
from madd import forced_align, madd_intervals
from exp1_sweep import lev, norm_ar

WHISPER_REV = "5c3c53fdf9272c4f6ee0bee09a1e5a4a615ee25c"
CTC_REV = "98c046d4f5fd74ed4c5a7078f5129fd85fb53d0c"
CACHE = Path.home() / ".cache" / "huggingface" / "hub"


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def score(reference: str, hypothesis: str) -> dict:
    ref = norm_ar(reference).split()
    hyp = norm_ar(hypothesis).split()
    if not ref:
        raise ValueError("Empty normalized reference")
    errors = lev(ref, hyp)
    return {"reference": reference, "hypothesis": hypothesis,
            "reference_normalized": " ".join(ref),
            "hypothesis_normalized": " ".join(hyp),
            "errors": errors, "reference_words": len(ref), "wer": errors / len(ref)}


def decode_transcript(tokenizer, ids: list[int]) -> str:
    """Exclude checkpoint control IDs explicitly; skip_special_tokens is insufficient."""
    control_ids = set(tokenizer.all_special_ids)
    control_ids.update(i for token, i in tokenizer.get_vocab().items()
                       if token.startswith("<|") and token.endswith("|>"))
    return tokenizer.decode([i for i in ids if i not in control_ids], skip_special_tokens=True)


def word_span(text: str, char_idx: int) -> tuple[int, int]:
    if not (0 <= char_idx < len(text)) or text[char_idx].isspace():
        raise ValueError("Madd character must be inside a word")
    for match in re.finditer(r"\S+", text):
        if match.start() <= char_idx < match.end():
            return match.start(), match.end()
    raise ValueError("No word found")


def query_rows(offsets: list, span: tuple[int, int]) -> list[int]:
    """Rows predicting overlapping gold tokens: target index j -> input row j-1."""
    lo, hi = span
    return [j - 1 for j, (s, e) in enumerate(offsets)
            if j > 0 and e > s and e > lo and s < hi]


def reference_frame_mask(n_valid: int, exclusions: list[tuple[float, float]]) -> np.ndarray:
    mask = np.ones(n_valid, dtype=bool)
    for start, end in exclusions:
        mask[max(0, int(start / .02) - 1):min(n_valid, int(np.ceil(end / .02)) + 1)] = False
    return mask


def merge_occurrences(items: list[dict], sr: int) -> list[dict]:
    out = []
    for item in sorted(items, key=lambda m: m["start"]):
        lo, hi = int(item["start"] * sr), int(item["end"] * sr)
        if out and lo <= out[-1]["end_sample"] + 1:
            prev = out[-1]
            prev["end_sample"] = max(prev["end_sample"], hi)
            prev["char_indices"].append(item["char_idx"])
        else:
            out.append({"start_sample": lo, "end_sample": hi,
                        "char_indices": [item["char_idx"]]})
    return out


class Models:
    def __init__(self, device: str = "mps"):
        self.device = torch.device(device)
        torch.set_num_threads(4)
        torch.manual_seed(20270911)
        ctc_path = CACHE / "models--rabah2026--wav2vec2-large-xlsr-53-arabic-quran-v2" / "snapshots" / CTC_REV
        whisper_path = CACHE / "models--tarteel-ai--whisper-base-ar-quran" / "snapshots" / WHISPER_REV
        self.cp = Wav2Vec2Processor.from_pretrained(ctc_path, local_files_only=True)
        self.cm = Wav2Vec2ForCTC.from_pretrained(ctc_path, local_files_only=True).eval().to(self.device)
        self.wp = WhisperProcessor.from_pretrained(whisper_path, local_files_only=True)
        self.fast = WhisperTokenizerFast.from_pretrained(whisper_path, local_files_only=True)
        self.wp.tokenizer.set_prefix_tokens(language="ar", task="transcribe", predict_timestamps=False)
        self.fast.set_prefix_tokens(language="ar", task="transcribe", predict_timestamps=False)
        self.wm = WhisperForConditionalGeneration.from_pretrained(
            whisper_path, local_files_only=True, attn_implementation="eager"
        ).eval().to(self.device)
        # This old checkpoint predates the explicit language/task generation API.
        # Recover its existing token IDs, not a different model's settings.
        self.wm.generation_config.lang_to_id = {"<|ar|>": self.wp.tokenizer.convert_tokens_to_ids("<|ar|>")}
        self.wm.generation_config.task_to_id = {"transcribe": self.wp.tokenizer.convert_tokens_to_ids("<|transcribe|>")}
        self.wm.generation_config.no_timestamps_token_id = self.wp.tokenizer.convert_tokens_to_ids("<|notimestamps|>")
        self.wm.generation_config.is_multilingual = True
        self.vocab = self.cp.tokenizer.get_vocab()
        self.blank = self.cm.config.pad_token_id

    @torch.inference_mode()
    def ctc_logits(self, audio: np.ndarray, sr: int = 16000) -> torch.Tensor:
        iv = self.cp(audio, sampling_rate=sr, return_tensors="pt").input_values
        return self.cm(iv.to(self.device)).logits[0].float().cpu()

    def ctc_text(self, logits: torch.Tensor) -> str:
        return self.cp.batch_decode(logits.argmax(-1)[None, :])[0]

    def align(self, audio: np.ndarray, text: str, sr: int = 16000) -> dict:
        logits = self.ctc_logits(audio, sr)
        ids, char_to_id, dropped = [], [], []
        for i, ch in enumerate(text):
            token = "|" if ch == " " else ch
            if token in self.vocab:
                char_to_id.append(len(ids)); ids.append(self.vocab[token])
            else:
                char_to_id.append(-1); dropped.append({"index": i, "character": ch})
        base = {"ctc": score(text, self.ctc_text(logits)), "dropped_characters": dropped}
        if not ids or len(ids) + sum(a == b for a, b in zip(ids, ids[1:])) > len(logits):
            return {**base, "excluded": "CTC target longer than feasible frame path"}
        _, mapping = forced_align(logits.log_softmax(-1).numpy().astype(np.float64), ids, self.blank)
        if set(mapping[mapping >= 0]) != set(range(len(ids))):
            return {**base, "excluded": "Incomplete CTC alignment"}
        found = madd_intervals(text, ids, char_to_id, mapping, len(audio) / sr / len(logits), min_counts=2)
        found = [m for m in found if m["dur_ms"] >= 60]
        occurrences = merge_occurrences(found, sr)
        if len(occurrences) < 2:
            return {**base, "excluded": "Fewer than two merged CTC intervals >=60 ms", "detected": found}
        tokens = self.fast(text, return_offsets_mapping=True)
        if tokens.input_ids != self.wp.tokenizer(text).input_ids:
            raise AssertionError("Fast tokenizer changes checkpoint token IDs")
        for item in occurrences:
            spans = [word_span(text, char) for char in item["char_indices"]]
            item["word_spans"] = spans
            item["query_rows"] = sorted({row for span in spans for row in query_rows(tokens.offset_mapping, span)})
            if not item["query_rows"]:
                raise AssertionError("No text-defined attention query")
        return {**base, "occurrences": occurrences, "token_ids": tokens.input_ids,
                "token_offsets": tokens.offset_mapping, "detected": found}

    @torch.inference_mode()
    def whisper(self, audio: np.ndarray, text: str, occurrences: list[dict],
                intervals: list[tuple[float, float]], sr: int = 16000,
                reuse_encoder: bool = True, reference_exclusions=None) -> dict:
        duration = len(audio) / sr
        if duration > 29:
            raise ValueError("Audio exceeds 29 s, must be excluded jointly across arms")
        feats = self.wp(audio, sampling_rate=sr, return_tensors="pt").input_features.to(self.device)
        ids = self.wp.tokenizer(text, return_tensors="pt").input_ids.to(self.device)
        encoder = self.wm.model.encoder(feats, return_dict=True) if reuse_encoder else None
        forward = {"encoder_outputs": encoder} if encoder is not None else {"input_features": feats}
        out = self.wm(**forward, decoder_input_ids=ids[:, :-1], output_attentions=True, use_cache=False)
        per_head = out.cross_attentions[-1][0].float().cpu().numpy()
        n_valid = min(per_head.shape[-1], int(np.ceil(duration / 0.02)))
        padding_mass = per_head[:, :, n_valid:].sum(-1)
        per_head = per_head[:, :, :n_valid]
        per_head /= per_head.sum(-1, keepdims=True) + 1e-12
        attention = per_head.mean(0)
        # Silence targets exclude gaps, but the reference must exclude those gaps too.
        reference_mask = reference_frame_mask(n_valid, intervals if reference_exclusions is None else reference_exclusions)
        reference_frames = int(reference_mask.sum())
        measures = []
        assert len(occurrences) == len(intervals)
        for index, (item, (start, end)) in enumerate(zip(occurrences, intervals)):
            lo, hi = int(start / 0.02), min(int(end / 0.02), n_valid)
            if hi - lo < 3:
                measures.append({"interval_id": index, "excluded": "Fewer than three attention frames"})
                continue
            masses = attention[:, lo:hi].sum(1)
            q = item["query_rows"]
            fixed = float(masses[q].mean())
            maximum = float(masses.max())
            fraction = (hi - lo) / n_valid
            relative = {}
            if reference_frames >= 10:
                target_mass = per_head[:, q, lo:hi].sum(-1)
                reference_mass = per_head[:, q, :][:, :, reference_mask].sum(-1)
                log_density_ratio = (np.log(np.maximum(target_mass, 1e-12) / (hi - lo))
                                     - np.log(np.maximum(reference_mass, 1e-12) / reference_frames))
                relative = {"log_density_ratio": float(log_density_ratio.mean()),
                            "per_head_log_density_ratio": log_density_ratio.mean(-1).tolist(),
                            "mass_floor_count": int((target_mass < 1e-12).sum() + (reference_mass < 1e-12).sum()),
                            "min_target_or_reference_mass": float(min(target_mass.min(), reference_mass.min()))}
            measures.append({"interval_id": index, "start": start, "end": end,
                             "duration_ms": 1000 * (end - start), "fraction": fraction,
                             "fixed_mass": fixed, "max_mass": maximum,
                             "fixed_enrichment": fixed / fraction,
                             "max_enrichment": maximum / fraction,
                             "query_rows": q, "reference_frames": reference_frames,
                             "mean_padding_mass": float(padding_mass[:, q].mean()), **relative})
        del out
        kwargs = {"encoder_outputs": encoder} if encoder is not None else {}
        generated = self.wm.generate(feats, **kwargs, max_new_tokens=180,
                                     do_sample=False, num_beams=1,
                                     language="ar", task="transcribe", return_timestamps=False,
                                     return_dict_in_generate=True)
        token_ids = generated.sequences[0].cpu().tolist()
        hypothesis = decode_transcript(self.wp.tokenizer, token_ids)
        eos = self.wm.generation_config.eos_token_id
        eos = eos if isinstance(eos, list) else [eos]
        return {**score(text, hypothesis), "generated_ids": token_ids,
                "ended_with_eos": token_ids[-1] in eos,
                "token_cap_hit": token_ids[-1] not in eos and len(token_ids) >= 180,
                "attention": measures, "duration_s": duration,
                "valid_encoder_frames": n_valid}

    def metadata(self) -> dict:
        import platform
        import librosa
        import transformers
        return {"python": platform.python_version(), "torch": torch.__version__,
                "transformers": transformers.__version__, "librosa": librosa.__version__,
                "device": str(self.device), "dtype": "float32", "whisper_revision": WHISPER_REV,
                "ctc_revision": CTC_REV, "generation_config": self.wm.generation_config.to_dict(),
                "generation_overrides": {"max_new_tokens": 180, "do_sample": False, "num_beams": 1,
                                         "language": "ar", "task": "transcribe", "return_timestamps": False,
                                         "return_dict_in_generate": True},
                "prefix_tokens": self.wp.tokenizer.prefix_tokens}
