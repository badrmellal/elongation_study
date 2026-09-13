# Local versus global duration robustness

Matched-duration ablation study using frozen Whisper and wav2vec2 CTC checkpoints.
No model training is performed. The main analysis covers 5,436 conditions, eight
reciters (five confirmation and three exploratory), and 410 candidate recordings.

## Install

Use Python 3.13 and install the recorded environment:

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r confirmation/requirements-lock.txt
```

The lock records the actual macOS environment, not a tested cross-platform
installer. Rubber Band CLI 4.0.0 is required for synthesis controls. The two
small modules in `release/src` retain the historical import layout. Only
the original normalization/edit-distance functions are extracted into
`exp1_sweep.py`; unrelated older experiments are not included.

## Reproduce the paper from saved transcripts

No source audio or model weights are loaded for this command:

```sh
python confirmation/analyze_confirmation.py --lexical-scoring --output analysis_reproduced.json
```

The analyzer validates all original scores and manifest coverage, then removes
nonlexical U+06DE from references and hypotheses and recomputes edit distance,
denominators, confidence intervals, and Holm-adjusted tests. This is a documented
post-hoc scoring amendment dated 2026-09-13, not a prospectively frozen rule.
Original observations and protocol remain unchanged. The correction affects
85 conditions from five recordings of verse 100:9. Compare results with
`confirmation/analysis_lexical.json`; source hashes can differ after packaging.
Primary equal-reciter contrast: 0.17280045351473922, 95% CI
[0.0847950680272109, 0.2711499433106575].

## Repeat inference on the same sample

Audio and pretrained weights are not redistributed. Obtain the exact mono
16 kHz WAV files identified by the frozen manifest and their matching hashes.
The original reference index is also required: set `QURAN_INDEX` to your
`quran_verses.json`; its hash must match `index_sha256` in the manifest. Retained
verse references are included in the manifest for score-only reproduction.
Public availability of recordings does not grant redistribution permission.

Download the pinned checkpoint snapshots to the default Hugging Face cache:

```sh
hf download tarteel-ai/whisper-base-ar-quran --revision 5c3c53fdf9272c4f6ee0bee09a1e5a4a615ee25c
hf download rabah2026/wav2vec2-large-xlsr-53-arabic-quran-v2 --revision 98c046d4f5fd74ed4c5a7078f5129fd85fb53d0c
python relocate_manifest.py --audio-root /path/to/audio --output manifest_local.json
export QURAN_INDEX=/path/to/quran_verses.json
python confirmation/run_confirmation.py run --manifest manifest_local.json --output rerun --device cpu
python confirmation/analyze_confirmation.py --manifest manifest_local.json --results rerun --lexical-scoring --output rerun_analysis.json
```

Use `--device mps` on compatible Macs. Resume only with the same command and
`--resume`. Never append new inference to the bundled historical results.
Relocation changes the manifest hash but preserves verse selection and source
hashes. Historical absolute paths in the original manifest are provenance,
not paths the score-only command needs to access. Cross-device bitwise identity
is not promised. This release has not repeated full model inference.

## Tests and figures

```sh
python -B -m unittest discover -s confirmation -q
python figures/make_figures.py --analysis analysis_reproduced.json
```

The tokenizer regression needs the pinned Whisper cache; backend tests need
Rubber Band. Figure outputs appear in `figures/figs`; numerical LaTeX inputs
appear beside the generator. All 61 tests passed in the recorded environment.
Full clean-environment installation remains untested.

## Scope and interpretation

Local, global, silence, boundary-blended, unity-rate sham and Rubber Band
controls are included. The frozen protocol defines sampling and exclusions.
Short clips, model-derived boundaries, training overlap uncertainty, and only
five confirmation voices limit generalization. Attention is descriptive, not
causal evidence. Failed exploratory mitigation experiments are not part of
this study or package. The historical inference metadata records source hashes
of the original runner; packaging changes input-path handling, disables new
sampling, isolates scoring helpers, and makes log tests independent of the
external reference index.