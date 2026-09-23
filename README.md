# Local lengthening versus global slowing in speech recognition

Code, protocols, saved outputs and executed notebooks for the ICASSP 2027 submission
*"Local Lengthening Versus Global Slowing in Speech Recognition: Controlling
Time-Stretch Artifacts and Fine-Tuning Exposure"* (B. Mellal, LRIT, Mohammed V University in Rabat).

**Question.** At equal added duration, is lengthening a few vowels harder for a speech recogniser than
slowing the whole utterance? Qur'anic recitation provides the targets: vowel lengthening (madd) is marked in
the text. Two confounds are controlled: artifacts of the time-stretching engine (a stretch-and-restore
round-trip check at unchanged duration) and training exposure (three Whisper-base models trained under one
recipe that heard none, the voices, or the recordings of the evaluated reciters).

## Notebooks (executed, outputs saved)

| Notebook | What it shows |
|---|---|
| `notebooks/01_study_design_and_data.ipynb` | evaluated recordings (410 candidates, 320 retained, 210 confirmatory), the EveryAyah exposure sets, which evaluated recordings occur in the training split |
| `notebooks/02_exposure_controlled_models.ipynb` | training logs of M0, M1, M2 (data sizes, steps, validation curves), the disclosed step deviation, SHA-256 of every evaluated weight file |
| `notebooks/03_stimuli_and_round_trip.ipynb` | the stimuli for one verse rebuilt with the paper's code, equal-length check, spectrograms of every arm, the round trip |
| `notebooks/04_results_figures_tables.ipynb` | every number, table and figure in the paper recomputed from the raw outputs, with an automatic check against the paper's values |
| `notebooks/05_confirmatory_tests_new_data.ipynb` | the tests fixed before their data existed (Amendment 4): unseen reciters, retraining seeds, a CTC recogniser, Whisper-large-v3, English read speech, and the engine-difference analysis, each with its pre-registered verdict and a check against the paper |

## Layout

```
data/          manifest of evaluated verses, EveryAyah census (revision 6ea5108), exposure sets, split membership
protocols/     time-stamped protocols with SHA-256 files (written before each experiment; amendments dated)
results/       raw per-condition outputs of every evaluation run (gzipped JSON lines), training logs
src/original/  the study's original evaluation code (stimulus construction, recognition, statistics), unchanged
src/training/  data preparation and fine-tuning (M0: train.py; M1, M2: train_exposure.py)
src/evaluation/ runners that call the original code with swapped weights; round-trip and dose-response arms
src/analysis/  analysis scripts used for the paper, and stats.py used by the notebooks
tools/         build_notebooks.py (builds and executes the notebooks)
```

## How the experiments were run

* **Training and every model evaluation** (including LibriSpeech and Whisper-large-v3) ran on one NVIDIA RTX 5080 (16 GB) under Windows 11, PyTorch 2.11
  (CUDA 12.8), transformers 5.16.1. Training data: `tarteel-ai/everyayah`, revision
  `6ea510862d64f59e555a4a8363eebc0f415621df`.
* **Evaluation** imports the original pipeline unchanged; `src/evaluation/run_eval_cuda.py` substitutes only
  the device, file locations, the Whisper weight directory and the Rubber Band 4.0.0 command-line path.
* **Statistics** use the original crossed bootstrap (`crossed_summary` in `analyze_confirmation.py`,
  10,000 resamples, seed 20270911); `src/analysis/stats.py` executes that exact function.
* **Pre-specification.** `protocols/` holds the protocols fixed before each experiment, with hashes. The
  paper's Table 2 lists every pre-specified test and its outcome.

## Confirmatory tests on new data (Amendment 4)

The main study's RB prediction for M0 to M2 was written after M0's result was known. Amendment 4
(`protocols/PROTOCOL_exposure.md`, SHA-256 `2ee4e2f3...`, fixed 21 Sep 2026 14:20 Africa/Casablanca before any of its data
existed) tests the RB contrast on new material. Verdicts:
- retraining seeds: supported, 6 of 6 models;
- unseen reciters: supported for M0, M1 and M2;
- CTC recogniser: supported;
- English LibriSpeech: supported for wav2vec2 and Whisper-base, including the round-trip check and the RB-versus-PV difference at k=2;
- Whisper-large-v3: **not supported**, +0.044 [-0.002, +0.093].

Additional data: `data/manifest_fresh.json` (unseen reciters, built by `src/evaluation/make_fresh.py`) and
`data/libri_manifest.json` (LibriSpeech test-clean, stressed-vowel targets from the Montreal Forced Aligner alignments
in `gilkeyio/librispeech-alignments`). The recognisers are:
- `facebook/wav2vec2-base-960h`, revision `22aad52`;
- `openai/whisper-base`, revision `e37978b`;
- `openai/whisper-large-v3`, revision `06f233fe`.

Operational notes, none of which can change a verdict:
- **Whisper-large-v3 setup.** It was evaluated from a real-file copy of its Hugging Face files, because cache symlinks do
  not resolve through the evaluation's cache junction. The copy's `config.json` dtype is set to float32, so it runs in
  float32 like every other model; the weights are identical.
- **Resuming.** Evaluations resume per condition after an interruption.
- **English scoring.** The normaliser also deletes digits, which moves every tested contrast by at most 0.0015.
- **M0 token suppression.** M0's generation config lacks Whisper's default suppression of 88 non-speech tokens. This can
  change greedy output only where such a token would be emitted: 4 of M0's 12,309 conditions, all phase vocoder at k=6.

Numbers: `results/amend4_results.json` and `results/ctc_mechanism.json`; `results/novelty_sweep_20260921.json` records the prior-work search.

## Disclosed deviation

The exposure protocol fixed 9,846 optimizer steps for all three models. M1 and M2 ran the full three epochs
(14,226 and 14,274 steps; checkpoints selected at step 14,000) because the training script sets its length in
epochs and no step cap was passed. M1 and M2 remain matched to each other, which is the comparison the
exposure tests rely on. See `protocols/PROTOCOL_exposure.md`, Amendment 3, and notebook 02.

## Reproducing

```bash
uv venv .venv --python 3.13 && uv pip install -r requirements.txt
.venv/bin/python tools/build_notebooks.py        # re-executes all notebooks
```

Notebooks 01, 02 and 04 need only the files in this repository. Notebook 03 needs a local copy of the
EveryAyah recordings (set `AUDIO_ROOT`) and the Rubber Band 4.0.0 command-line tool; source audio is not
redistributed. Re-running training or recognition requires the EveryAyah dataset, a CUDA GPU, and the scripts
in `src/training` and `src/evaluation`.

## Licence and data

Code: see `LICENSE`. EveryAyah recordings belong to their owners and are not included. The public checkpoint
`tarteel-ai/whisper-base-ar-quran` (revision `5c3c53fdf927`) and the CTC aligner
`rabah2026/wav2vec2-large-xlsr-53-arabic-quran-v2` (revision `98c046d4f5fd`) are loaded from Hugging Face.
