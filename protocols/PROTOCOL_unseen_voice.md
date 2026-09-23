# Unseen-voice retrain protocol

Written 2026-09-19 (Africa/Casablanca afternoon), BEFORE any training step and
BEFORE any recognition of the paper's clips by a retrained model. SHA256 is
recorded in `PROTOCOL.sha256`. Amendments must be dated, give the reason, and
say whether any outcome had been seen.

## Question

The paper's primary contrast (per-verse WER of PV local k=4 minus PV global
k=4, equal-reciter mean over the five confirmation reciters) is +0.173
[0.086, 0.271] for `tarteel-ai/whisper-base-ar-quran`. All eight paper
reciters are in EveryAyah train, and 166/210 confirmation recordings match an
EveryAyah train recording (heldout_check/, 15 Sep). Reviewers read the effect
as disruption of memorised audio. Question: is the same contrast, computed by
the paper's unchanged pipeline, positive for a whisper-base fine-tuned WITHOUT
any of these voices?

## Model (one run, no search against the paper's clips)

- Init `openai/whisper-base`. Tokenizer/processor = the paper's Tarteel files
  (no weights). Decoder prefix `<|ar|><|transcribe|><|notimestamps|>`, the
  prefix the paper decodes with.
- Data: `tarteel-ai/everyayah` revision `6ea510862d64f59e555a4a8363eebc0f415621df`
  (updated 17 Sep 2026), train split, rows whose `reciter` column is NOT one of
  12 excluded labels: abdul_basit, husary, minshawi, menshawi, alafasy,
  abdurrahmaan_as-sudais, saood_ash-shuraym, yasser_ad-dussary, hani_rifai
  (paper) and sahl_yassin, akram_alalaqimy, muhsin_al_qasim (Quran-Lab
  "held-out"). 0.5 s <= duration <= 29 s. Census: 105,026 usable rows, 24
  reciters. Filtering is per row, by the reciter column.
- Fixed hyperparameters: 3 epochs, batch 32, AdamW lr 1e-5, weight decay 0,
  500 warmup steps then linear decay, bf16 autocast, grad clip 1.0, seed
  20260919. Code: `ws/*.py` (hashes in `PROTOCOL.sha256`).
- Checkpoint selection: lowest WER (paper normaliser `norm_ar`) on a fixed
  800-row sample of EveryAyah VALIDATION rows from allowed reciters, every
  2000 steps and at the end. The paper's clips are never read in training.
- One allowed rerun, disclosed: if the best validation WER is above 0.50, rerun
  once with lr 5e-6. No other reruns.

## Evaluation

`run_unseen.py` runs `confirmation/run_confirmation.py run` with
`manifest_v2.json` (sha 7d34d0dd...), device mps, output `unseen_voice/results/`.
The only change: the Whisper weights directory is redirected to the selected
checkpoint (the CTC aligner, rabah2026, is unchanged). Checkpoint sha256 goes
in `results/SWAP.json`. Analysis: `confirmation/analyze_confirmation.py`
unchanged, default scoring (the paper's `analysis.json` setting).

## Gate and decision rules (fixed now)

- Informative gate: unmodified-audio (k=1) macro WER of the retrained model
  over the five confirmation reciters <= 0.30. Above that, the result is
  "uninformative (model too weak)", not a null.
- POSITIVE: equal-reciter mean > 0 and its 95% crossed bootstrap CI excludes 0.
- NULL: CI includes 0. REVERSED: CI entirely below 0.
- Secondary, descriptive only: per-reciter contrasts, k=2, the three
  exploratory reciters.

## Use of the result (author decision, 19 Sep 2026, before training)

Included in the ICASSP 2027 paper only if POSITIVE. Otherwise the author
submits the chosen version unchanged and keeps this result for rebuttal or a
later version.

## Stated limitation

The retrained model is not the Tarteel checkpoint. A positive result supports
"the contrast appears in a Whisper model trained without these voices", not a
statement about Tarteel's training.

## Amendment 1 (2026-09-19, before training, no outcome seen)

`run_unseen.py` takes `UNSEEN_CKPT` / `UNSEEN_OUT` so the swap mechanism can be
validated first: pointed at the ORIGINAL Tarteel weights for one reciter, it
must reproduce the paper's observations for that reciter exactly (same
hypotheses and WER). If not identical, the swap is broken and no retrain
result is reported until it is fixed.

## Amendment 2 (2026-09-19, before the real training run, no outcome seen)

Engineering fixes found in 200-300-step smoke runs (run name `smoke`, not used
for anything): (1) prep index written as UTF-8 (Windows default encoding
truncated Arabic text) and prep skips files whose size differs from the census
(still downloading); (2) the effective batch of 32 is now 2 micro-batches of
16 with gradient accumulation, because batch 32 reached 14.3 of 16 GB and
spilled into system memory (0.5 steps/s); validation generation also uses
batches of 16. Hyperparameters are otherwise unchanged. Downloads use curl
(Python's Xet path stalled); files are verified by census size.
Swap validation (Amendment 1) PASSED: with the original Tarteel weights, all
816 Rifai rows are identical to the paper's observations (hypothesis and WER,
Whisper and CTC).

## Amendment 3 (2026-09-19 17:25, after training, BEFORE any recognition of the paper's clips)

Training finished: 9,846 steps, best validation WER 0.013 at the final step
(0.534 before fine-tuning); `train_log.jsonl`. The saved checkpoint carried
openai/whisper-base's default generation settings (88 suppressed tokens), which
differ from the paper's decoding. So that ONLY the weights differ from the
paper run, `checkpoint/generation_config.json` is replaced by the exact
`generation_config` recorded in `confirmation/results/metadata.json` (no
token suppression, begin-suppress [220, 50257]); the trained file is kept as
`generation_config.trained.json`. Weights are untouched. Validation WER above
was computed with the default suppression, so it is a selection number only.
