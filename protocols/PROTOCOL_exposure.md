# Exposure-control protocol (ICASSP 2027 recast)

Written 2026-09-19 (~23:30 Africa/Casablanca), BEFORE the exposure models exist and
before any of their recognition results. Hash in PROTOCOL.sha256. Amendments dated,
with a statement of what had been seen.

## Question

The 19 Sep unseen-voice retrain (unseen_voice/) gave, versus the Tarteel checkpoint:
phase-vocoder (PV) local-minus-global k=4 equal-reciter mean +0.046 [-0.047, +0.138]
(NULL) against the paper's +0.173 [0.086, 0.271]; Rubber Band (RB) +0.111
[+0.028, +0.199] against +0.179 [+0.085, +0.386]; global PV WER 0.414 against 0.219.
That comparison confounds EXPOSURE with RECIPE (Tarteel's training data and recipe are
undocumented; its card says "None dataset"). This experiment removes the confound by
holding the recipe fixed and varying exposure only.

## Models (same recipe, same optimizer steps, same seed, same validation selection)

Recipe identical to unseen_voice/ws/train.py: init openai/whisper-base, the paper's
tokenizer files, prefix <|ar|><|transcribe|><|notimestamps|>, 9,846 optimizer steps,
effective batch 32 (16 x 2), AdamW lr 1e-5, 500 warmup then linear decay, bf16, seed
20260919, selection by WER on the SAME fixed 800-row validation sample of allowed-reciter
EveryAyah validation rows. EveryAyah revision 6ea510862d64f59e555a4a8363eebc0f415621df.
Only the training set changes; the three Quran-Lab held-out reciters stay excluded everywhere.

- M0 = the existing model: no rows from the 8 paper reciters (already trained, 105,026 rows).
- M1 = VOICE-exposed, RECORDING-unseen: M0's rows plus all rows of the 9 paper reciter labels
  EXCEPT rows whose normalised text (norm_ar) matches any evaluated verse text for that
  reciter (exposure/eval_texts.json, built from manifest_v2.json before training).
- M2 = RECORDING-exposed: M1's rows plus those excluded evaluated-verse rows.

M1 vs M2 differ ONLY by the evaluated recordings, so that contrast isolates recording
exposure. M0 vs M1 isolates voice exposure (data size also grows; stated as a limitation).

## Evaluation

All four models (Tarteel, M0, M1, M2) are evaluated on the RTX 5080 with
unseen_voice/ws/run_eval_cuda.py: the paper's pipeline imported unchanged, only device
(cuda), audio path prefix, model cache root, Whisper weights dir and the Rubber Band 4.0.0
Windows CLI path substituted. Rubber Band 4.0.0 is the version the paper used. Analysis
by the paper's unchanged analyze_confirmation.py. Every model in the comparison is
evaluated on the SAME device; Mac (MPS) numbers are not mixed into it. Device
reproduction check: Tarteel on CUDA versus the paper's MPS numbers is reported first, and
if the k=4 PV equal-reciter mean moves by more than 0.03, the device change is reported as
a caveat on every CUDA number.

## Pre-registered predictions (exposure hypothesis)

E1. Damage relative to each model's own clean WER, global PV k=4 arm: M2 < M1 < M0.
E2. The PV local-minus-global k=4 contrast is larger for M2 than for M0.
E3. The RB local-minus-global k=4 contrast stays positive for all of M0, M1, M2
    (the duration effect is not exposure-driven).
E4. Clean (k=1) WER: M2 <= M1 <= M0 on the evaluated clips; reported alongside every
    damage number, since exposure also lowers clean WER (Teixeira 2024).

Primary test for the exposure claim: E1 and E2 on the M1 vs M2 pair (recording exposure),
with the paper's crossed reciter/verse bootstrap and 95% CIs. A claim counts as supported
only if its CI excludes 0.

## Decision rules

- If M1 vs M2 shows no difference in E1 and E2 (both CIs include 0), the paper does NOT
  claim that exposure masks TSM artifacts; it reports the null and keeps only the
  artifact/TSM findings (C1, C3, C4).
- If E3 fails (RB contrast not positive), the paper does not claim an artifact-controlled
  local-lengthening penalty.
- All arms and all four models are reported regardless of sign.

## Stated limitations

Recipe is ours, not Tarteel's; no augmentation is used, so "exposure" here means exposure
to the recordings/voices, not to tempo augmentation. Voice exposure (M0 vs M1) is
confounded with training-set size. Text-based exclusion removes every row whose verse text
matches an evaluated verse, which for repeated verses removes more than the evaluated
recording itself.

## Amendment 1 (2026-09-19 23:35, before M1/M2 exist, no exposure result seen)

Two contributions are added now, both pre-registered, because the design already produces
the data they need.

**A. Artifact-corrected duration sensitivity (round-trip correction).**
Ochiai et al. (TASLP 2024) split enhancement damage into target and artifact parts by
BSS-Eval projection; that projection is undefined when the time axis changes. Replacement
that works for TSM: a ROUND-TRIP arm. Stretch by k with engine E, then stretch back by 1/k
with the same engine, giving the ORIGINAL duration and about twice the artifact exposure.
Let R_E(k) be its WER minus clean WER: artifact-only damage, duration held fixed. Define
artifact-corrected damage for each arm as observed damage minus lambda * R_E(k), with
lambda estimated per engine from the sham (k=1) and round-trip arms.
Pre-registered prediction A1: the corrected local-minus-global contrast AGREES between PV
and Rubber Band (their difference's 95% CI includes 0), while the uncorrected contrasts
disagree. A1 failing means the correction does not work and is reported as such.

**B. Time-stretch robustness as a contamination probe.**
Label-only membership inference (Choquette-Choo ICML 2021; Teixeira 2024 for ASR with
noise and PGD) says training members stay correct under content-preserving perturbations.
TSM has never been used as that perturbation, and never as an audit of a benchmark.
Probe score for a recording: s = WER(stretched) - WER(clean), for global TSM at each k.
Ground truth membership is known by construction: for M2 the evaluated recordings are
members; for M1 the voice is a member but the recording is not; for M0 neither.
Pre-registered predictions: B1 s separates M2-members from M1-non-members with ROC AUC
> 0.65 (bootstrap CI over recordings, reported whatever it is); B2 the probe is stronger
with the clean stretcher than with PV, because PV artifacts add variance; B3 applied to the
Tarteel checkpoint, whose training data is undocumented, the probe's score distribution is
compared with M2's and M0's, and the result is reported as an estimate with its CI, never
as proof of contamination.
If B1 fails, the paper reports the probe as unsuccessful and keeps A and the descriptive
exposure results.

## Amendment 2 (2026-09-21 11:20, round-trip data pulled, A1 NOT yet computed, no A1 number seen)

Operationalisation of "lambda estimated per engine from the sham (k=1) and round-trip arms",
fixed before computing A1. A round trip applies the engine twice (k up, then 1/k down), so the
artifact cost of ONE forward pass is estimated as half the round-trip damage above the k=1 sham:
  artifact_E(arm, k) = 0.5 * [ WER(arm_rt, k) - WER(sham_arm, 1) ]
with sham = sham_pv_local / sham_pv_global for PV and 0 for Rubber Band (its unity call returns
the input unchanged, so the sham is exactly the clean signal).
Corrected damage = [WER(arm, k) - clean] - artifact_E(arm, k); corrected contrast = local minus global.
A1 test, per model, k = 4: per-verse difference (corrected PV contrast - corrected RB contrast),
paper crossed bootstrap. A1 PASSES for a model if that CI includes 0 while the UNCORRECTED
difference's CI excludes 0. If the uncorrected difference already includes 0, A1 is "not testable"
for that model. Sensitivity (not the test): lambda = 1 (full round-trip cost).

## Amendment 3 (2026-09-21, after the review panel; NO k=2/k=6 Rubber Band result exists yet)

DEVIATION DISCLOSED: M1 and M2 were trained for 3 epochs (14,226 and 14,274 steps; selected at
step 14,000), not the 9,846 steps fixed above; train_exposure.py inherited epochs=3 from M0's
script and no max-steps cap was passed. M1 vs M2 remain matched (same steps, same selection step),
so the primary exposure tests are unaffected; M0 differs from M1/M2 in steps as well as data.

Rubber Band dose-response, pre-specified before running: for every model (Tarteel, M0, M1, M2),
add RB Local and Global arms at k=2 and k=6 (same pipeline, same duration matching, 29 s joint
exclusion) plus RB round trips at k=2 and k=6.
D1: Delta_RB(k=6) > 0 with CI excluding 0 in each of M0, M1, M2.
D2: dose-response, Delta_RB(6) - Delta_RB(2) > 0 (paired per verse, crossed bootstrap) in each of M0, M1, M2.
D3 (descriptive, no prediction): sign of Delta_RB(2).
D4: RB round-trip damage <= 0.05 WER in every arm at k=2 and k=6 (the check passes at every k).
All four reported whatever they show.

## Amendment 4 (2026-09-21 14:20 Africa/Casablanca). Confirmatory tests on NEW data, fixed BEFORE any of it exists

Motivation: reviewers noted (a) one domain and one model family, (b) the RB prediction E3 was fixed after
M0's RB result was known, (c) one training seed, (d) five confirmatory reciters. Every test below is fixed
before its data are produced or, for the CTC recogniser, before its saved scores are first analysed (the
pipeline has recorded CTC WER in every run since 19 Sep; no CTC contrast has been computed or viewed).
Everything is reported whatever it shows. Primary contrast throughout: Delta_RB(k=4) = WER(RB Local) -
WER(RB Global), equal-reciter/speaker mean, 95% bootstrap interval; "positive" means the interval excludes 0.

F1 Seeds. M1 and M2 are retrained with data-order seeds 2 and 3 (same recipe, three epochs, same validation
   subset and selection rule; only the batch order changes). Prediction F1: Delta_RB(4) positive in each of
   the six models (M1, M2 x seeds 1-3). Reported descriptively: seed spread of Delta_RB and clean WER, and the
   seed-averaged M2-minus-M1 differences compared with the seed spread.
F2 Fresh reciters. The three EveryAyah reciters excluded from every trained model (sahl_yassin,
   akram_alalaqimy, muhsin_al_qasim): 60 verse IDs each drawn from the same 28-surah candidate pool with
   numpy seed 20260921, eligibility and pipeline unchanged. Prediction F2: Delta_RB(4) positive in each of M0,
   M1, M2 (and reported for Tarteel), with the bootstrap over reciters and verses.
F3 CTC recogniser (rabah2026 wav2vec2, also the aligner, stated as a confound). Prediction F3a: Delta_RB(4)
   positive; F3b: Delta_RB(6) - Delta_RB(2) positive.
F4 Whisper-large-v3 (openai, not Qur'an-tuned), same pipeline and decoding. Gate: clean WER <= 0.50, else
   uninformative. Prediction F4: Delta_RB(4) positive.
F5 English read speech, LibriSpeech test-clean (speakers disjoint from LibriSpeech training). Targets:
   vowels with primary stress (ARPAbet suffix 1) from Montreal Forced Aligner alignments, >= 60 ms, >= 2 per
   utterance, utterances <= 8 s; 15 utterances from each of 20 speakers drawn with numpy seed 20260921.
   Same arms and engines (PV, RB), k in {2, 4, 6}, round trips at k=4. Recognisers: facebook/wav2vec2-base-960h
   (trained on LibriSpeech training speakers only) and openai/whisper-base. English WER after lower-casing and
   removing punctuation. Predictions F5a: Delta_RB(4) positive for both recognisers; F5b: RB passes the
   round-trip check (upper bounds < 0.05) at k=4; F5c: Delta_RB(2) - Delta_PV(2) positive (PV bias).
   Bootstrap: resample speakers, then utterances within speaker (utterances are not shared across speakers).
M  Mechanism (analysis of existing data, fixed before computing): across the 8 cells (4 models x k in {2,4}),
   the PV bias Delta_RB - Delta_PV increases with the round-trip artifact imbalance A_PV(Global) - A_PV(Local)
   (Spearman rho > 0 reported with its value; 8 cells, descriptive).
