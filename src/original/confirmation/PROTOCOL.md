# Prospective confirmation protocol

Written 2026-09-11 before new-reciter inference. This is a local timestamped
protocol, not an externally registered study. Amendments must be dated and
retain their reason; do not overwrite hypotheses after observing outcomes.

## Population and sampling

Confirmation reciters: Alafasy (128 kbps source folder), As-Sudais, Ash-Shuraym,
Ad-Dussary, and Hani ar-Rifai. These are new to this study's analysis, not proven
unseen in pretrained ASR training. Do not count Alafasy's second bitrate as an
independent reciter. Existing Abdul Basit/Husary/Minshawy results remain
exploratory; use original-reciter audio only for development and reproduction.

Retain original 28-surah candidate list and <=8 s source-duration filter.
Use only verse IDs present and duration-eligible in all five new reciters (214
at inventory). Select 60 unique candidates with NumPy RNG seed 20270911 from
numerically sorted IDs, before alignment or ASR. Each reciter is processed on
those candidates. After CTC alignment require at least two intervals >=60 ms.
Record all exclusions; no replacement sampling and no outcome-dependent stop.
Any transformed audio >29 s excludes that verse at that factor across all
arms, not just whichever arm fails. Never truncate a long waveform to fit.

## Frozen models and inference

Whisper: tarteel-ai/whisper-base-ar-quran,
revision 5c3c53fdf9272c4f6ee0bee09a1e5a4a615ee25c.
Alignment/CTC: rabah2026/wav2vec2-large-xlsr-53-arabic-quran-v2,
revision 98c046d4f5fd74ed4c5a7078f5129fd85fb53d0c.
Use eval mode, float32, eager attention, deterministic greedy decoding,
max_new_tokens=180. Retain the checkpoint's language/task prefix and record it.
Report token-cap hits rather than silently treating them as successful endings.
Record actual package versions, hardware/device, source hashes, file hashes,
generation settings, and raw reference/hypothesis for every condition.

Attention primary sensitivity: fix query rows using the gold-text word
containing each orthographic madd character, independent of observed attention.
Shift teacher forcing correctly: a query is the decoder row predicting the
associated token, not the row after ingesting that token. Average all query rows
whose token offsets overlap that word. Save deepest-layer mass normalized over
valid non-padding frames, interval fraction, enrichment, and odds. Also save
post-hoc max-query values as a labeled sensitivity, not the primary estimand.
CTC-derived intervals remain model estimates, not manual vowel boundaries.

## Conditions

Baseline k=1; local, global, silence, and endpoint-blended local at k=2,4,6.
Added sample counts are exactly matched across arms at each factor. Endpoint
blend width is min(10 ms, one quarter of the original interval), restoring
original local boundary samples while preserving total duration. It is a
sensitivity test, not a guarantee that all vocoder artifacts disappear.

Unity-rate local and global shams force STFT/vocoder/ISTFT processing, unlike
the original early return at k=1. At k=4, add local and global Rubber Band
transformations with identical backend settings. Rubber Band must pass exact
length/timeline smoke tests before confirmation. Record any backend failures;
do not select whichever backend gives the stronger effect.

## Predictions and decision rules

Primary recognition contrast: per-verse WER(local PV k=4) - WER(global PV k=4).
Directional prediction is positive, with a planning effect of +0.10 absolute
WER, based on exploratory contrasts. Report estimate and 95% verse-paired
bootstrap interval for each of five reciters; Holm-correct the five two-sided
paired randomization tests as one family. Show all five, regardless of sign.
Report equal-reciter mean descriptively and uncertainty with both verse and
reciter resampling; five voices do not establish population universality.

Artifact knockouts: repeat local-global at k=4 with endpoint blending and
Rubber Band. If the sign changes, the effect disappears, or shams themselves
materially damage recognition, do not attribute a PV-only contrast to local
duration alone. These are sensitivity analyses, not independent replications.

Attention prediction: fixed-query within-occurrence change in log odds divided
by change in log interval duration is below the proportional-reference value
one. Compare every interval to its own baseline; report the arbitrary-model
assumption behind that reference. Also report enrichment against interval
fraction. Bounded raw mass alone is not evidence of a new saturation law.
No causal claim that attention allocation causes WER. Do not pool intervals
as independent replicates or bootstrap individual frames.

Other factors k=2,6, CTC recognition, query variants, and natural-style results
are explicitly secondary/exploratory. Use 10,000 bootstrap/permutation draws,
seed 20270911, save analysis code and all inclusion counts.

## Gates and reporting

Inspect raw prediction versus reference and audio/interval diagnostics on
prespecified original-reciter pilot samples. Reproduce the old decoder's
baseline WER for an overlapping stored verse before scaling; compare CPU/MPS
prediction parity if accelerator is used. Training/convergence gates do not
apply to these frozen inference-only models. No timing-performance claim.

Manual phonetic validation requires a qualified human's annotations. Diagnostic
plots or another model do not substitute for that validation. Retain this
limitation unless such annotations are actually provided.

## Amendment 1: original-reciter pilot, before confirmation

2026-09-11: the CPU and MPS pilot on Abdul Basit 67:14 reproduced saved WER 1/7,
but the actual Arabic transcript was perfect. The error was language/task
control tokens retained by this checkpoint's decoder despite
`skip_special_tokens=True`. No baseline in the old three-reciter files had zero
WER. Those files save no hypotheses, so their WER cannot be repaired reliably
without inference. Explicitly filter known control token IDs before decoding;
keep raw generated IDs. Set the same Arabic/transcribe/no-timestamps prefix
for teacher forcing and free decoding. Request generation output retaining EOS
to distinguish normal completion from token-cap truncation.

Rerun original three reciters on their original successfully processed verse
sets (35/39/36), clearly exploratory, using the corrected runner and all new
controls. This avoids presenting old scoring artifacts as recognition errors.
The five-reciter candidate sampling and primary decision rules above remain
unchanged. Freeze this amendment before any confirmation-reciter inference.

## Amendment 2: valid attention reference, before confirmation

2026-09-11: adversarial mathematical review shows that the unit-exponent
reference is not valid for the proposed multi-interval, head-averaged statistic.
Even fixed per-frame scores give sublinear raw mass from softmax normalization;
simultaneously extending other intervals also makes focal odds sublinear.
Taking odds after averaging heterogeneous heads or queries adds another such
effect. Therefore the preceding attention exponent prediction is retired before
new-reciter inference, rather than tested against an invalid null.

Replace it with a within-occurrence relative-density contrast. For each fixed
gold-word query and head, calculate log(target mass / target frame count) minus
log(reference mass / reference frame count). Reference support excludes all
manipulated intervals, padding, and a one-frame (20 ms) boundary guard. Require
at least ten reference frames. Compute log ratios before averaging heads and
queries. Save paired changes relative to baseline, aggregate intervals within
verse, then verses within reciter. Reference zero means unchanged relative
mean frame scores; a negative change is the directional planning prediction.
This cancels the shared softmax denominator, not contextual acoustic changes,
and cannot establish that attention causes recognition errors. Floor masses at
1e-12 only for logarithms and report every floor event and minimum mass.

Retain raw/max-query masses and enrichment only as descriptive sensitivities.
Recognition's primary contrast, sampling, and artifact controls are unchanged.
No new attention result was inspected before this amendment. The original
three-reciter pilot preceded it and remains development data.
