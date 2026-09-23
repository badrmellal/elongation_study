"""Fine-tune openai/whisper-base on EveryAyah rows from the 24 reciters NOT in common.EXCLUDED.
Same tokenizer files and decoder prefix (<|ar|><|transcribe|><|notimestamps|>) as the paper's
Tarteel checkpoint. Checkpoint selection uses validation rows of allowed reciters only;
the paper's evaluation clips are never read here."""
import argparse, json, random, shutil, threading, queue, time
import numpy as np, torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from common import PREP, PROC, RUNS, EXCLUDED, norm_ar, lev

ap = argparse.ArgumentParser()
ap.add_argument("--run", default="wb_unseen_v1")
ap.add_argument("--epochs", type=float, default=3.0)
ap.add_argument("--bs", type=int, default=32, help="effective batch per optimizer step")
ap.add_argument("--accum", type=int, default=2, help="micro-batches per step (VRAM limit on 16 GB)")
ap.add_argument("--lr", type=float, default=1e-5)
ap.add_argument("--warmup", type=int, default=500)
ap.add_argument("--eval-every", type=int, default=2000)
ap.add_argument("--val-n", type=int, default=800)
ap.add_argument("--seed", type=int, default=20260919)
ap.add_argument("--max-steps", type=int, default=0, help="smoke test only")
a = ap.parse_args()
random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
out = RUNS / a.run; out.mkdir(parents=True, exist_ok=True)
log = open(out / "log.jsonl", "a", encoding="utf-8")
def emit(**kw):
    kw["t"] = round(time.time(), 1); log.write(json.dumps(kw, ensure_ascii=False) + "\n"); log.flush(); print(kw, flush=True)

proc = WhisperProcessor.from_pretrained(PROC)
tok, fe = proc.tokenizer, proc.feature_extractor
tok.set_prefix_tokens(language="ar", task="transcribe", predict_timestamps=False)
SOT = tok.convert_tokens_to_ids("<|startoftranscript|>")

train, val, arrays = [], [], {}
for j in sorted(PREP.glob("*.json")):
    idx = json.load(open(j, encoding="utf-8"))
    arrays[j.stem] = np.load(j.with_suffix(".npy"), mmap_mode="r")
    split = "train" if idx["source"].startswith("train") else "val"
    for off, n, text, rec in idx["rows"]:
        assert rec not in EXCLUDED
        ids = tok(text).input_ids
        if len(ids) > 448:
            continue
        (train if split == "train" else val).append((j.stem, off, n, ids, text, rec))
val_sub = random.Random(a.seed).sample(val, min(a.val_n, len(val)))
emit(event="data", train=len(train), val=len(val), val_sub=len(val_sub),
     train_reciters=len({r[5] for r in train}), val_reciters=len({r[5] for r in val_sub}))

def audio(item):
    stem, off, n = item[:3]
    return np.asarray(arrays[stem][off:off + n], dtype=np.float32) / 32768.0

def feats(items):
    x = fe([audio(it) for it in items], sampling_rate=16000, return_tensors="pt", device="cuda").input_features
    return x.to("cuda", non_blocking=True)

def labels(items):
    seqs = [it[3][1:] if it[3][0] == SOT else it[3] for it in items]
    L = max(map(len, seqs)); lab = torch.full((len(seqs), L), -100, dtype=torch.long)
    for i, s in enumerate(seqs):
        lab[i, :len(s)] = torch.tensor(s)
    return lab.to("cuda", non_blocking=True)

model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-base").to("cuda")
model.config.forced_decoder_ids = None; model.generation_config.forced_decoder_ids = None
model.config.use_cache = False

@torch.no_grad()
def evaluate(step):
    model.eval(); errs = words = 0; samples = []
    for i in range(0, len(val_sub), 16):
        chunk = val_sub[i:i + 16]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            gen = model.generate(input_features=feats(chunk), language="ar", task="transcribe",
                                 max_new_tokens=180, num_beams=1, do_sample=False)
        for it, hyp in zip(chunk, tok.batch_decode(gen, skip_special_tokens=True)):
            r, h = norm_ar(it[4]).split(), norm_ar(hyp).split()
            errs += lev(r, h); words += len(r)
            if len(samples) < 3: samples.append([it[4][:60], hyp[:60]])
    model.train(); wer = errs / max(words, 1)
    emit(event="eval", step=step, val_wer=round(wer, 4), samples=samples)
    return wer

steps_per_epoch = len(train) // a.bs
total = a.max_steps or int(steps_per_epoch * a.epochs)
opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.0, fused=True)
sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / a.warmup) * max(0.0, (total - s) / max(1, total - a.warmup)) if s >= a.warmup else (s + 1) / a.warmup)

def batches():
    rng = random.Random(a.seed + 1); order = []
    while True:
        if not order:
            order = list(range(len(train))); rng.shuffle(order)
        mb = a.bs // a.accum
        b = [train[order.pop()] for _ in range(min(mb, len(order)))]
        if len(b) == mb:
            yield b
q = queue.Queue(maxsize=8)
def producer():
    for b in batches():
        q.put(([audio(it) for it in b], b))
threading.Thread(target=producer, daemon=True).start()

emit(event="start", total_steps=total, steps_per_epoch=steps_per_epoch, **vars(a))
best = evaluate(0) if not a.max_steps else 9.9
model.train(); t0 = time.time(); run_loss = 0.0
for step in range(1, total + 1):
    for _ in range(a.accum):
        auds, b = q.get()
        x = fe(auds, sampling_rate=16000, return_tensors="pt", device="cuda").input_features.to("cuda", non_blocking=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = model(input_features=x, labels=labels(b)).loss / a.accum
        loss.backward()
        run_loss += loss.item()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
    if step % 100 == 0:
        emit(event="train", step=step, loss=round(run_loss / 100, 4), lr=sched.get_last_lr()[0],
             steps_per_s=round(step / (time.time() - t0), 2), gpu_mem_gb=round(torch.cuda.max_memory_allocated() / 1e9, 2))
        run_loss = 0.0
    if step % a.eval_every == 0 or step == total:
        wer = evaluate(step)
        if wer < best:
            best = wer; model.save_pretrained(out / "best")
            for f in PROC.iterdir(): shutil.copy(f, out / "best" / f.name)
            emit(event="saved_best", step=step, val_wer=round(wer, 4))
model.save_pretrained(out / "last")
for f in PROC.iterdir(): shutil.copy(f, out / "last" / f.name)
emit(event="done", best_val_wer=round(best, 4), minutes=round((time.time() - t0) / 60, 1))
