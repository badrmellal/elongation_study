"""Pre-registered contamination probe (exposure/PROTOCOL.md Amendment 1B). s = WER(stretched) - WER(clean).
Members: verses whose recording for that reciter is in EveryAyah train (heldout_check/split_membership.json);
member under M2, non-member under M1 (rows dropped by text), same recordings."""
import json, collections, numpy as np
mem = json.load(open("heldout_check/split_membership.json"))["study"]
def load(m):
    d = collections.defaultdict(dict)
    for l in open(f"exposure/results/{m}/observations.jsonl"):
        r = json.loads(l)
        if r["kind"] == "condition":
            d[(r["reciter"], r["verse"])][(r["arm"], r["k"])] = r["whisper"]["wer"]
    return d
D = {m: load(m) for m in ["tarteel", "M0", "M1", "M2"]}
members = [(n, v) for n, info in mem.items() for v, splits in info["per_verse"].items() if "train" in splits]
print("member recordings (in EveryAyah train):", len(members), "of", sum(len(i["per_verse"]) for i in mem.values()))
def auc(pos, neg):  # P(score_pos > score_neg), ties 0.5; score = -s (more robust = higher)
    pos, neg = np.asarray(pos), np.asarray(neg)
    return float(((pos[:, None] > neg[None, :]).mean() + 0.5 * (pos[:, None] == neg[None, :]).mean()))
rng = np.random.default_rng(20260921); out = {}
for arm, k in [("global", 4.0), ("global_rb", 4.0), ("global", 2.0)]:
    keys = [key for key in members if all((arm, k) in D[m].get(key, {}) and ("baseline", 1.0) in D[m].get(key, {}) for m in ["M1", "M2"])]
    s = {m: np.array([D[m][key][(arm, k)] - D[m][key][("baseline", 1.0)] for key in keys]) for m in D if all(key in D[m] and (arm, k) in D[m][key] for key in keys)}
    a = auc(-s["M2"], -s["M1"])
    boots = []
    for _ in range(2000):
        idx = rng.integers(len(keys), size=len(keys)); boots.append(auc(-s["M2"][idx], -s["M1"][idx]))
    lo, hi = np.percentile(boots, [2.5, 97.5])
    out[f"{arm}@{k:g}"] = {"n": len(keys), "auc_M2member_vs_M1": a, "ci95": [float(lo), float(hi)],
                          "mean_s": {m: float(v.mean()) for m, v in s.items()}}
    print(f"{arm}@{k:g}: n={len(keys)} AUC(M2 members vs M1) = {a:.3f} [{lo:.3f},{hi:.3f}]  mean s: " +
          ", ".join(f"{m} {v.mean():+.3f}" for m, v in s.items()))
json.dump(out, open("exposure/probe_results.json", "w"), indent=1)
