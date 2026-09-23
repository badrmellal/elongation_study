"""Tables for recast v2 (9 pt minimum): Table 1 arm-level WER at k=4; Table 2 pre-specified tests.
Reads figs/numbers.json, figs/arm_table.json, probe_results.json, figs/dose.json (if present)."""
import json, pathlib
H = pathlib.Path(__file__).resolve().parent
N = json.load(open(H / "figs" / "numbers.json")); S, T = N["models"], N["tests"]
AT = json.load(open(H / "figs" / "arm_table.json"))["arm_table"]
probe = json.load(open(H / "probe_results.json"))
dose = json.load(open(H / "figs" / "dose.json")) if (H / "figs" / "dose.json").exists() else None
LAB = {"tarteel": "Tarteel", "M0": "M0 (none)", "M1": "M1 (voices)", "M2": "M2 (recordings)"}
num = lambda v, d=3: f"${v:+.{d}f}$".replace("-", "{-}")
ci = lambda x, d=2: f"{num(x[0], d)} [{num(x[1], 2)}, {num(x[2], 2)}]"
rows = "\n".join(f"{LAB[m]} & {AT[m]['baseline']:.3f} & {AT[m]['nucleus']:.3f} & {AT[m]['global']:.3f} & "
                 f"{AT[m]['nucleus_rb']:.3f} & {AT[m]['global_rb']:.3f} & {AT[m]['silence']:.3f} \\\\" for m in LAB)
(H / "figs" / "table_arms.tex").write_text(r"""\begin{table}[t]
\centering\setlength{\tabcolsep}{3.2pt}
\caption{WER by arm at $k=4$, equal-reciter mean over five confirmatory reciters.
Models by what they heard in training: M0 none of the evaluated reciters, M1 their voices,
M2 also 167 of the 210 evaluated recordings; Tarteel is the public checkpoint with
undocumented training. PV: phase vocoder; RB: Rubber Band; L: local; G: global.
Contrasts with intervals are in Fig.~\ref{fig:models}.}
\label{tab:arms}
\begin{tabular}{lcccccc}
\toprule
 & & \multicolumn{2}{c}{PV} & \multicolumn{2}{c}{RB} & \\
\cmidrule(lr){3-4}\cmidrule(lr){5-6}
Model & Clean & L & G & L & G & Silence \\
\midrule
""" + rows + r"""
\bottomrule
\end{tabular}
\end{table}
""")
ok = lambda x: r"$\checkmark$" if x[1] > 0 else r"$\times$"
own = ["M0", "M1", "M2"]
rb_all = all(S[m]["rb"][1] > 0 for m in own)
rng = lambda key: f"{num(min(S[m][key][0] for m in own), 2)} to {num(max(S[m][key][0] for m in own), 2)}"
pr = [
 (r"Does the PV contrast hold on unseen voices? ($\Delta_{\mathrm{PV}}{>}0$, M0)", ci(S["M0"]["pv"]), ok(S["M0"]["pv"])),
 (r"Does hearing the recordings reduce global-arm PV damage? (M1 $>$ M2)", ci(T["E1"]), ok(T["E1"])),
 (r"Does hearing the recordings enlarge $\Delta_{\mathrm{PV}}$? (M2 $>$ M1)", ci(T["E2"]), ok(T["E2"])),
 (r"Is $\Delta_{\mathrm{RB}}$ positive at every exposure? (M0--M2)", rng("rb") + (r", all CIs $>0$" if rb_all else ""), r"$\checkmark$" if rb_all else r"$\times$"),
 (r"Does hearing the recordings lower clean WER? (M1 $>$ M2)", f"{num(T['E4'][0],3)} [{num(T['E4'][1],3)}, {num(T['E4'][2],3)}]", ok(T["E4"])),
 (r"Does stretch robustness reveal heard recordings? (AUC $>0.65$)", f"AUC {probe['global@4']['auc_M2member_vs_M1']:.2f} [{probe['global@4']['ci95'][0]:.2f}, {probe['global@4']['ci95'][1]:.2f}]", r"$\times$"),
 (r"Does subtracting round-trip damage align PV with RB?", "not testable", "--"),
]
if dose:
    d1 = all(dose["six"][m][1] > 0 for m in own); d2 = all(dose["growth"][m][1] > 0 for m in own); d4 = dose["rt_max_hi"] < 0.05
    pr += [(r"Is $\Delta_{\mathrm{RB}}$ positive at $k{=}6$? (M0--M2)", f"{num(min(dose['six'][m][0] for m in own),2)} to {num(max(dose['six'][m][0] for m in own),2)}, all CIs $>0$", r"$\checkmark$" if d1 else r"$\times$"),
           (r"Does $\Delta_{\mathrm{RB}}$ grow from $k{=}2$ to $k{=}6$? (M0--M2)", f"{num(min(dose['growth'][m][0] for m in own),2)} to {num(max(dose['growth'][m][0] for m in own),2)}, min.\\ bound {num(min(dose['growth'][m][1] for m in own),3)}", r"$\checkmark$" if d2 else r"$\times$"),
           (r"Does RB pass the round-trip check at $k{=}2,6$? (all bounds $<0.05$)", f"max bound {dose['rt_max_hi']:.3f} (M0 global, $k{{=}}6$)", r"$\checkmark$" if d4 else r"$\times$")]
(H / "figs" / "table_prereg.tex").write_text(r"""\begin{table}[t]
\centering\setlength{\tabcolsep}{4pt}
\caption{Pre-specified tests, reported regardless of outcome. Paired per verse, 95\% crossed-bootstrap
intervals (the AUC interval is a clip bootstrap); $\checkmark$: pre-specified criterion met.
Models as in Table~\ref{tab:arms}.}
\label{tab:prereg}
\begin{tabular}{@{}>{\raggedright\arraybackslash}p{4.3cm}>{\raggedright\arraybackslash}p{2.6cm}c@{}}
\toprule
Question (prediction) & Estimate [95\% CI] & Met \\
\midrule
""" + "\n".join(f"{a} & {b} & {c} \\\\" for a, b, c in pr) + r"""
\bottomrule
\end{tabular}
\end{table}
""")
print("tables written; dose rows:", bool(dose))
