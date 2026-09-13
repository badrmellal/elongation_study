"""Generate paper plots and numerical LaTeX fields from validated confirmation."""

from pathlib import Path
import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np

HERE = Path(__file__).resolve().parent
DATA = HERE.parents[1] / "confirmation" / "analysis.json"
NEW = ["Alafasy", "Sudais", "Shuraym", "Dussary", "Rifai"]
OLD = ["AbdulBasit", "Husary", "Minshawy"]
ORDER = OLD + NEW
NAMES = {"AbdulBasit": "Abdul Basit", "Husary": "Husary", "Minshawy": "Minshawy",
         "Alafasy": "Alafasy", "Sudais": "As-Sudais", "Shuraym": "Ash-Shuraym",
         "Dussary": "Ad-Dussary", "Rifai": "Hani ar-Rifai"}
BLUE, ORANGE, GREEN, GRAY = "#0072B2", "#D55E00", "#009E73", "#666666"
plt.rcParams.update({"font.family": "serif", "font.size": 9.5, "axes.labelsize": 9.5,
    "axes.titlesize": 10, "xtick.labelsize": 9.5, "ytick.labelsize": 9.5,
    "legend.fontsize": 9.5, "pdf.fonttype": 42, "ps.fonttype": 42,
    "axes.spines.top": False, "axes.spines.right": False})


def write_figure(fig, stem):
    target = HERE / "figs"
    target.mkdir(exist_ok=True)
    fig.savefig(target / f"{stem}.pdf", metadata={"Title": stem, "Author": "Badr Mellal"})
    fig.savefig(target / f"{stem}.png", dpi=400, facecolor="white")
    plt.close(fig)


def schematic():
    fig, ax = plt.subplots(figsize=(3.4, 1.7), layout="constrained")
    parts = [("Baseline", [(0,1),(1,1),(0,1),(1,1),(0,1)]),
             ("Local", [(0,1),(1,2),(0,1),(1,2),(0,1)]),
             ("Global", [(0,1.4),(1,1.4),(0,1.4),(1,1.4),(0,1.4)]),
             ("Silence", [(0,1),(1,1),(2,1),(0,1),(1,1),(2,1),(0,1)])]
    colors = {0: "#D9E5EC", 1: ORANGE, 2: "white"}
    for y, (label, segments) in zip([3,2,1,0], parts):
        x = 0
        for kind, width in segments:
            ax.add_patch(Rectangle((x,y),width,.52,facecolor=colors[kind],edgecolor=GRAY,lw=.55,
                                   hatch="///" if kind==2 else None))
            x += width
        ax.text(-.2,y+.25,label,ha="right",va="center",fontsize=9.5)
    ax.set_xlim(-1.65,7.1); ax.set_ylim(-.25,3.75); ax.axis("off")
    write_figure(fig,"fig1_conditions")


def primary_figure(data):
    fig, axes = plt.subplots(1,2,figsize=(7.0,2.85),layout="constrained",sharey=True)
    ys=np.arange(len(ORDER))[::-1]
    for ax, model, title in zip(axes,["whisper","ctc"],["(a) Whisper","(b) wav2vec2 CTC (secondary)"]):
        for y,name in zip(ys,ORDER):
            cell=data["models"][model][name]["local_global"]["4.0"]
            lo,hi=cell["ci95"]; mean=cell["mean"]
            ax.plot([lo,hi],[y,y],color=BLUE,lw=1.2)
            ax.plot(mean,y,"o",ms=4,color=BLUE,mfc=BLUE if name in NEW else "white")
        ax.axvline(0,color=GRAY,lw=.7,ls=":")
        ax.axhline(4.5,color="#cccccc",lw=.6)
        ax.set_title(title,loc="left");ax.set_xlabel("WER(local) - WER(global), k=4")
        ax.grid(axis="x",alpha=.12)
    axes[0].set_yticks(ys,[NAMES[n] for n in ORDER])
    write_figure(fig,"fig2_reciters")


def sensitivity_figure(data):
    fig, axes=plt.subplots(1,2,figsize=(7.0,2.65),layout="constrained")
    ys=np.arange(5)[::-1]
    styles=[("local_global","PV",BLUE,-.2,"o"),("edge_global_k4","PV + edge",ORANGE,0,"^"),
            ("rubberband_local_global_k4","Rubber Band",GREEN,.2,"s")]
    for key,label,color,offset,marker in styles:
        for y,name in zip(ys,NEW):
            cell=data["models"]["whisper"][name][key]
            if key=="local_global":cell=cell["4.0"]
            lo,hi=cell["ci95"]
            axes[0].plot([lo,hi],[y+offset]*2,color=color,lw=.7)
            axes[0].plot(cell["mean"],y+offset,marker,ms=4.2,color=color,mec="#222222",mew=.35)
        axes[0].plot([],[],marker,ms=4.2,color=color,mec="#222222",mew=.35,label=label)
    axes[0].set_title("(a) Backend and boundary sensitivity",loc="left")
    axes[0].set_xlabel("Whisper WER contrast, k=4")
    axes[0].set_yticks(ys,[NAMES[n] for n in NEW])
    axes[0].legend(loc="upper left",bbox_to_anchor=(0,-.22),ncol=2,frameon=False,columnspacing=.6,handletextpad=.25)
    for offset,arm,label,color,marker in [(-.12,"nucleus","Local",BLUE,"o"),(.12,"global","Global",GRAY,"s")]:
        for y,name in zip(ys,NEW):
            cell=data["attention"][name][f"{arm}@4"]
            lo,hi=cell["ci95"]
            axes[1].plot([lo,hi],[y+offset]*2,color=color,lw=.8)
            axes[1].plot(cell["mean"],y+offset,marker,ms=4.2,color=color,mec="#222222",mew=.35)
        axes[1].plot([],[],marker,ms=4.2,color=color,mec="#222222",mew=.35,label=label)
    axes[1].set_title("(b) Fixed-query relative attention",loc="left")
    axes[1].set_xlabel("Change in log frame-density ratio")
    axes[1].set_yticks(ys,[NAMES[n] for n in NEW])
    axes[1].legend(loc="upper left",bbox_to_anchor=(0,-.22),ncol=2,frameon=False,columnspacing=.8)
    for ax in axes:
        ax.axvline(0,color=GRAY,lw=.7,ls=":");ax.grid(axis="x",alpha=.12)
    write_figure(fig,"fig3_controls")


def numbers(data):
    macros={"TotalVerses":sum(c["included"] for c in data["coverage"].values()),
            "NewVerses":sum(data["coverage"][n]["included"] for n in NEW),
            "Conditions":data["quality"]["conditions"],"CapHits":data["quality"]["token_cap_hits"],
            "ExcludedCandidates":data["quality"]["excluded_candidates"],
            "PrimaryPositive":sum(data["models"]["whisper"][n]["local_global"]["4.0"]["mean"]>0 for n in NEW),
            "PrimarySignificant":sum(data["models"]["whisper"][n]["local_global"]["4.0"]["p_holm"]<.05 for n in NEW)}
    aggregate=data["primary_equal_reciter_mean"]
    interval = aggregate["ci95"]
    macros.update({"MeanContrast":f"{aggregate['mean']:.3f}",
                   "MeanLower":f"{interval[0]:.3f}" if interval is not None else "NA",
                   "MeanUpper":f"{interval[1]:.3f}" if interval is not None else "NA"})
    lines=[f"\\newcommand{{\\{key}}}{{{value}}}" for key,value in macros.items()]
    (HERE/"result_numbers.tex").write_text("% Generated from validated confirmation/analysis.json\n"+"\n".join(lines)+"\n")
    table=[]
    for n in ORDER:
        c=data["coverage"][n]
        w=data["models"]["whisper"][n]["means"]["baseline@1"]["mean"]
        t=data["models"]["ctc"][n]["means"]["baseline@1"]["mean"]
        table.append(f"{NAMES[n]} & {c['candidates']} & {c['included']} & {w:.3f} & {t:.3f} \\\\")
        if n==OLD[-1]:table.append("\\midrule")
    (HERE/"sample_table.tex").write_text("\\newcommand{\\SampleTableRows}{%\n"+"\n".join(table)+"\n}\n")
    detailed=[]
    for n in NEW:
        c=data["models"]["whisper"][n]["local_global"]["4.0"]
        pvalue = "$<0.001$" if c["p_holm"] < .001 else f"{c['p_holm']:.3f}"
        detailed.append(f"{NAMES[n]} & {c['n']} & {c['mean']:+.3f} & [{c['ci95'][0]:+.3f}, {c['ci95'][1]:+.3f}] & {pvalue} \\\\")
    (HERE/"primary_table.tex").write_text("\\newcommand{\\PrimaryTableRows}{%\n"+"\n".join(detailed)+"\n}\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis", type=Path,
                        default=DATA if DATA.exists() else HERE / "analysis.json")
    args = parser.parse_args()
    data=json.loads(args.analysis.read_text())
    schematic();primary_figure(data);sensitivity_figure(data);numbers(data)


if __name__=="__main__":main()
