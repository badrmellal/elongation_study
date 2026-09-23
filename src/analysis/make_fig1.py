"""Fig. 1 (duration-matched arms at k=2), drawn at its printed size so text prints at 9.5 pt (ICASSP minimum 9 pt).
Same schematic as delivery_20260913/latex/make_figures.py:schematic(), resized; Times to match Fig. 2."""
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
OUT = Path.home() / "Downloads/ICASSP2027_Mellal_recast/figs"
plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Times", "STIX Two Text", "DejaVu Serif"],
                     "font.size": 9.5, "pdf.fonttype": 42, "ps.fonttype": 42})
ORANGE, GRAY = "#D55E00", "#666666"
fig, ax = plt.subplots(figsize=(2.5, 0.85))
fig.subplots_adjust(0, 0, 1, 1)
parts = [("Baseline", [(0, 1), (1, 1), (0, 1), (1, 1), (0, 1)]),
         ("Local", [(0, 1), (1, 2), (0, 1), (1, 2), (0, 1)]),
         ("Global", [(0, 1.4), (1, 1.4), (0, 1.4), (1, 1.4), (0, 1.4)]),
         ("Silence", [(0, 1), (1, 1), (2, 1), (0, 1), (1, 1), (2, 1), (0, 1)])]
colors = {0: "#D9E5EC", 1: ORANGE, 2: "white"}
for y, (label, segments) in zip([3, 2, 1, 0], parts):
    x = 0
    for kind, width in segments:
        ax.add_patch(Rectangle((x, y), width, .62, facecolor=colors[kind], edgecolor=GRAY, lw=.55, hatch="///" if kind == 2 else None))
        x += width
    ax.text(-.2, y + .31, label, ha="right", va="center", fontsize=9.5)
ax.set_xlim(-2.1, 7.05); ax.set_ylim(-.12, 3.74); ax.axis("off")
fig.savefig(OUT / "fig1_conditions.pdf", metadata={"Title": "fig1_conditions", "Author": "Badr Mellal"})
fig.savefig(OUT / "fig1_conditions.png", dpi=300, facecolor="white")
print("written", OUT / "fig1_conditions.pdf")
