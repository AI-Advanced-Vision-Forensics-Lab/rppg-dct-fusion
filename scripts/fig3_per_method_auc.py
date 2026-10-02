#!/usr/bin/env python3
"""
fig3_per_method_auc.py — Figure 3: per-method AUC for rPPG (1D ResNet), lip-region DCT
and Concat fusion. Left: per-method training. Right: leave-one-generator-out.

Plotting only: reads the shipped result JSONs, trains nothing.
  rPPG, per-method   : results/reruns/r1_r2_rppg_resnet/r1_results.json
  DCT, per-method    : results/6_29_2026/dct_only_experiments.json
  Concat, per-method : results/6_29_2026/fusion_experiments.json
  LOGO (all three)   : results/reruns/r3_lomo/results.json
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"]  = 42
matplotlib.rcParams["font.family"]  = "DejaVu Sans"
import matplotlib.pyplot as plt
import numpy as np

ap = argparse.ArgumentParser(description="Figure 3: per-method AUC bars (plotting only)")
ap.add_argument("--results-dir", default="results", help="folder with the shipped result JSONs (default: ./results)")
ap.add_argument("--out-dir", default="figures", help="folder for the figure (default: ./figures)")
args = ap.parse_args()

RES = Path(args.results_dir)
OUT = Path(args.out_dir)

METHODS = ["AniTalker", "EchoMimic", "EDTalk", "FLOAT", "IP_LAP", "Real3DPortrait", "SadTalker"]
LABEL   = {m: m.replace("IP_LAP", "IP-LAP") for m in METHODS}


def load(rel):
    with open(RES / rel) as f:
        return json.load(f)


r1   = load("reruns/r1_r2_rppg_resnet/r1_results.json")
dct  = load("6_29_2026/dct_only_experiments.json")
cat  = load("6_29_2026/fusion_experiments.json")
logo = load("reruns/r3_lomo/results.json")

in_domain = {m: {"rPPG":   r1[m]["mean_auc"],
                 "DCT":    dct[m]["mean_auc"],
                 "Concat": cat[m]["mean_auc"]} for m in METHODS}
held_out  = {m: {"rPPG":   logo[m]["rPPG-only"]["mean_auc"],
                 "DCT":    logo[m]["DCT-lip"]["mean_auc"],
                 "Concat": logo[m]["Concat-lip"]["mean_auc"]} for m in METHODS}

# Rows ordered top to bottom by per-method rPPG AUC (descending)
order = sorted(METHODS, key=lambda m: in_domain[m]["rPPG"], reverse=True)

STYLE = {
    "rPPG":   dict(color="0.6", edgecolor="black", hatch=""),
    "DCT":    dict(color="white", edgecolor="black", hatch="////"),
    "Concat": dict(color="black", edgecolor="black", hatch=""),
}
BAR_H = 0.26
FONT  = 7

fig, axes = plt.subplots(1, 2, figsize=(3.4, 2.3), sharey=True,
                         gridspec_kw={"wspace": 0.18})
y = np.arange(len(order))[::-1]  # first method at the top

for ax, data, title in [(axes[0], in_domain, "Per-method training"),
                        (axes[1], held_out,  "Leave-one-generator-out")]:
    for k, key in enumerate(["rPPG", "DCT", "Concat"]):
        vals = [data[m][key] for m in order]
        ax.barh(y + (1 - k) * BAR_H, np.array(vals) - 0.5, left=0.5, height=BAR_H,
                linewidth=0.5, label=key, **STYLE[key])
    ax.set_xlim(0.5, 1.0)
    ax.set_xticks(np.arange(0.5, 1.01, 0.1))
    ax.tick_params(labelsize=FONT - 1, length=2)
    ax.set_xlabel("AUC", fontsize=FONT)
    ax.set_title(title, fontsize=FONT)
    ax.grid(axis="x", linestyle=":", linewidth=0.4, color="0.6")
    ax.set_axisbelow(True)

axes[0].set_yticks(y)
axes[0].set_yticklabels([LABEL[m] for m in order], fontsize=FONT - 1)
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", ncol=3, fontsize=FONT - 1,
           frameon=True, bbox_to_anchor=(0.55, 1.06), handlelength=1.6)

OUT.mkdir(parents=True, exist_ok=True)
for ext in ("pdf", "png"):
    fig.savefig(OUT / f"fig3_per_method_auc.{ext}", dpi=300, bbox_inches="tight", pad_inches=0.02)

print(f"{'Method':<15} {'rPPG':>6} {'DCT':>6} {'Concat':>6} | {'rPPG':>6} {'DCT':>6} {'Concat':>6}")
for m in order:
    a, b = in_domain[m], held_out[m]
    print(f"{LABEL[m]:<15} {a['rPPG']:6.3f} {a['DCT']:6.3f} {a['Concat']:6.3f} | "
          f"{b['rPPG']:6.3f} {b['DCT']:6.3f} {b['Concat']:6.3f}")
print(f"Saved to {OUT}/fig3_per_method_auc.{{pdf,png}}")
