import argparse
import json
import os

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "figures")

parser = argparse.ArgumentParser()
parser.add_argument("--layers", type=int, nargs=2, default=[0, 2])
parser.add_argument("--run", default="adaptive90")
parser.add_argument("--warmup", type=int, default=8)
parser.add_argument("--sinks", type=int, default=2)
parser.add_argument("--n-max", type=int, default=8)
args = parser.parse_args()

root = os.path.join(HERE, "results", "dog", "len18")
info = json.load(open(os.path.join(root, args.run, "run.json")))
tau = info["tau"]
saved = {int(k): v for k, v in info["adaptive_windows"].items()}
analysis = torch.load(os.path.join(root, "analysis", "analysis.pt"), weights_only=False)
mass = {(s, l): m[0] for s, l, m in analysis["mass"]}  # first guidance branch, (H, G)

# the same profile as Lab.update_profile, over the warm-up steps
first = 2 + args.sinks
profile = {}
for (s, l), m in mass.items():
    cur = s + 1
    if s < 2 or s > args.warmup or cur <= first:
        continue
    dist = m[:, first:cur].mean(0).flip(0).numpy()
    p = profile.setdefault(l, np.zeros(64))
    p[: dist.size] += dist


def choose(p):
    p = p[p > 0]
    cover = np.cumsum(p) / p.sum()
    d = int((cover < tau).sum()) + 1
    return max(1, min(args.n_max, d + 1)), cover


for l, p in sorted(profile.items()):
    w, _ = choose(p)
    assert w == saved[l], f"layer {l}: rebuilt window {w}, saved {saved[l]}"
print("rebuilt windows match the run:", saved)

INK, MUTED, KEEP, DROP = "#262626", "#7F7F7F", "#1F6FB5", "#C9C9C9"
fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), dpi=200, sharey=True)
for ax, l in zip(axes, args.layers):
    w, cover = choose(profile[l])
    share = profile[l][profile[l] > 0] / profile[l].sum()
    d = np.arange(1, share.size + 1)
    needed = w - 1  # previous frames needed to reach tau
    ax.bar(d, share, color=[KEEP if i <= w else DROP for i in d], width=0.7)  # the code keeps w frames
    ax.plot(d, cover, color=INK, lw=2, marker="o", ms=5)
    ax.axhline(tau, color=MUTED, ls="--", lw=1)
    ax.text(share.size + 0.4, tau + 0.015, f"tau = {tau:.0%}", ha="right", va="bottom", fontsize=9, color=MUTED)
    ax.annotate(f"{cover[needed - 1]:.0%} reached after\n{needed} previous frame{'s' if needed > 1 else ''}",
                xy=(needed, cover[needed - 1]), xytext=(needed + 0.6, cover[needed - 1] - 0.3),
                fontsize=9, color=INK, arrowprops=dict(arrowstyle="->", color=MUTED))
    ax.set_title(f"Layer {l}: {needed} frame{'s' if needed > 1 else ''} reach {tau:.0%}, keeps {w} (one spare) + anchors",
                 fontsize=11, color=INK, loc="left")
    ax.set_xticks(d)
    ax.set_xlabel("how many frames back (1 = previous frame)", fontsize=10, color=INK)
    ax.set_ylim(0, 1.05)
    ax.tick_params(length=0, labelsize=9)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
axes[0].set_ylabel("share of attention to past frames", fontsize=10, color=INK)
fig.legend(handles=[plt.Rectangle((0, 0), 1, 1, fc=KEEP), plt.Rectangle((0, 0), 1, 1, fc=DROP),
                    plt.Line2D([], [], color=INK, lw=2, marker="o")],
           labels=["attention to this frame: kept", "attention to this frame: dropped", "running total"],
           loc="lower center", ncol=3, frameon=False, fontsize=9, bbox_to_anchor=(0.5, 0.0))
fig.text(0.01, 0.955, f"Warm-up attention (latent frames {first} to {args.warmup}), anchor frames left out. "
         f"Only {args.warmup + 1 - first} frames back exist during the warm-up, so the profile cannot see further. NOVA 0.6B, seed 0, prompt: a dog drinking water.", fontsize=8, color=MUTED)
fig.tight_layout(rect=(0, 0.07, 1, 0.94))
os.makedirs(OUT, exist_ok=True)
path = os.path.join(OUT, "9_how_a_layer_picks_its_window.png")
fig.savefig(path, facecolor="white")
print("saved", path)
