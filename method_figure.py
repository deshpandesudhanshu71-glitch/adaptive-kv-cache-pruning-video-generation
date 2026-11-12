import argparse
import json
import os

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Patch, Rectangle

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "figures")

parser = argparse.ArgumentParser()
parser.add_argument("--run", default="adaptive90")
parser.add_argument("--length", type=int, default=18)
parser.add_argument("--sinks", type=int, default=2)
parser.add_argument("--warmup", type=int, default=8)
args = parser.parse_args()

info = json.load(open(os.path.join(HERE, "results", "dog", f"len{args.length}", args.run, "run.json")))
windows = {int(k): v for k, v in info["adaptive_windows"].items()}
tau = info.get("tau", 0.9)
layers = sorted(windows)
past = args.length - 1  # frames already in the cache when the last one is made

INK, MUTED = "#262626", "#7F7F7F"
SINK, KEEP, DROP, NOW = "#C2185B", "#1F6FB5", "#E6E6E6", "#262626"

fig = plt.figure(figsize=(12, 7.6), dpi=200)
flow = fig.add_axes([0.04, 0.80, 0.92, 0.16])
grid = fig.add_axes([0.08, 0.15, 0.70, 0.57])
side = fig.add_axes([0.80, 0.15, 0.17, 0.57])

# steps
steps = [
    ("1. Warm-up", f"keep everything for\n{args.warmup} frames, record\nattention per layer"),
    ("2. Anchors", f"always keep text tokens\nand the first {args.sinks}\ngenerated frames"),
    ("3. Window per layer", f"fewest recent frames\nthat cover {tau:.0%} of\nattention to the past"),
    ("4. Re-check (new)", "every few frames,\nmeasure again and\nupdate the windows"),
]
flow.set_xlim(0, 4)
flow.set_ylim(0, 1)
flow.axis("off")
for i, (head, text) in enumerate(steps):
    new = "new" in head
    flow.add_patch(FancyBboxPatch((i + 0.06, 0.05), 0.80, 0.9, boxstyle="round,pad=0.01,rounding_size=0.04",
                                  fc="white", ec=KEEP if not new else MUTED, lw=1.2, ls="-" if not new else "--"))
    flow.text(i + 0.46, 0.76, head, ha="center", va="center", fontsize=11, weight="bold", color=INK)
    flow.text(i + 0.46, 0.36, text, ha="center", va="center", fontsize=9, color=INK, linespacing=1.3)
    if i < len(steps) - 1:
        flow.annotate("", xy=(i + 1.05, 0.5), xytext=(i + 0.88, 0.5),
                      arrowprops=dict(arrowstyle="->", color=MUTED, lw=1.2))

# cache per layer while the last frame is made
for row, layer in enumerate(layers):
    w = windows[layer]
    y = len(layers) - 1 - row
    for f in range(1, past + 1):
        if f <= args.sinks:
            c = SINK
        elif f > past - w:
            c = KEEP
        else:
            c = DROP
        grid.add_patch(Rectangle((f - 0.5 + 0.04, y - 0.5 + 0.06), 0.92, 0.88, fc=c, ec="none"))
    grid.add_patch(Rectangle((past + 1 - 0.5 + 0.04, y - 0.5 + 0.06), 0.92, 0.88, fc="white", ec=NOW, lw=1))
grid.set_xlim(0.4, past + 1.6)
grid.set_ylim(-0.6, len(layers) - 0.4)
grid.set_xticks(range(1, past + 2))
grid.set_xticklabels([str(f) for f in range(1, past + 1)] + ["now"], fontsize=8, color=INK)
grid.set_yticks(range(len(layers)))
grid.set_yticklabels([f"layer {l}" for l in reversed(layers)], fontsize=8, color=INK)
grid.set_xlabel("latent frame in the cache", fontsize=10, color=INK)
grid.tick_params(length=0)
for s in grid.spines.values():
    s.set_visible(False)
grid.set_title(f"What each temporal layer keeps while making frame {args.length} "
               f"(adaptive, tau = {tau:.2f}, windows from the real run)", fontsize=11, color=INK, loc="left")
grid.legend(handles=[Patch(fc=SINK, label=f"anchor frames (first {args.sinks})"),
                     Patch(fc=KEEP, label="kept: recent window"),
                     Patch(fc=DROP, label="dropped"),
                     Patch(fc="white", ec=NOW, label="frame being made")],
            loc="upper left", bbox_to_anchor=(0, -0.09), ncol=4, frameon=False, fontsize=9)

# frames kept per layer
kept = [args.sinks + windows[l] for l in layers]
side.barh(range(len(layers)), list(reversed(kept)), color=KEEP, height=0.6)
side.axvline(past, color=MUTED, lw=1, ls=":")
side.text(past - 0.3, -0.45, "full cache", fontsize=8, color=MUTED, ha="right", va="bottom")
for y, k in enumerate(reversed(kept)):
    side.text(k + 0.3, y, str(k), va="center", fontsize=8, color=INK)
side.set_ylim(-0.6, len(layers) - 0.4)
side.set_xlim(0, past + 1)
side.set_yticks([])
side.set_xlabel("frames kept", fontsize=10, color=INK)
side.tick_params(length=0, labelsize=8)
for s in ("top", "right", "left"):
    side.spines[s].set_visible(False)
side.set_title(f"{sum(kept)} of {past * len(layers)} kept", fontsize=10, color=INK, loc="left")

fig.text(0.04, 0.012, f"NOVA 0.6B, RTX 4050 Laptop (6 GB), seed 0, {4 * args.length - 3} frames, prompt: a dog drinking water. "
         f"Cache at the end: {info['final_cache_mb']:.0f} MB (full cache: 842 MB).", fontsize=8, color=MUTED)
os.makedirs(OUT, exist_ok=True)
path = os.path.join(OUT, "8_method_cache_per_layer.png")
fig.savefig(path, facecolor="white")
print("saved", path)
