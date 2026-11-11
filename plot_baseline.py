"""Plot how NOVA's temporal KV cache grows with video length (baseline, full cache)."""
import json
import os

import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "figures", "kv_cache_growth.png")

with open(os.path.join(HERE, "results", "baseline_len18.json")) as fh:
    run = json.load(fh)
frames = [f["latent_frame"] for f in run["frames"]]
video_frames = [4 * (n - 1) + 1 for n in frames]
cache = [f["cache_mb"] for f in run["frames"]]

ink, muted, grid, blue = "#0b0b0b", "#52514e", "#e4e3df", "#2a78d6"
plt.rcParams.update({"font.family": "Arial", "font.size": 10})
fig, ax = plt.subplots(figsize=(6.4, 3.6), dpi=200)
ax.plot(video_frames, cache, color=blue, linewidth=2, marker="o", markersize=4.5,
        markeredgecolor="white", markeredgewidth=1)
ax.axvline(33, color=muted, linewidth=1, linestyle="--")
ax.text(34, 60, "training length\n(33 frames)", color=muted, fontsize=8.5, va="bottom")
ax.annotate(f"{cache[8]:.0f} MB", (video_frames[8], cache[8]), xytext=(-48, 10),
            textcoords="offset points", color=ink, fontsize=9)
ax.annotate(f"{cache[-1]:.0f} MB", (video_frames[-1], cache[-1]), xytext=(-52, 4),
            textcoords="offset points", color=ink, fontsize=9)
ax.set_xlabel("Video frames generated", color=muted)
ax.set_ylabel("Temporal KV cache (MB)", color=muted)
ax.set_title("NOVA 0.6B: KV cache grows with every frame", loc="left", color=ink, fontsize=11)
ax.set_xlim(0, 72)
ax.set_ylim(0, 950)
ax.grid(axis="y", color=grid, linewidth=0.8)
for side in ("top", "right"):
    ax.spines[side].set_visible(False)
for side in ("left", "bottom"):
    ax.spines[side].set_color(grid)
ax.tick_params(colors=muted, length=0)
fig.text(0.01, 0.01, "RTX 4050 Laptop (6 GB), fp16, guidance 5 (batch of 2), 64 spatial steps, 25 diffusion steps",
         color=muted, fontsize=7)
fig.tight_layout(rect=(0, 0.04, 1, 1))
os.makedirs(os.path.dirname(OUT), exist_ok=True)
fig.savefig(OUT)
print("saved", OUT)
