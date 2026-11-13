"""Figures and tables for the KV cache comparison.

Part 1 needs only the full cache run with --analyse (attention and shadow errors).
Part 2 needs decode_eval.py to have written summary.json.
Figures go to figures/.

    python analyse.py --prompt dog --length 18
"""
import argparse
import json
import os
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import LinearSegmentedColormap, LogNorm

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "figures")

parser = argparse.ArgumentParser()
parser.add_argument("--prompt", default="dog")
parser.add_argument("--length", type=int, default=18)
args = parser.parse_args()
root = os.path.join(HERE, "results", args.prompt, f"len{args.length}")
os.makedirs(OUT, exist_ok=True)

ink, muted, grid, base, surface = "#0b0b0b", "#52514e", "#e1e0d9", "#c3c2b7", "#fcfcfb"
blue, orange, aqua, yellow, magenta = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"
blues = LinearSegmentedColormap.from_list(
    "blues", ["#f4f8fd", "#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#184f95", "#0d366b"])
FOOT = "NOVA 0.6B, RTX 4050 Laptop (6 GB), fp16, seed 0, 69 frames, prompt: a dog drinking water"
plt.rcParams.update({"font.family": "Arial", "font.size": 9, "axes.edgecolor": base,
                     "axes.labelcolor": muted, "xtick.color": muted, "ytick.color": muted,
                     "figure.facecolor": surface, "axes.facecolor": surface})


def style(ax, ygrid=True):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(length=0)
    if ygrid:
        ax.grid(axis="y", color=grid, linewidth=0.8)
        ax.set_axisbelow(True)


def save(fig, name, foot=FOOT):
    fig.text(0.01, 0.01, foot, color=muted, fontsize=7)
    fig.savefig(os.path.join(OUT, name), dpi=200)
    plt.close(fig)
    print("saved", name)


# Part 1: where the attention goes (full cache, teacher forced).

# the second analysis run also covers the frame sink windows
src = "analysis" if os.path.exists(os.path.join(root, "analysis", "analysis.pt")) else "full"
analysis = torch.load(os.path.join(root, src, "analysis.pt"))
L = 1 + max(l for _, l, _ in analysis["mass"])
steps = 1 + max(s for s, _, _ in analysis["mass"])
mass = {(s, l): m[0].numpy() for s, l, m in analysis["mass"]}  # first guidance branch, (H, G)

# 1. Share of attention per layer: prompt, BOV, past frames, current frame.
share = np.zeros((L, 4))
for l in range(L):
    for s in range(2, steps):
        m = mass[(s, l)].mean(0)
        share[l] += [m[0], m[1], m[2 : s + 1].sum(), m[s + 1]]
    share[l] /= steps - 2
fig, ax = plt.subplots(figsize=(7.2, 4.0))
left = np.zeros(L)
names = ["prompt tokens", "start (BOV) tokens", "past frames", "current frame"]
for i, (name, col) in enumerate(zip(names, [blue, orange, aqua, yellow])):
    ax.barh(np.arange(L), share[:, i], left=left, color=col, height=0.72, label=name,
            edgecolor=surface, linewidth=1)
    left += share[:, i]
ax.set_yticks(np.arange(L), [f"layer {l}" for l in range(L)])
ax.invert_yaxis()
ax.set_xlim(0, 1)
ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
ax.set_xlabel("share of the new frame's attention (mean over heads and frames 3 to 18)")
ax.legend(ncol=4, frameon=False, loc="lower left", bbox_to_anchor=(0, 1.0), fontsize=8)
ax.set_title("Most of the temporal attention goes to the prompt and start tokens", loc="left",
             color=ink, fontsize=10.5, pad=24)
style(ax, ygrid=False)
fig.tight_layout(rect=(0, 0.04, 1, 1))
save(fig, "1_attention_by_group.png")

# 2. Attention to past frames by distance, last step, per layer.
s = steps - 1
dist = np.zeros((L, s - 1))
for l in range(L):
    past = mass[(s, l)].mean(0)[2 : s + 1][::-1]  # distance 1 = previous frame
    dist[l] = past / past.sum()
fig, ax = plt.subplots(figsize=(7.2, 4.0))
im = ax.imshow(dist, aspect="auto", cmap=blues, norm=LogNorm(vmin=1e-3, vmax=1))
ax.set_xticks(np.arange(s - 1), np.arange(1, s))
ax.set_yticks(np.arange(L), [f"layer {l}" for l in range(L)])
ax.set_xlabel("how many frames back")
cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
cb.set_label("share of the attention to past frames", color=muted)
cb.outline.set_visible(False)
ax.set_title("Attention to past frames falls off with distance, at the last frame",
             loc="left", color=ink, fontsize=10.5)
for side in ax.spines.values():
    side.set_visible(False)
ax.tick_params(length=0)
fig.tight_layout(rect=(0, 0.04, 1, 1))
save(fig, "2_attention_by_distance.png")

# 3. Shadow error: error each method would add at each layer if the cache so far were full.
err = defaultdict(lambda: np.zeros((steps, L)))
for st, l, cfg, e in analysis["shadow"]:
    err[cfg][st, l] = e
cfgs = ["window1", "window2", "window4", "window8", "sinkwindow1", "sinkwindow2", "sinkwindow4",
        "h2o2", "h2o4", "h2o8", "quant8", "quant4"]
cfgs = [c for c in cfgs if c in err]
labels = {"window1": "window 1", "window2": "window 2", "window4": "window 4",
          "window8": "window 8", "sinkwindow1": "sinks + 1", "sinkwindow2": "sinks + 2",
          "sinkwindow4": "sinks + 4", "h2o2": "H2O 2", "h2o4": "H2O 4", "h2o8": "H2O 8",
          "quant8": "8 bit", "quant4": "4 bit"}
late = slice(9, steps)  # frames past the training length, where pruning is active for all
mat = np.array([err[c][late].mean(0) for c in cfgs])
fig, ax = plt.subplots(figsize=(7.6, 0.9 + 0.32 * len(cfgs)))
im = ax.imshow(mat, aspect="auto", cmap=blues, norm=LogNorm(vmin=1e-3, vmax=max(0.3, mat.max())))
ax.set_yticks(np.arange(len(cfgs)), [labels[c] for c in cfgs])
ax.set_xticks(np.arange(L), np.arange(L))
ax.set_xlabel("temporal layer")
for i in range(len(cfgs)):
    for j in range(L):
        v = mat[i, j]
        ax.text(j, i, f"{v * 100:.1f}" if v >= 0.001 else "0", ha="center", va="center",
                fontsize=6, color="#ffffff" if v > 0.03 else ink)
cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
cb.set_label("relative error of the attention output", color=muted)
cb.outline.set_visible(False)
ax.set_title("Error each method adds per layer (in %), frames 10 to 18, full cache history",
             loc="left", color=ink, fontsize=10.5)
for side in ax.spines.values():
    side.set_visible(False)
ax.tick_params(length=0)
fig.tight_layout(rect=(0, 0.04, 1, 1))
save(fig, "3_error_per_layer.png")

attention_tables = {
    "share_by_group": {f"layer {l}": dict(zip(names, map(float, share[l]))) for l in range(L)},
    "shadow_error_mean_frames_10_18": {c: float(err[c][late].mean()) for c in cfgs},
    "shadow_error_per_layer": {c: list(map(float, err[c][late].mean(0))) for c in cfgs},
}
json.dump(attention_tables, open(os.path.join(root, "attention_tables.json"), "w"), indent=2)

# Part 2: the real runs, each generating its own video.

summary_path = os.path.join(root, "summary.json")
if not os.path.exists(summary_path):
    raise SystemExit("summary.json not found yet, run decode_eval.py for part 2")
summary = {r["run"]: r for r in json.load(open(summary_path))}


def family(run):
    if run.startswith("window"):
        return blue
    if run.startswith("sink"):
        return magenta
    if run.startswith("h2o"):
        return orange
    if run.startswith("adaptive"):
        return yellow
    if "int" in run:
        return aqua
    return ink


def nice(run):
    """window4_int8 -> 'window 4 + 8 bit', full_int4 -> '4 bit'."""
    name, _, bits = run.partition("_int")
    if name.startswith("sink"):
        name = "sinks + " + name.split("window")[1]
    for key, label in (("window", "window "), ("h2o", "H2O "), ("adaptive", "adaptive ")):
        if name.startswith(key):
            n = name[len(key):]
            name = label + (f"{int(n) / 100:.2f}" if key == "adaptive" else n)
    if name == "full" and bits:
        return f"{bits} bit"
    return name + (f" + {bits} bit" if bits else "")


runs = [r for r in summary if r != "full_repeat"]

# 4. Cache size over the video.
fig, ax = plt.subplots(figsize=(7.2, 4.0))
for run in runs:
    info = json.load(open(os.path.join(root, run, "run.json")))
    x = [4 * (f["latent_frame"] - 1) + 1 for f in info["frames"]]
    y = [f["cache_mb"] for f in info["frames"]]
    ax.plot(x, y, color=family(run), linewidth=2 if run == "full" else 1.4,
            linestyle="--" if "int" in run and run.startswith(("window", "adaptive")) else "-")
    ax.annotate(nice(run), (x[-1], y[-1]), xytext=(4, 0), textcoords="offset points",
                color=muted, fontsize=6.5, va="center")
ax.axvline(33, color=base, linewidth=1, linestyle=":")
ax.text(34, ax.get_ylim()[1] * 0.95, "training length", color=muted, fontsize=7.5, va="top")
ax.set_xlabel("video frames generated")
ax.set_ylabel("temporal KV cache (MB)")
ax.set_xlim(0, 84)
ax.set_title("Cache size per method", loc="left", color=ink, fontsize=10.5)
style(ax)
fig.tight_layout(rect=(0, 0.04, 1, 1))
save(fig, "4_cache_size.png")

# 5. Memory against quality. Points that sit close together get their labels moved apart.
LABEL_AT = {"window8": (6, 4, "left"), "h2o8": (6, -11, "left"), "adaptive90": (0, -15, "center"),
            "adaptive80": (-6, 4, "right"), "h2o4": (6, 5, "left"), "window4": (-6, 4, "right"),
            "sink2_window2": (6, -11, "left"), "sink2_window1": (6, 2, "left"),
            "window2": (-6, 2, "right")}
fig, ax = plt.subplots(figsize=(7.2, 4.2))
for run in runs:
    if run == "full":
        continue
    r = summary[run]
    ax.scatter(r["final_cache_mb"], r["lpips"], s=46, color=family(run), edgecolor=surface,
               linewidth=1.5, zorder=3)
    dx, dy, ha = LABEL_AT.get(run, (5, 3, "left"))
    ax.annotate(nice(run), (r["final_cache_mb"], r["lpips"]), xytext=(dx, dy), ha=ha,
                textcoords="offset points", color=muted, fontsize=7)
noise = summary.get("full_repeat", {}).get("lpips")
if noise is not None:
    ax.axhline(noise, color=base, linewidth=1, linestyle=":")
    ax.text(5, noise, " rerun of the full cache", color=muted, fontsize=7, va="bottom")
ax.axvline(summary["full"]["final_cache_mb"], color=ink, linewidth=1, linestyle=":")
ax.text(summary["full"]["final_cache_mb"], ax.get_ylim()[1], " full cache", color=muted,
        fontsize=7, va="top")
ax.set_xlabel("cache size at the last frame (MB)")
ax.set_ylabel("LPIPS against the full cache video (lower is closer)")
ax.set_xlim(left=0)
ax.set_ylim(bottom=0)
ax.set_title("Memory saved against change in the video", loc="left", color=ink, fontsize=10.5)
style(ax)
fig.tight_layout(rect=(0, 0.04, 1, 1))
save(fig, "5_memory_vs_quality.png")

# 6. LPIPS frame by frame, one panel per family.
groups = [("window", blue), ("sink", magenta), ("h2o", orange), ("int", aqua), ("adaptive", yellow)]
titles = ["sliding window", "frame sinks + window", "H2O", "quantisation", "adaptive"]
fig, axes = plt.subplots(1, 5, figsize=(11.5, 3.2), sharey=True)
for ax, (key, col), title in zip(axes, groups, titles):
    members = [r for r in runs if (r.startswith(key) if key != "int" else r.startswith("full_int"))]
    if key == "adaptive":
        members += [r for r in runs if r.startswith("window4_int")]
    for i, run in enumerate(members):
        m = json.load(open(os.path.join(root, run, "metrics.json")))
        alpha = 0.45 + 0.55 * (i + 1) / len(members)
        ax.plot(m["lpips"], color=col if "window4_int" not in run else blue, alpha=alpha,
                linewidth=1.4)
        ax.annotate(nice(run), (len(m["lpips"]) - 1, m["lpips"][-1]), xytext=(-2, 4),
                    textcoords="offset points", ha="right", color=muted, fontsize=6.5)
    if noise is not None:
        m = json.load(open(os.path.join(root, "full_repeat", "metrics.json")))
        ax.plot(m["lpips"], color=base, linewidth=1, linestyle=":")
    ax.axvline(33, color=base, linewidth=1, linestyle=":")
    ax.set_title(title, loc="left", color=ink, fontsize=9.5)
    ax.set_xlabel("video frame")
    style(ax)
axes[0].set_ylabel("LPIPS against full cache")
fig.tight_layout(rect=(0, 0.05, 1, 1))
save(fig, "6_lpips_per_frame.png")

# Table.
cols = ["run", "final_cache_mb", "peak_gpu_mb", "temporal_s", "total_s", "psnr", "ssim", "lpips",
        "lpips_after33", "latent_err", "feature_err", "motion"]
full_mb = summary["full"]["final_cache_mb"]
lines = ["| " + " | ".join(cols + ["cache saved"]) + " |", "|" + "---|" * (len(cols) + 1)]
for run in summary:
    r = summary[run]
    saved = f"{100 * (1 - r['final_cache_mb'] / full_mb):.0f}%"
    lines.append("| " + " | ".join(str(r[c]) for c in cols) + f" | {saved} |")
open(os.path.join(OUT, "results_table.md"), "w").write("\n".join(lines) + "\n")
print("\n".join(lines))
