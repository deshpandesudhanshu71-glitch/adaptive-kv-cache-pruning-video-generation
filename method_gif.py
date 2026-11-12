import argparse
import io
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "figures")

parser = argparse.ArgumentParser()
parser.add_argument("--run", default="adaptive90")
parser.add_argument("--layers", type=int, nargs="+", default=[0, 7, 14, 2])
parser.add_argument("--length", type=int, default=18)
parser.add_argument("--warmup", type=int, default=8)
parser.add_argument("--sinks", type=int, default=2)
args = parser.parse_args()

root = os.path.join(HERE, "results", "dog", f"len{args.length}")
info = json.load(open(os.path.join(root, args.run, "run.json")))
full_info = json.load(open(os.path.join(root, "full", "run.json")))
windows = {int(k): v for k, v in info["adaptive_windows"].items()}
video = np.load(os.path.join(root, args.run, "frames.npy"), mmap_mode="r")
L = len(windows)

# cache size: a fixed part (text and start tokens) plus 45 MB per latent frame over all layers
fixed = full_info["final_cache_mb"] - 45.0 * args.length
per_slot = 45.0 / L  # one frame in one layer


def kept(t, w):
    """Frames in a layer's cache while frame t is made (t itself included)."""
    if t <= args.warmup or w is None:
        return list(range(1, t + 1))
    return sorted(set(range(1, args.sinks + 1)) | set(range(t - w, t + 1)))


def size(t, adaptive):
    slots = sum(len(kept(t, windows[l] if adaptive else None)) for l in range(L))
    return fixed + per_slot * slots


INK, MUTED = "#333333", "#888888"
SINK, KEEP, DROP, NOW = "#C2185B", "#1F6FB5", "#E4E4E4", "#F2B705"
rows = [("NOVA (full cache)", None)] + [(f"layer {l} (window {windows[l]})", windows[l]) for l in args.layers]
end_full, end_ours = size(args.length, False), size(args.length, True)


def draw(t):
    fig = plt.figure(figsize=(10, 4.6), dpi=100)
    ax_img = fig.add_axes([0.02, 0.30, 0.33, 0.58])
    ax = fig.add_axes([0.40, 0.30, 0.58, 0.58])
    ax_bar = fig.add_axes([0.40, 0.07, 0.58, 0.13])

    vid = 0 if t == 1 else 4 * (t - 1)
    ax_img.imshow(video[min(vid, len(video) - 1)])
    ax_img.axis("off")
    ax_img.set_title(f"making latent frame {t} of {args.length}", fontsize=11, color=INK, loc="left")

    for r, (name, w) in enumerate(rows):
        y = len(rows) - 1 - r
        keep = kept(t, w)
        for f in range(1, args.length + 1):
            if f > t:
                continue
            if f == t:
                c = NOW
            elif f not in keep:
                c = DROP
            elif f <= args.sinks and w is not None:
                c = SINK
            else:
                c = KEEP
            ax.add_patch(Rectangle((f - 0.45, y - 0.38), 0.9, 0.76, fc=c, ec="none"))
        ax.text(0.2, y, name, ha="right", va="center", fontsize=9.5, color=INK)
        if r == 0:
            ax.axhline(y - 0.62, color=MUTED, lw=0.6)
    ax.set_xlim(-4.6, args.length + 0.6)
    ax.set_ylim(-0.6, len(rows) - 0.4)
    ax.set_xticks(range(1, args.length + 1))
    ax.tick_params(length=0, labelsize=8, colors=MUTED)
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    if t <= args.warmup:
        msg = f"warm-up: nothing deleted yet, measuring where each layer looks ({t}/{args.warmup})"
    else:
        msg = "each layer keeps the first 2 frames + its own recent window"
    ax.set_title(msg, fontsize=10.5, color=INK, loc="left")
    ax.text(-4.6, -1.05, "pink: first 2 frames (anchors)    blue: kept    grey: deleted    yellow: frame being made",
            fontsize=8.5, color=MUTED, va="center")

    full_mb, ours_mb = size(t, False), size(t, True)
    ax_bar.barh([1, 0], [full_mb, ours_mb], color=[MUTED, KEEP], height=0.6)
    ax_bar.text(full_mb + 8, 1, f"{full_mb:.0f} MB", va="center", fontsize=9, color=INK)
    ax_bar.text(ours_mb + 8, 0, f"{ours_mb:.0f} MB", va="center", fontsize=9, color=INK)
    ax_bar.set_yticks([1, 0])
    ax_bar.set_yticklabels(["full cache", "per-layer"], fontsize=9, color=INK)
    ax_bar.set_xlim(0, end_full * 1.12)
    ax_bar.set_xticks([])
    ax_bar.tick_params(length=0)
    for s in ax_bar.spines.values():
        s.set_visible(False)
    ax_bar.set_title("cache size", fontsize=9.5, color=INK, loc="left")

    buf = io.BytesIO()
    fig.savefig(buf, format="png", facecolor="white")
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


frames = [draw(t) for t in range(1, args.length + 1)]
durations = [700] * len(frames)
durations[args.warmup - 1] = 1800  # pause where the windows are chosen
durations[-1] = 3500               # hold the last frame
pal = frames[-1].quantize(colors=128, method=Image.Quantize.MEDIANCUT)
frames = [f.quantize(palette=pal, dither=Image.Dither.NONE) for f in frames]
os.makedirs(OUT, exist_ok=True)
path = os.path.join(OUT, "method_animation.gif")
frames[0].save(path, save_all=True, append_images=frames[1:], duration=durations, loop=0, optimize=True)
print("saved", path, f"{os.path.getsize(path) / 1e6:.1f} MB", f"end: full {end_full:.0f} MB, per-layer {end_ours:.0f} MB")
