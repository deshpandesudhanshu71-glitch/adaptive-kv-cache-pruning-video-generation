"""Quality scores that do not need the full cache video, following VBench's definitions.

    subject consistency     DINO ViT-B/16 features of each frame against the first and the
                            previous frame (VBench uses the same model)
    temporal flickering     1 - mean absolute change between neighbouring frames
Background consistency and text alignment need CLIP and are not computed here.

This is a subset computed with the same backbones, not the official VBench toolkit.
Writes quality.json next to summary.json.

    python quality.py --prompt dog --length 18
"""
import argparse
import json
import os

import numpy as np
import torch
from transformers import ViTImageProcessor, ViTModel

HERE = os.path.dirname(os.path.abspath(__file__))
PROMPT = "A dog drinking water from a bowl in a sunny garden."

parser = argparse.ArgumentParser()
parser.add_argument("--prompt", default="dog")
parser.add_argument("--length", type=int, default=18)
args = parser.parse_args()
root = os.path.join(HERE, "results", args.prompt, f"len{args.length}")
runs = sorted(d for d in os.listdir(root) if os.path.exists(os.path.join(root, d, "frames.npy")))

dino = ViTModel.from_pretrained("facebook/dino-vitb16", add_pooling_layer=False).cuda().eval().half()
dino_proc = ViTImageProcessor.from_pretrained("facebook/dino-vitb16")


@torch.no_grad()
def features(frames, batch=8):
    d = []
    for i in range(0, len(frames), batch):
        ims = list(frames[i : i + batch])
        x = dino_proc(images=ims, return_tensors="pt")["pixel_values"].cuda().half()
        d.append(dino(pixel_values=x).last_hidden_state[:, 0].float())
    return torch.nn.functional.normalize(torch.cat(d), dim=-1)


def consistency(f):
    """VBench: mean over frames 2..T of (sim to first frame + sim to previous frame) / 2."""
    first = (f[1:] @ f[0]).clamp(min=0)
    prev = (f[1:] * f[:-1]).sum(-1).clamp(min=0)
    return float(((first + prev) / 2).mean())


out = {}
for run in runs:
    frames = np.load(os.path.join(root, run, "frames.npy"))
    d = features(frames)
    diff = np.abs(np.diff(frames.astype(np.int16), axis=0)).mean()
    out[run] = {
        "subject_consistency": round(consistency(d), 4),
        "temporal_flickering": round(float(1 - diff / 255), 4),
        # the same on the frames past the training length only
        "subject_consistency_after33": round(consistency(d[32:]), 4),
    }
    print(run, out[run], flush=True)
json.dump(out, open(os.path.join(root, "quality.json"), "w"), indent=2)
print("saved quality.json")
