"""Decode every run's latents to video and compare it with the full cache run.

For each run in results/<prompt>/len<L>/ this writes video.mp4 and frames.npy (if missing), then
metrics.json with, per frame, PSNR, SSIM and LPIPS against the full cache video, plus the
relative error of the latents and of the temporal features handed to the spatial layers.
A summary of every run goes to summary.json in the same folder.

Run it after all the runs have finished, since the VAE needs the GPU:
    python decode_eval.py --prompt dog --length 18
"""
import argparse
import gc
import json
import os

import lpips
import numpy as np
import torch
from huggingface_hub import snapshot_download
from skimage.metrics import structural_similarity
from diffnext.models.autoencoders.autoencoder_kl_opensora import AutoencoderKLOpenSora
from diffnext.utils import export_to_video

MODEL_ID = "BAAI/nova-d48w1024-osp480"
HERE = os.path.dirname(os.path.abspath(__file__))

parser = argparse.ArgumentParser()
parser.add_argument("--prompt", default="dog")
parser.add_argument("--length", type=int, default=18)
parser.add_argument("--ref", default="full")
args = parser.parse_args()
root = os.path.join(HERE, "results", args.prompt, f"len{args.length}")
runs = sorted(d for d in os.listdir(root) if os.path.exists(os.path.join(root, d, "latents.pt"))
              and not d.startswith("analysis"))


def free_gpu():
    gc.collect()
    torch.cuda.empty_cache()


# Tiled decoding, the same as generate_video.py, so it fits in 6 GB.


def tile_starts(size, tile, overlap):
    if size <= tile:
        return [0]
    return list(range(0, size - tile, tile - overlap)) + [size - tile]


def edge_ramp(length, ramp, fade_in, fade_out):
    w = torch.ones(length)
    r = torch.arange(1, ramp + 1, dtype=torch.float32) / (ramp + 1)
    if fade_in:
        w[:ramp] = r
    if fade_out:
        w[-ramp:] = r.flip(0)
    return w


@torch.no_grad()
def tiled_decode(vae, z, tile=32, overlap=8):
    scale = 8
    _, _, _, h, w = z.shape
    out = weight = None
    for y in tile_starts(h, tile, overlap):
        for x in tile_starts(w, tile, overlap):
            th, tw = min(tile, h), min(tile, w)
            part = vae.decode(z[..., y : y + th, x : x + tw].cuda()).sample.float().cpu()
            if out is None:
                out = torch.zeros(part.shape[:3] + (h * scale, w * scale))
                weight = torch.zeros(h * scale, w * scale)
            ramp = overlap * scale
            wy = edge_ramp(th * scale, ramp, y > 0, y + th < h)
            wx = edge_ramp(tw * scale, ramp, x > 0, x + tw < w)
            mask = wy[:, None] * wx[None, :]
            ys_, xs_ = slice(y * scale, (y + th) * scale), slice(x * scale, (x + tw) * scale)
            out[..., ys_, xs_] += part * mask
            weight[ys_, xs_] += mask
            del part
            free_gpu()
    return out / weight


todo = [r for r in runs if not os.path.exists(os.path.join(root, r, "frames.npy"))]
if todo:
    path = snapshot_download(MODEL_ID)
    vae = AutoencoderKLOpenSora.from_pretrained(os.path.join(path, "vae"),
                                                torch_dtype=torch.float16).cuda().eval()
    vae.latent_min_t = 2
    for run in todo:
        print(f"decoding {run}", flush=True)
        latents = torch.load(os.path.join(root, run, "latents.pt"))
        video = tiled_decode(vae, vae.unscale_(latents.clone()).to(vae.dtype))
        frames = video[0].permute(1, 2, 3, 0).mul(127.5).add(127.5).clamp(0, 255).byte().numpy()
        np.save(os.path.join(root, run, "frames.npy"), frames)
        export_to_video(frames, os.path.join(root, run, "video.mp4"), fps=12)
        del video, frames
    del vae
    free_gpu()


def rel_err(a, b):
    """Relative error per frame along dim 0."""
    a, b = a.flatten(1).float(), b.flatten(1).float()
    return ((a - b).norm(dim=1) / b.norm(dim=1)).tolist()


def psnr(a, b):
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    return float("inf") if mse == 0 else float(10 * np.log10(255.0**2 / mse))


lp = lpips.LPIPS(net="alex", verbose=False).cuda().eval()


@torch.no_grad()
def lpips_frames(a, b, batch=4):
    out = []
    for i in range(0, len(a), batch):
        ta = torch.from_numpy(a[i : i + batch]).permute(0, 3, 1, 2).float().div(127.5).sub(1)
        tb = torch.from_numpy(b[i : i + batch]).permute(0, 3, 1, 2).float().div(127.5).sub(1)
        out += lp(ta.cuda(), tb.cuda()).flatten().tolist()
    return out


ref_dir = os.path.join(root, args.ref)
ref_frames = np.load(os.path.join(ref_dir, "frames.npy"))
ref_lat = torch.load(os.path.join(ref_dir, "latents.pt"))[0].transpose(0, 1)  # (T, C, H, W)
ref_feat = torch.load(os.path.join(ref_dir, "features.pt"))[:, 0]  # first guidance branch
summary = []
for run in runs:
    d = os.path.join(root, run)
    info = json.load(open(os.path.join(d, "run.json")))
    frames = np.load(os.path.join(d, "frames.npy"))
    lat = torch.load(os.path.join(d, "latents.pt"))[0].transpose(0, 1)
    feat = torch.load(os.path.join(d, "features.pt"))[:, 0]
    m = {
        "psnr": [psnr(a, b) for a, b in zip(frames, ref_frames)],
        "ssim": [float(structural_similarity(a, b, channel_axis=2, data_range=255))
                 for a, b in zip(frames, ref_frames)],
        "lpips": lpips_frames(frames, ref_frames),
        "latent_err": rel_err(lat, ref_lat),
        "feature_err": rel_err(feat, ref_feat),
        # how much the video moves from frame to frame, to spot frozen or jittery output
        "motion": [float(np.abs(frames[i + 1].astype(np.int16) - frames[i]).mean())
                   for i in range(len(frames) - 1)],
    }
    json.dump(m, open(os.path.join(d, "metrics.json"), "w"))
    finite = [p for p in m["psnr"] if p != float("inf")]
    row = {
        "run": run,
        "final_cache_mb": info["final_cache_mb"],
        "peak_gpu_mb": info["peak_gpu_mb"],
        "temporal_s": info["temporal_s"],
        "total_s": info["total_s"],
        "psnr": round(float(np.mean(finite)), 2) if finite else None,
        "ssim": round(float(np.mean(m["ssim"])), 4),
        "lpips": round(float(np.mean(m["lpips"])), 4),
        # frames past the 33 NOVA was trained on, where the cache is largest
        "lpips_after33": round(float(np.mean(m["lpips"][33:])), 4),
        "latent_err": round(float(np.mean(m["latent_err"])), 4),
        "feature_err": round(float(np.mean(m["feature_err"])), 4),
        "motion": round(float(np.mean(m["motion"])), 2),
        "adaptive_windows": info.get("adaptive_windows", {}),
    }
    summary.append(row)
    print(row, flush=True)
json.dump(summary, open(os.path.join(root, "summary.json"), "w"), indent=2)
print("saved", os.path.join(root, "summary.json"))
