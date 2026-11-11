"""Record NOVA's baseline cost with the full KV cache: time, peak GPU memory and cache size.

Nothing in the NOVA code is changed. The script wraps a few methods at run time to time the
temporal layers (video_encoder) and the spatial layers (generate_frame) separately, and reads
the size of the temporal KV cache after every latent frame.

Usage:
    python baseline.py --length 9      # 33 video frames, the training length
    python baseline.py --length 18     # 69 video frames, twice the training length
"""
import argparse
import json
import os
import time

import torch
from diffnext.pipelines import NOVAPipeline

from baseline_prompt import PROMPT

MODEL_ID = "BAAI/nova-d48w1024-osp480"

parser = argparse.ArgumentParser()
parser.add_argument("--length", type=int, default=9, help="latent frames (9 = 33 video frames)")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--out-dir", default=os.path.join(os.path.dirname(__file__), "results"))
args = parser.parse_args()
os.makedirs(args.out_dir, exist_ok=True)


def sync_time():
    torch.cuda.synchronize()
    return time.perf_counter()


# The prompt is encoded once by encode_prompt.py, so the 5.5 GB Phi-2 text encoder is never
# loaded here.
embeds_path = os.path.join(args.out_dir, "prompt_embeds.pt")
if not os.path.exists(embeds_path):
    raise SystemExit(f"{embeds_path} not found. Run encode_prompt.py first.")
prompt_embeds = torch.load(embeds_path)
pipe = NOVAPipeline.from_pretrained(
    MODEL_ID, torch_dtype=torch.float16, trust_remote_code=True, text_encoder=None
)
pipe.transformer.text_embed.encoders = []
pipe.text_encoder = None
pipe.enable_model_cpu_offload()

model = pipe.transformer
blocks = model.video_encoder.blocks
stats = {"temporal_s": 0.0, "spatial_s": 0.0, "frames": []}


def cache_bytes():
    total = 0
    for blk in blocks:
        kv = blk.attn.cache_kv
        if isinstance(kv, list):
            total += sum(t.numel() * t.element_size() for t in kv)
    return total


def cache_tokens():
    kv = blocks[0].attn.cache_kv
    return kv[0].shape[2] if isinstance(kv, list) else 0


orig_encoder = model.video_encoder.forward
orig_frame = model.generate_frame


def timed_encoder(*a, **kw):
    t0 = sync_time()
    out = orig_encoder(*a, **kw)
    stats["temporal_s"] += sync_time() - t0
    return out


def timed_frame(*a, **kw):
    t0 = sync_time()
    out = orig_frame(*a, **kw)
    dt = sync_time() - t0
    stats["spatial_s"] += dt
    stats["frames"].append({
        "latent_frame": len(stats["frames"]) + 1,
        "spatial_s": round(dt, 2),
        "cache_tokens": cache_tokens(),
        "cache_mb": round(cache_bytes() / 2**20, 1),
        "gpu_alloc_mb": round(torch.cuda.memory_allocated() / 2**20, 1),
    })
    f = stats["frames"][-1]
    print(f"  frame {f['latent_frame']}: {f['spatial_s']} s, cache {f['cache_tokens']} tokens, "
          f"{f['cache_mb']} MB, GPU {f['gpu_alloc_mb']} MB")
    return out


model.video_encoder.forward = timed_encoder
model.generate_frame = timed_frame

torch.cuda.reset_peak_memory_stats()
torch.manual_seed(args.seed)
t0 = sync_time()
pipe(prompt_embeds=[prompt_embeds], max_latent_length=args.length, guidance_scale=5.0,
     motion_flow=5.0, output_type="latent")
total = sync_time() - t0

result = {
    "model": MODEL_ID,
    "gpu": torch.cuda.get_device_name(0),
    "prompt": PROMPT,
    "latent_frames": args.length,
    "video_frames": 4 * (args.length - 1) + 1,
    "settings": "64 spatial steps, 25 diffusion steps, guidance 5, fp16, CPU offload",
    "total_s": round(total, 1),
    "temporal_s": round(stats["temporal_s"], 2),
    "spatial_s": round(stats["spatial_s"], 1),
    "peak_gpu_mb": round(torch.cuda.max_memory_allocated() / 2**20, 1),
    "final_cache_mb": stats["frames"][-1]["cache_mb"] if stats["frames"] else 0,
    "frames": stats["frames"],
}
path = os.path.join(args.out_dir, f"baseline_len{args.length}.json")
with open(path, "w") as fh:
    json.dump(result, fh, indent=2)
print(json.dumps({k: v for k, v in result.items() if k != "frames"}, indent=2))
print(f"Saved {path}")
