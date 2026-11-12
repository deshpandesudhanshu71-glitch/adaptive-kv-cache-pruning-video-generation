import argparse
import json
import os
import time

import torch
from diffnext.pipelines import NOVAPipeline

import kvlab

MODEL_ID = "BAAI/nova-d48w1024-osp480"
HERE = os.path.dirname(os.path.abspath(__file__))

parser = argparse.ArgumentParser()
parser.add_argument("--method", default="full", choices=["full", "window", "sinkwindow", "h2o", "adaptive"])
parser.add_argument("--n", type=int, default=4, help="frames kept (window, h2o)")
parser.add_argument("--bits", type=int, default=16, choices=[16, 8, 4], help="cache storage")
parser.add_argument("--tau", type=float, default=0.9, help="adaptive: attention to cover")
parser.add_argument("--warmup", type=int, default=8, help="adaptive: full cache steps first")
parser.add_argument("--sink-frames", type=int, default=2, help="sinkwindow, adaptive: frames kept")
parser.add_argument("--recheck", type=int, default=0, help="adaptive: resize windows every R frames (0 = off)")
parser.add_argument("--analyse", action="store_true", help="log attention and shadow errors")
parser.add_argument("--length", type=int, default=18, help="latent frames (18 = 69 video frames)")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--prompt", default="dog", help="name of results/<prompt>/prompt_embeds.pt")
parser.add_argument("--tag", default=None, help="run name, made from the options if not given")
args = parser.parse_args()

tag = args.tag or {
    "full": "full", "window": f"window{args.n}", "h2o": f"h2o{args.n}",
    "sinkwindow": f"sink{args.sink_frames}_window{args.n}",
    "adaptive": f"adaptive{int(args.tau * 100)}" + (f"_recheck{args.recheck}" if args.recheck else ""),
}[args.method] + (f"_int{args.bits}" if args.bits < 16 else "")
prompt_dir = os.path.join(HERE, "results", args.prompt)
out_dir = os.path.join(prompt_dir, f"len{args.length}", tag)
os.makedirs(out_dir, exist_ok=True)


def sync_time():
    torch.cuda.synchronize()
    return time.perf_counter()


embeds_path = os.path.join(prompt_dir, "prompt_embeds.pt")
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

shadow = [("window", n) for n in (1, 2, 4, 8)] + [("sinkwindow", n) for n in (1, 2, 4)]
shadow += [("h2o", n) for n in (2, 4, 8)]
shadow += [("quant", 8), ("quant", 4)]
lab = kvlab.Lab(args.method, n=args.n, store_bits=args.bits, tau=args.tau, warmup=args.warmup,
                sink_frames=args.sink_frames, recheck=args.recheck, analyse=args.analyse,
                shadow=shadow if args.analyse else ())
kvlab.install(model, lab)

stats = {"temporal_s": 0.0, "spatial_s": 0.0, "frames": []}
features = []  # temporal output handed to the spatial layers, one per generated frame
orig_encoder, orig_frame = model.video_encoder.forward, model.generate_frame


def timed_encoder(*a, **kw):
    t0 = sync_time()
    out = orig_encoder(*a, **kw)
    stats["temporal_s"] += sync_time() - t0
    return out


def timed_frame(states, inputs):
    features.append(states["c"].detach().float().cpu())
    t0 = sync_time()
    out = orig_frame(states, inputs)
    dt = sync_time() - t0
    stats["spatial_s"] += dt
    tokens = lab.cache_tokens()
    stats["frames"].append({
        "latent_frame": len(stats["frames"]) + 1,
        "spatial_s": round(dt, 2),
        "cache_tokens_per_layer": tokens,
        "cache_mb": round(lab.cache_bytes() / 2**20, 1),
        "gpu_alloc_mb": round(torch.cuda.memory_allocated() / 2**20, 1),
    })
    f = stats["frames"][-1]
    print(f"  frame {f['latent_frame']}: {f['spatial_s']} s, cache {sum(tokens) // len(tokens)} "
          f"tokens per layer, {f['cache_mb']} MB, GPU {f['gpu_alloc_mb']} MB", flush=True)
    return out


model.video_encoder.forward = timed_encoder
model.generate_frame = timed_frame

print(f"Run {tag}: {args.length} latent frames, seed {args.seed}", flush=True)
torch.cuda.reset_peak_memory_stats()
torch.manual_seed(args.seed)
t0 = sync_time()
latents = pipe(prompt_embeds=[prompt_embeds], max_latent_length=args.length, guidance_scale=5.0,
               motion_flow=5.0, output_type="latent").frames
total = sync_time() - t0

torch.save(latents.float().cpu(), os.path.join(out_dir, "latents.pt"))
torch.save(torch.stack(features), os.path.join(out_dir, "features.pt"))
if args.analyse:
    torch.save({"mass": lab.mass_log, "shadow": lab.shadow_log},
               os.path.join(out_dir, "analysis.pt"))

result = {
    "tag": tag,
    "method": args.method, "n": args.n, "bits": args.bits, "tau": args.tau,
    "warmup": args.warmup, "sink_frames": args.sink_frames, "recheck": args.recheck,
    "adaptive_windows": {str(k): v for k, v in sorted(lab.windows.items())},
    "window_history": [{"step": s, "layer": l, "window": w} for s, l, w in lab.window_log],
    "model": MODEL_ID,
    "gpu": torch.cuda.get_device_name(0),
    "prompt": args.prompt,
    "seed": args.seed,
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
with open(os.path.join(out_dir, "run.json"), "w") as fh:
    json.dump(result, fh, indent=2)
print(json.dumps({k: v for k, v in result.items() if k != "frames"}, indent=2))
print(f"Saved {out_dir}")
