"""Generate a short video with NOVA (BAAI/nova-d48w1024-osp480) on a small GPU.

Usage:
    python generate_video.py "Many spotted jellyfish pulsating under water."
    python generate_video.py "A dog drinking water." --out dog.mp4 --hq
    python generate_video.py --decode-only --out dog.mp4    # retry decoding saved latents

The generated latents are saved next to the video (dog.latents.pt) before decoding,
so if decoding fails you can retry it without generating again.
"""
import argparse
import gc
import os

import torch
from accelerate.hooks import remove_hook_from_module
from diffnext.pipelines import NOVAPipeline
from diffnext.utils import export_to_video

MODEL_ID = "BAAI/nova-d48w1024-osp480"

parser = argparse.ArgumentParser()
parser.add_argument("prompt", nargs="?")
parser.add_argument("--out", default="output.mp4")
parser.add_argument("--hq", action="store_true", help="more steps, better quality, much slower")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--steps", type=int, help="autoregressive steps per frame (default 64, --hq 128)")
parser.add_argument("--diffusion-steps", type=int, help="denoising steps per token (default 25, --hq 100)")
parser.add_argument("--guidance", type=float, default=5.0, help="how strictly to follow the prompt")
parser.add_argument("--motion", type=float, default=5.0, help="amount of motion, higher = more movement")
parser.add_argument("--negative", default=None, help="things to avoid, e.g. 'blurry, distorted'")
parser.add_argument("--length", type=int, default=9, help="latent frames, 9 = 33 frames (training length)")
parser.add_argument("--decode-only", action="store_true", help="skip generation, decode saved latents")
parser.add_argument("--tile", type=int, default=32, help="VAE tile size in latent pixels (x8 = video pixels)")
args = parser.parse_args()
if not args.decode_only and not args.prompt:
    parser.error("a prompt is required unless --decode-only is used")

latents_path = os.path.splitext(args.out)[0] + ".latents.pt"


def free_gpu():
    gc.collect()
    torch.cuda.empty_cache()


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
    """Decode (B, C, T, H, W) latents in overlapping spatial tiles and blend the seams."""
    scale = 8  # The VAE upsamples 8x in height and width.
    _, _, _, h, w = z.shape
    out = weight = None
    ys, xs = tile_starts(h, tile, overlap), tile_starts(w, tile, overlap)
    for i, y in enumerate(ys):
        for j, x in enumerate(xs):
            print(f"  decoding tile {i * len(xs) + j + 1}/{len(ys) * len(xs)}", end="\r")
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
    print()
    return out / weight


@torch.no_grad()
def encode_on_cpu(pipe, text):
    """Run the Phi-2 text encoder on the CPU. It is about 5.5 GB, too big for a 6 GB GPU."""
    ids = pipe.tokenizer(text, max_length=256, truncation=True).input_ids
    if not ids:
        return None  # Empty prompt: the pipeline falls back to its learned null embedding.
    return pipe.text_encoder(torch.tensor([ids])).last_hidden_state[0].to(torch.float16)


pipe = NOVAPipeline.from_pretrained(MODEL_ID, torch_dtype=torch.float16, trust_remote_code=True)

if not args.decode_only:
    print("Encoding prompt on CPU...")
    pipe.text_encoder.to(torch.bfloat16)  # bfloat16 is much faster than float16 on CPU.
    prompt_embeds = encode_on_cpu(pipe, args.prompt)
    negative_embeds = encode_on_cpu(pipe, args.negative) if args.negative else None

# Drop the text encoder so only the video transformer and VAE ever go to the GPU.
pipe.transformer.text_embed.encoders = []
pipe.text_encoder = None
free_gpu()
pipe.enable_model_cpu_offload()

if args.decode_only:
    latents = torch.load(latents_path)
    print(f"Loaded latents from {latents_path}")
else:
    steps = {"num_inference_steps": 128, "num_diffusion_steps": 100} if args.hq else {}
    steps.update({"num_inference_steps": args.steps} if args.steps else {})
    steps.update({"num_diffusion_steps": args.diffusion_steps} if args.diffusion_steps else {})
    torch.manual_seed(args.seed)
    # Each latent frame is 4 video frames, plus 1: length 9 gives 33 frames at 768x480.
    latents = pipe(
        prompt_embeds=[prompt_embeds],
        negative_prompt_embeds=[negative_embeds] if negative_embeds is not None else None,
        max_latent_length=args.length,
        guidance_scale=args.guidance,
        motion_flow=args.motion,
        output_type="latent",
        **steps,
    ).frames
    latents = latents.float().cpu()
    torch.save(latents, latents_path)
    print(f"Saved latents to {latents_path}")

# Free the text encoder and transformer, then put only the VAE on the GPU.
vae = pipe.vae
remove_hook_from_module(vae, recurse=True)
del pipe
free_gpu()
vae.to("cuda")
# Decode 2 latent frames (5 video frames) at a time instead of 5 (17 frames).
vae.latent_min_t = 2

z = vae.unscale_(latents.clone()).to(vae.dtype)
video = tiled_decode(vae, z, tile=args.tile)

# (B, 3, T, H, W) in [-1, 1] -> (T, H, W, 3) uint8, same as the pipeline's postprocessing.
frames = video[0].permute(1, 2, 3, 0).mul(127.5).add(127.5).clamp(0, 255).byte().numpy()
export_to_video(frames, args.out, fps=12)
print(f"Saved {args.out} ({frames.shape[0]} frames, {frames.shape[2]}x{frames.shape[1]})")
