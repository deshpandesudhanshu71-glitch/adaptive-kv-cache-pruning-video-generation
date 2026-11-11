# KV cache pruning in NOVA

Code for my MSc thesis on adaptive KV cache pruning for autoregressive video generation. Everything runs on NOVA (BAAI/nova-d48w1024-osp480, 0.6B) without changing its source. kvlab.py replaces the attention forward of the temporal blocks at run time, so each cache method can be switched on from the command line.

All results here use one prompt ("A dog drinking water from a bowl in a sunny garden."), seed 0 and 18 latent frames (69 video frames), on an RTX 4050 laptop GPU with 6 GB.

## Files

| File | What it does |
|---|---|
| kvlab.py | The cache methods: full, sliding window, frame sinks + window, H2O, 8 and 4 bit quantisation, and the adaptive per layer window. Also logs attention per cache group and the teacher forced (shadow) error. |
| run_method.py | Generates one video with one method and saves latents, temporal features and run info. |
| decode_eval.py | Decodes the latents with the VAE and compares each run with the full cache run (PSNR, SSIM, LPIPS, latent and feature error). |
| quality.py | Subject consistency (DINO ViT-B/16) and temporal flickering, following the VBench definitions. This is not the official VBench toolkit. |
| analyse.py | Figures and the results table. |
| method_figure.py, window_figure.py, method_gif.py | Figures and an animation of what the adaptive method keeps per layer. |
| baseline.py, plot_baseline.py | Time, peak GPU memory and cache size with the full cache. |
| encode_prompt.py, generate_video.py | Prompt encoding on its own (Phi-2 does not fit next to the model in 6 GB) and a general video generation script. |
| run_experiments.sh | Every run used in the results, then decoding, metrics and figures. |

## Setup

Install NOVA from https://github.com/baaivision/NOVA, then `pip install -r requirements.txt`. The model weights download from Hugging Face on the first run.

## Running

```
python encode_prompt.py --out results/dog/prompt_embeds.pt
python run_method.py --method adaptive --tau 0.9
python decode_eval.py --prompt dog --length 18
python quality.py --prompt dog --length 18
python analyse.py
```

results/dog/prompt_embeds.pt is already included, so the first step can be skipped. Figures are written to figures/.

## Results

results/dog/len18/ has one folder per run with run.json, metrics.json and video.mp4. The same folder has summary.json (cache size, time, PSNR, SSIM and LPIPS for every run), quality.json and attention_tables.json. Latents, features and decoded frames are about 2.6 GB, so they are not in the repo. I can share them separately.
