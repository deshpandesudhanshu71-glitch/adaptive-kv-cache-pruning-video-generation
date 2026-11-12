import argparse
import os

import torch
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer
from diffnext.models.text_encoders.phi import PhiEncoderModel

from baseline_prompt import PROMPT

parser = argparse.ArgumentParser()
parser.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "results", "prompt_embeds.pt"))
out = parser.parse_args().out

root = snapshot_download("BAAI/nova-d48w1024-osp480")
os.makedirs(os.path.dirname(out), exist_ok=True)

tokenizer = AutoTokenizer.from_pretrained(os.path.join(root, "tokenizer"))
encoder = PhiEncoderModel.from_pretrained(
    os.path.join(root, "text_encoder"), torch_dtype=torch.float16, low_cpu_mem_usage=True
).eval()

ids = tokenizer(PROMPT, max_length=256, truncation=True).input_ids
with torch.no_grad():
    embeds = encoder(torch.tensor([ids])).last_hidden_state[0].to(torch.float16)
torch.save(embeds, out)
print(f"Saved {out}: {tuple(embeds.shape)} for {len(ids)} prompt tokens")
