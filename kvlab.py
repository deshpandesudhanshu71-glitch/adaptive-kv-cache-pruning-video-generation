import math
import types

import torch
from torch import nn

FRAME = 360  # tokens per latent frame: 768x480 / 32 per side = 24x15


# Storage: fp16 as NOVA does, or quantised per token.


def quantize(x, bits):
    """Per token, per head symmetric quantisation over the head dimension."""
    qmax = 2 ** (bits - 1) - 1
    scale = x.abs().amax(-1, keepdim=True).float().clamp_min(1e-6) / qmax
    q = (x.float() / scale).round_().clamp_(-qmax, qmax).to(torch.int8)
    if bits == 4:  # pack two 4 bit values into one byte along the head dimension
        u = (q + 8).to(torch.uint8)
        q = u[..., 0::2] | (u[..., 1::2] << 4)
    return q, scale.half()


def dequantize(q, scale, bits):
    if bits == 4:
        lo, hi = (q & 15).to(torch.int8) - 8, (q >> 4).to(torch.int8) - 8
        q = torch.stack([lo, hi], -1).flatten(-2)
    return (q.half() * scale)


class Store:
    """Keys and values of one layer, plus a group id and an attention score per token."""

    def __init__(self, bits=16):
        self.bits = bits
        self.k = self.v = self.ks = self.vs = None
        self.gid = self.score = None  # (B, H, T)

    def append(self, k, v, gid, score):
        if self.bits < 16:
            (k, ks), (v, vs) = quantize(k, self.bits), quantize(v, self.bits)
        cat = (lambda a, b: b if a is None else torch.cat([a, b], 2))
        self.k, self.v = cat(self.k, k), cat(self.v, v)
        if self.bits < 16:
            self.ks, self.vs = cat(self.ks, ks), cat(self.vs, vs)
        self.gid, self.score = cat(self.gid, gid), cat(self.score, score)

    def read(self):
        if self.k is None:
            return None, None
        if self.bits < 16:
            return dequantize(self.k, self.ks, self.bits), dequantize(self.v, self.vs, self.bits)
        return self.k, self.v

    def keep(self, idx):
        """Keep the tokens at idx, shape (B, H, T') in original order."""
        take = lambda t: t.gather(2, idx.unsqueeze(-1).expand(-1, -1, -1, t.size(-1)))
        self.k, self.v = take(self.k), take(self.v)
        if self.bits < 16:
            self.ks, self.vs = take(self.ks), take(self.vs)
        self.gid, self.score = self.gid.gather(2, idx), self.score.gather(2, idx)

    def keep_mask(self, mask):
        """Same as keep, for a (T,) mask shared by every head."""
        self.k, self.v = self.k[:, :, mask], self.v[:, :, mask]
        if self.bits < 16:
            self.ks, self.vs = self.ks[:, :, mask], self.vs[:, :, mask]
        self.gid, self.score = self.gid[:, :, mask], self.score[:, :, mask]

    def nbytes(self):
        ts = [self.k, self.v, self.ks, self.vs]
        return sum(t.numel() * t.element_size() for t in ts if t is not None)

    def tokens(self):
        return 0 if self.k is None else self.k.size(2)


# Attention statistics.


@torch.no_grad()
def attention_received(q, k, heads_per_chunk=4):
    """Mean attention each key gets from the queries, (B, H, T). Computed in fp32 chunks."""
    out, scale = [], 1.0 / math.sqrt(q.size(-1))
    for h in range(0, q.size(1), heads_per_chunk):
        s = torch.matmul(q[:, h : h + heads_per_chunk].float(),
                         k[:, h : h + heads_per_chunk].float().transpose(-1, -2)) * scale
        out.append(s.softmax(-1).mean(2))
        del s
    return torch.cat(out, 1)


def group_mass(attn, gid, n_groups):
    """Sum the attention per group, (B, H, n_groups)."""
    out = attn.new_zeros(attn.shape[:2] + (n_groups,))
    return out.scatter_add_(2, gid, attn)


# The lab: one object holds the method, the per layer stores and the logs.


class Lab:
    def __init__(self, method="full", n=4, store_bits=16, tau=0.9, warmup=8, n_max=8,
                 sink_frames=2, recheck=0, analyse=False, shadow=()):
        self.method, self.n, self.bits = method, n, store_bits
        self.sink_frames = sink_frames
        self.tau, self.warmup, self.n_max = tau, warmup, n_max
        self.recheck = recheck  # 0 = windows stay as chosen after the warm-up
        self.analyse, self.shadow = analyse, list(shadow)
        self.reset()

    def reset(self):
        self.stores, self.step = {}, {}
        self.windows = {}       # adaptive: chosen window per layer
        self.profile = {}       # adaptive: attention to past frames by distance, per layer
        self.recent = {}        # recheck: the same, since the last decision
        self.window_log = []    # adaptive: (step, layer, window) every time a window is set
        self.mass_log = []      # analyse: (step, layer, B, H, groups)
        self.shadow_log = []    # analyse: (step, layer, config, relative error)

    @property
    def needs_attention(self):
        return self.method in ("h2o", "adaptive") or self.analyse

    # Called by the patched attention forward of temporal block `layer`.

    def __call__(self, layer, q, k_new, v_new):
        st = self.stores.get(layer)
        step = self.step.get(layer, -1) + 1
        self.step[layer] = step
        B, H, T, _ = k_new.shape
        if st is None:
            st = self.stores[layer] = Store(self.bits)
            text = T - FRAME
            gid = torch.cat([k_new.new_zeros(text, dtype=torch.long),
                             k_new.new_ones(FRAME, dtype=torch.long)])
        else:
            assert T == FRAME, f"expected {FRAME} new tokens, got {T}"
            gid = torch.full((T,), step + 1, dtype=torch.long, device=k_new.device)
        gid = gid.expand(B, H, T)
        k_old, v_old = st.read()
        k = k_new if k_old is None else torch.cat([k_old, k_new], 2)
        v = v_new if v_old is None else torch.cat([v_old, v_new], 2)
        o = nn.functional.scaled_dot_product_attention(q, k, v)

        attn = attention_received(q, k) if self.needs_attention else None
        st.append(k_new, v_new, gid, k_new.new_zeros(B, H, T, dtype=torch.float32))
        if attn is not None:
            st.score += attn  # H2O: attention adds up over the steps
            n_groups = step + 2
            mass = group_mass(attn, st.gid, n_groups)
            if self.analyse:
                self.mass_log.append((step, layer, mass.cpu()))
                self.run_shadow(step, layer, q, k, v, o, st, attn)
            if self.method == "adaptive" and step >= 2:
                self.update_profile(layer, step, mass)
        self.prune(layer, step, st)
        return o

    # Pruning rules, applied after the step so the next frame sees the reduced cache.

    def prune(self, layer, step, st):
        cur = step + 1
        if self.method == "window":
            self.keep_window(st, cur, self.n)
        elif self.method == "sinkwindow":
            self.keep_window(st, cur, self.n, 1 + self.sink_frames)
        elif self.method == "h2o":
            budget = FRAME * (self.n - 1)  # heavy hitters on top of sinks and current frame
            other = (st.gid >= 2) & (st.gid < cur)  # per head: heads keep different tokens
            if int(other[0, 0].sum()) > budget:
                score = st.score.masked_fill(~other, -1.0)
                top = score.topk(budget, dim=2).indices
                fixed = (~other[0, 0]).nonzero().squeeze(1).expand(st.score.shape[:2] + (-1,))
                idx = torch.cat([fixed, top], 2).sort(2).values
                st.keep(idx)
        elif self.method == "adaptive":
            if step == self.warmup:
                self.choose_windows()
                self.window_log.append((step, layer, self.windows[layer]))
                self.recent.pop(layer, None)
            elif self.recheck and step > self.warmup and (step - self.warmup) % self.recheck == 0:
                self.recheck_window(layer, step)
            if layer in self.windows:
                self.keep_window(st, cur, self.windows[layer], 1 + self.sink_frames)

    @staticmethod
    def keep_window(st, cur, n, last_sink=1):
        g = st.gid[0, 0]
        mask = (g <= last_sink) | (g > cur - n)
        if not bool(mask.all()):
            st.keep_mask(mask)

    def update_profile(self, layer, step, mass):
        """Attention from the current frame to each past frame, indexed by distance."""
        cur = step + 1
        first = 2 + self.sink_frames  # frame sinks are always kept, so leave them out
        if cur <= first:
            return
        past = mass[0, :, first:cur].mean(0)  # prompt branch, mean over heads
        dist = past.flip(0)               # distance 1 = previous frame
        prof = self.profile.setdefault(layer, torch.zeros(64, device=mass.device))
        prof[: dist.numel()] += dist
        if self.recheck and step > self.warmup:
            rec = self.recent.setdefault(layer, torch.zeros(64, device=mass.device))
            rec[: dist.numel()] += dist  # deleted frames add 0

    def choose_windows(self):
        for layer, prof in self.profile.items():
            p = prof[prof > 0]
            if p.numel() == 0:
                self.windows[layer] = 1
                continue
            cover = (p.cumsum(0) / p.sum()).cpu()
            d = int((cover < self.tau).sum()) + 1  # previous frames needed to reach tau
            self.windows[layer] = max(1, min(self.n_max, d + 1))  # + the current frame

    def recheck_window(self, layer, step):
        """Resize one layer's window from its attention since the last decision.

        Pruning runs after a step, so the next frame sees `w` previous frames: the d it needed
        to reach tau plus one spare. Deleted frames can't be measured, but the spare can. If
        tau now takes all w frames, the spare is in use and the window grows by one; if it
        takes fewer than w - 1, the window shrinks.
        """
        w, rec = self.windows[layer], self.recent.pop(layer, None)
        if rec is None or float(rec[:w].sum()) == 0:
            return
        cover = (rec[:w].cumsum(0) / rec[:w].sum()).cpu()
        d = int((cover < self.tau).sum()) + 1
        self.windows[layer] = max(1, min(self.n_max, d + 1))
        self.window_log.append((step, layer, self.windows[layer]))

    # Teacher forced comparison: what each method's cache would give at this step if the
    # cache so far had been the full one. Only used with method full.

    def run_shadow(self, step, layer, q, k, v, o, st, attn):
        cur, g = step + 1, st.gid[0, 0]
        ref = o.float()
        norm = ref.norm()
        for cfg in self.shadow:
            kind, val = cfg
            if kind == "window":
                mask = (g <= 1) | (g > cur - val)
                ks, vs = k[:, :, mask], v[:, :, mask]
            elif kind == "sinkwindow":
                mask = (g <= 1 + self.sink_frames) | (g > cur - val)
                ks, vs = k[:, :, mask], v[:, :, mask]
            elif kind == "h2o":
                budget = FRAME * (val - 1)
                other = (g >= 2) & (g < cur)
                if int(other.sum()) <= budget:
                    ks, vs = k, v
                else:
                    # scores from earlier steps only, as in a real run
                    score = (st.score - attn).masked_fill(
                        ~other.expand_as(st.score), -1.0)
                    top = score.topk(budget, dim=2).indices
                    fixed = (~other).nonzero().squeeze(1).expand(st.score.shape[:2] + (-1,))
                    idx = torch.cat([fixed, top], 2).sort(2).values.unsqueeze(-1)
                    idx = idx.expand(-1, -1, -1, k.size(-1))
                    ks, vs = k.gather(2, idx), v.gather(2, idx)
            elif kind == "quant":
                ks = dequantize(*quantize(k, val), val)
                vs = dequantize(*quantize(v, val), val)
            else:
                continue
            os_ = nn.functional.scaled_dot_product_attention(q, ks, vs).float()
            err = float((os_ - ref).norm() / norm)
            self.shadow_log.append((step, layer, f"{kind}{val}", err))

    # Reporting.

    def cache_bytes(self):
        total = sum(s.nbytes() for s in self.stores.values())
        if self.method == "h2o":  # H2O has to keep its fp32 scores as well
            total += sum(s.score.numel() * 4 for s in self.stores.values())
        return total

    def cache_tokens(self):
        return [self.stores[l].tokens() for l in sorted(self.stores)]


def install(model, lab):
    """Route the temporal blocks' attention through lab. Returns a function that undoes it."""
    blocks = model.video_encoder.blocks
    originals = []
    for layer, blk in enumerate(blocks):
        attn = blk.attn
        originals.append(attn.forward)

        def forward(self, x, layer=layer):
            shape = [-1, x.size(1), 3, self.num_heads, self.head_dim]
            q, k, v = self.qkv(x).view(shape).permute(2, 0, 3, 1, 4).unbind(dim=0)
            q, k = (self.pe_func(q), self.pe_func(k)) if self.pe_func else (q, k)
            if self.cache_kv:  # True or a list while NOVA generates a video
                o = lab(layer, q, k, v)
            else:
                o = nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=self.attn_mask)
            return self.proj(o.transpose(1, 2).flatten(2))

        attn.forward = types.MethodType(forward, attn)

    def undo():
        for blk, f in zip(blocks, originals):
            blk.attn.forward = f

    return undo
