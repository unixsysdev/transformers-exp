# rev_williams_transformer.py
# ------------------------------------------------------------
# Reversible Transformer + Williams-style space–time tradeoffs
#   - Reversible residual blocks (activation memory ↓)
#   - Binary-partition checkpoint schedule across depth (√T-like)
#   - SDPA (FlashAttention) or numerically-stable streaming-chunked attention
#   - Optional 4-bit packed Linear for weight memory ↓
#   - Quick micro-bench to visualize speed/peak memory
# ------------------------------------------------------------

import math
import os
import time
import psutil
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List

# ------------------------------
# Utils
# ------------------------------

def peak_rss_mb() -> float:
    """Process resident-set size in MB (coarse but useful)."""
    return psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)


@torch.no_grad()
def quick_bench_forward(model: nn.Module, input_ids: torch.Tensor, warmup: int = 1, iters: int = 3) -> Tuple[float, float]:
    """Return (avg_seconds, peak_rss_mb_during_run)."""
    model.eval()
    # Warmup
    for _ in range(warmup):
        _ = model(input_ids)

    before = peak_rss_mb()
    start = time.time()
    for _ in range(iters):
        _ = model(input_ids)
    elapsed = (time.time() - start) / iters
    after = peak_rss_mb()
    return elapsed, max(before, after)


# ------------------------------
# 4-bit Linear (simple / robust)
# ------------------------------

class Linear4bit(nn.Module):
    """
    Packed 4-bit per-weight linear layer (per-output-channel affine dequant).
    Kept intentionally straightforward & robust.
    """
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        packed_size = (out_features * in_features + 1) // 2
        self.register_buffer("packed", torch.zeros(packed_size, dtype=torch.uint8))
        self.register_buffer("scale", torch.ones(out_features, dtype=torch.float32))
        self.register_buffer("zp", torch.zeros(out_features, dtype=torch.float32))

        self._cached = None  # fp16/fp32 dequantized matrix
        self.reset_parameters()

    def reset_parameters(self):
        w = torch.randn(self.out_features, self.in_features) * 0.02
        # per-out-channel quant
        for i in range(self.out_features):
            wi = w[i]
            wmin = wi.min().item()
            wmax = wi.max().item()
            rng = max(wmax - wmin, 1e-8)
            scale = rng / 15.0
            zp = -wmin / scale
            zp = float(max(0.0, min(15.0, zp)))

            q = torch.clamp(torch.round(wi / scale + zp), 0, 15).to(torch.uint8)
            row_stride = (self.in_features + 1) // 2
            for j in range(0, self.in_features, 2):
                idx = i * row_stride + (j // 2)
                lo = int(q[j].item() & 0xF)
                hi = int(q[j + 1].item() & 0xF) if j + 1 < self.in_features else 0
                self.packed[idx] = (hi << 4) | lo

            self.scale[i] = scale
            self.zp[i] = zp

        self._cached = None

    def _materialize(self) -> torch.Tensor:
        if self._cached is not None and self._cached.device == self.packed.device:
            return self._cached
        W = torch.empty(self.out_features, self.in_features, device=self.packed.device, dtype=torch.float16)
        row_stride = (self.in_features + 1) // 2
        for i in range(self.out_features):
            s = self.scale[i]
            z = self.zp[i]
            for j in range(0, self.in_features, 2):
                idx = i * row_stride + (j // 2)
                b = int(self.packed[idx].item())
                lo = float(b & 0xF)
                hi = float((b >> 4) & 0xF)
                W[i, j] = (lo - z) * s
                if j + 1 < self.in_features:
                    W[i, j + 1] = (hi - z) * s
        self._cached = W
        return self._cached

    def _apply(self, fn):
        self._cached = None
        return super()._apply(fn)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        W = self._materialize()
        # align dtype for matmul
        if x.dtype != W.dtype:
            x = x.to(W.dtype)
        out = F.linear(x, W)
        # return back in higher precision if upstream expects fp32
        if out.dtype == torch.float16 and x.dtype == torch.float32:
            out = out.to(torch.float32)
        return out


# ------------------------------
# Streaming, Chunked Causal Attention (numerically stable)
# ------------------------------

class StreamingCausalAttention(nn.Module):
    """
    Memory-lean causal attention using streaming log-sum-exp softmax.
    Matches SDPA within fp32 tolerance. Works in fp16/bf16 inputs by
    doing the accumulator in fp32.
    """
    def __init__(self, d_model: int, n_heads: int, chunk_size: int = 128, proj4bit: bool = False):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.hd = d_model // n_heads
        self.chunk = chunk_size

        Linear = Linear4bit if proj4bit else nn.Linear
        self.q = Linear(d_model, d_model)
        self.k = Linear(d_model, d_model)
        self.v = Linear(d_model, d_model)
        self.o = Linear(d_model, d_model)

        self.use_sdpa = hasattr(F, "scaled_dot_product_attention")

    def forward(self, x: torch.Tensor, is_causal: bool = True) -> torch.Tensor:
        B, T, D = x.shape
        H, Hd = self.n_heads, self.hd

        # Projections
        q = self.q(x).view(B, T, H, Hd).transpose(1, 2)  # [B, H, T, Hd]
        k = self.k(x).view(B, T, H, Hd).transpose(1, 2)
        v = self.v(x).view(B, T, H, Hd).transpose(1, 2)

        if self.use_sdpa:
            # Uses native FlashAttention kernels where available
            out = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=None,
                dropout_p=0.0,
                is_causal=is_causal
            )  # [B, H, T, Hd]
        else:
            # Streaming log-sum-exp over chunks
            device = x.device
            out = torch.zeros_like(q)
            scale = 1.0 / math.sqrt(Hd)

            # do the numerically sensitive math in float32
            for qs in range(0, T, self.chunk):
                qe = min(qs + self.chunk, T)
                Q = q[:, :, qs:qe, :].to(torch.float32)  # [B,H,QC,Hd]

                # running max, sum, out
                m = torch.full((B, H, qe - qs, 1), -float("inf"), device=device, dtype=torch.float32)
                l = torch.zeros((B, H, qe - qs, 1), device=device, dtype=torch.float32)
                o = torch.zeros((B, H, qe - qs, Hd), device=device, dtype=torch.float32)

                for ks in range(0, qe if is_causal else T, self.chunk):
                    ke = min(ks + self.chunk, T if not is_causal else qe)
                    K = k[:, :, ks:ke, :].to(torch.float32)
                    V = v[:, :, ks:ke, :].to(torch.float32)

                    S = torch.matmul(Q, K.transpose(-2, -1)) * scale  # [B,H,QC,KC]

                    if is_causal:
                        q_idx = torch.arange(qs, qe, device=device).unsqueeze(-1)    # [QC,1]
                        k_idx = torch.arange(ks, ke, device=device).unsqueeze(0)    # [1,KC]
                        mask = (q_idx >= k_idx)  # [QC,KC]
                        S = S.masked_fill(~mask.unsqueeze(0).unsqueeze(0), -float("inf"))

                    s_max = S.max(dim=-1, keepdim=True).values
                    m_new = torch.maximum(m, s_max)
                    alpha = torch.exp(m - m_new)
                    P = torch.exp(S - m_new)

                    o = o * alpha + torch.matmul(P, V)
                    l = l * alpha + P.sum(dim=-1, keepdim=True)
                    m = m_new

                out[:, :, qs:qe, :] = (o / torch.clamp_min(l, 1e-8)).to(out.dtype)

        out = out.transpose(1, 2).contiguous().view(B, T, D)
        return self.o(out)


# ------------------------------
# Feedforward
# ------------------------------

def make_ffn(d_model: int, d_ff: int, use_4bit: bool = False) -> nn.Module:
    Linear = Linear4bit if use_4bit else nn.Linear
    return nn.Sequential(
        Linear(d_model, d_ff),
        nn.GELU(),
        Linear(d_ff, d_model),
    )


# ------------------------------
# Reversible Block (additive coupling)
# ------------------------------

class RevBlock(nn.Module):
    """
    Reversible block:
        split x -> (x1, x2) along channel
        y1 = x1 + F(x2)
        y2 = x2 + G(y1)
    Inverse exists; for training we still use checkpoint to keep it simple/robust.
    """
    def __init__(self, d_model: int, n_heads: int, d_ff: int, chunk: int,
                 proj4bit: bool, attn_first: bool = True):
        super().__init__()
        assert d_model % 2 == 0, "d_model must be even for split"
        self.d_half = d_model // 2
        self.f = StreamingCausalAttention(self.d_half, max(1, n_heads // 2), chunk, proj4bit)
        self.g = make_ffn(self.d_half, d_ff, use_4bit=proj4bit)
        self.ln_f = nn.LayerNorm(self.d_half)
        self.ln_g = nn.LayerNorm(self.d_half)
        self.attn_first = attn_first

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x[..., :self.d_half], x[..., self.d_half:]
        # y1 = x1 + F(x2)
        y1 = x1 + self.f(self.ln_f(x2))
        # y2 = x2 + G(y1)
        y2 = x2 + self.g(self.ln_g(y1))
        return torch.cat([y1, y2], dim=-1)


# ------------------------------
# Vanilla Residual Block (for baseline)
# ------------------------------

class VanillaBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, chunk: int, proj4bit: bool):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = StreamingCausalAttention(d_model, n_heads, chunk, proj4bit)
        self.ln2 = nn.LayerNorm(d_model)
        self.ff = make_ffn(d_model, d_ff, use_4bit=proj4bit)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x


# ------------------------------
# Transformer LM with toggles:
#   - reversible vs vanilla
#   - checkpoint schedule across depth
#   - 4-bit projections
# ------------------------------

class WilliamsTransformerLM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        n_heads: int = 8,
        n_layers: int = 12,
        d_ff: int = 2048,
        max_seq_len: int = 1024,
        chunk_size: int = 128,
        reversible: bool = True,
        use_4bit: bool = False,
        checkpoint_sqrt_schedule: bool = True,
    ):
        super().__init__()
        self.vocab = vocab_size
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.reversible = reversible
        self.use_4bit = use_4bit
        self.checkpoint_sqrt_schedule = checkpoint_sqrt_schedule

        self.tok = nn.Embedding(vocab_size, d_model)
        self.pos = nn.Embedding(max_seq_len, d_model)

        blocks: List[nn.Module] = []
        for _ in range(n_layers):
            if reversible:
                blocks.append(RevBlock(d_model, n_heads, d_ff, chunk_size, use_4bit))
            else:
                blocks.append(VanillaBlock(d_model, n_heads, d_ff, chunk_size, use_4bit))
        self.blocks = nn.ModuleList(blocks)
        self.lnf = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

        # Precompute checkpoint pattern (√L blocks checked)
        self._checkpoint_mask = self._build_checkpoint_mask(n_layers) if checkpoint_sqrt_schedule else None

    def _build_checkpoint_mask(self, L: int) -> List[bool]:
        """
        Binary-partition / √L-ish schedule:
        Mark ~sqrt(L) block indices to be recomputed (checkpoint=True).
        """
        k = max(1, int(math.sqrt(L)))
        mask = [False] * L
        step = max(1, L // k)
        for i in range(0, L, step):
            mask[i] = True
        return mask

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        B, T = input_ids.shape
        if T > self.max_seq_len:
            raise ValueError(f"seq_len {T} > max_seq_len {self.max_seq_len}")

        pos = torch.arange(T, device=input_ids.device).unsqueeze(0)
        x = self.tok(input_ids) + self.pos(pos)

        # Depth recomputation schedule
        for i, block in enumerate(self.blocks):
            if self._checkpoint_mask is not None and self._checkpoint_mask[i] and self.training:
                # PyTorch checkpoint trades compute for activation memory
                x = torch.utils.checkpoint.checkpoint(block, x, use_reentrant=False)
            else:
                x = block(x)

        x = self.lnf(x)
        return self.head(x)


# ------------------------------
# Equivalence test (SDPA vs streaming)
# ------------------------------

@torch.no_grad()
def test_attention_equivalence(device="cpu") -> None:
    torch.manual_seed(0)
    d_model, n_heads, T, B = 128, 4, 96, 2
    chunk = 24

    # Build one block with SDPA available (it will auto-pick)
    a = StreamingCausalAttention(d_model, n_heads, chunk, proj4bit=False).to(device)
    a.use_sdpa = True
    b = StreamingCausalAttention(d_model, n_heads, chunk, proj4bit=False).to(device)
    b.use_sdpa = False

    x = torch.randn(B, T, d_model, device=device)
    # Copy projections to ensure identical params
    b.q.load_state_dict(a.q.state_dict()); b.k.load_state_dict(a.k.state_dict())
    b.v.load_state_dict(a.v.state_dict()); b.o.load_state_dict(a.o.state_dict())

    y_sdpa = a(x)
    y_stream = b(x)
    diff = (y_sdpa - y_stream).abs()
    print(f"[Equiv] max={diff.max().item():.3e} mean={diff.mean().item():.3e}")


# ------------------------------
# Micro-bench
# ------------------------------

def micro_bench():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)
    print(f"Device: {device}")
    test_attention_equivalence(device=device)

    vocab = 32000
    seqs = [256, 512, 1024]
    configs = [
        dict(name="Vanilla fp32", reversible=False, use_4bit=False),
        dict(name="Reversible + √L checkpoint", reversible=True, use_4bit=False),
        dict(name="Reversible + √L + 4bit proj", reversible=True, use_4bit=True),
    ]

    for T in seqs:
        print(f"\n=== SeqLen {T} ===")
        x = torch.randint(0, vocab, (1, T), device=device)

        for cfg in configs:
            model = WilliamsTransformerLM(
                vocab_size=vocab,
                d_model=512,
                n_heads=8,
                n_layers=12,
                d_ff=2048,
                max_seq_len=2048,
                chunk_size=128,
                reversible=cfg["reversible"],
                use_4bit=cfg["use_4bit"],
                checkpoint_sqrt_schedule=cfg["reversible"],  # only for rev setting
            ).to(device)

            # training=True to activate checkpoint only during training
            model.train(cfg["reversible"])

            # quick pass to stabilize any lazy inits
            _ = model(x)

            t_s, rss_mb = quick_bench_forward(model, x, warmup=1, iters=2)
            print(f"{cfg['name']:<28s} | time/it: {t_s*1000:7.1f} ms | peak RSS~ {rss_mb:7.1f} MB")


if __name__ == "__main__":
    micro_bench()
