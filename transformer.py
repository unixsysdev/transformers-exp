import math
import os
import time
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# Optional (CPU RSS readout). If psutil isn't available, we fall back gracefully.
try:
    import psutil
except Exception:
    psutil = None


# ---------------------------
# 4-bit Linear (cached)
# ---------------------------
class ActualReal4BitLinear(nn.Module):
    """
    Efficient 4-bit quantized linear layer with cached dequantization.

    Features:
    - 2x 4-bit per byte storage (4x smaller than fp16; 8x smaller than fp32)
    - Cached weights (fp32 on CPU, fp16 on CUDA) -> fast matmuls
    - Cache invalidates correctly on .to(device)/.cpu()/.cuda()
    - Inference-only: grads don't flow to packed buffers

    get_memory_size(include_cache=False): return packed storage; with include_cache=True
    also counts the cached dense weights currently materialized.
    """

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        packed_size = (out_features * in_features + 1) // 2
        self.register_buffer("packed_weights", torch.zeros(packed_size, dtype=torch.uint8))
        self.register_buffer("scale", torch.ones(out_features, dtype=torch.float16))
        self.register_buffer("zero_point", torch.zeros(out_features, dtype=torch.float16))

        self._cached_weight: Optional[torch.Tensor] = None
        self._cache_dirty = True

        self._initialize_packed_weights()

    def _initialize_packed_weights(self):
        # Fake random init -> quantize per-output-channel
        w = (torch.randn(self.out_features, self.in_features) * 0.02).to(torch.float32)
        for i in range(self.out_features):
            wi = w[i]
            w_min, w_max = wi.min(), wi.max()
            if (w_max - w_min).abs().item() < 1e-12:
                scale = 1.0
                zp = 8.0
            else:
                scale = max((w_max - w_min).item() / 15.0, 1e-8)
                zp = (-w_min / scale).item()
            zp = max(0.0, min(15.0, zp))

            q = torch.clamp(torch.round(wi / scale + zp), 0, 15).to(torch.uint8)
            row_stride = (self.in_features + 1) // 2
            for j in range(0, self.in_features, 2):
                idx = i * row_stride + j // 2
                lo = q[j] & 0xF
                hi = (q[j + 1] & 0xF) if (j + 1) < self.in_features else 0
                self.packed_weights[idx] = lo | (hi << 4)

            self.scale[i] = scale
            self.zero_point[i] = zp

        self._cache_dirty = True

    def _apply(self, fn):
        """Invalidate runtime cache when the module moves device/dtype."""
        super()._apply(fn)
        self._cached_weight = None
        self._cache_dirty = True
        return self

    def _get_cached_weight(self) -> torch.Tensor:
        """Dequantize and cache. Use fp16 on CUDA, fp32 on CPU for faster matmul."""
        if self._cached_weight is None or self._cache_dirty:
            device = self.packed_weights.device
            # First unpack into fp16 buffer (compact), then convert to runtime dtype
            base = torch.zeros(
                self.out_features, self.in_features, device=device, dtype=torch.float16
            )
            row_stride = (self.in_features + 1) // 2
            for i in range(self.out_features):
                zp = self.zero_point[i]
                sc = self.scale[i]
                for j in range(0, self.in_features, 2):
                    idx = i * row_stride + j // 2
                    b = self.packed_weights[idx]
                    lo = (b & 0xF).to(torch.float32)
                    hi = ((b >> 4) & 0xF).to(torch.float32)
                    base[i, j] = (lo - zp) * sc
                    if j + 1 < self.in_features:
                        base[i, j + 1] = (hi - zp) * sc

            want_dtype = torch.float16 if device.type == "cuda" else torch.float32
            self._cached_weight = base.to(want_dtype)
            self._cache_dirty = False
        return self._cached_weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self._get_cached_weight()
        if x.dtype != w.dtype:
            x = x.to(w.dtype)
        return F.linear(x, w)

    def get_memory_size(self, include_cache: bool = False) -> int:
        """Bytes: packed storage (+ optional runtime cache)."""
        packed_bytes = int(self.packed_weights.numel())  # uint8 => 1 byte each
        scale_bytes = int(self.scale.numel()) * 2  # fp16
        zero_bytes = int(self.zero_point.numel()) * 2  # fp16
        total = packed_bytes + scale_bytes + zero_bytes
        if include_cache and self._cached_weight is not None:
            total += int(self._cached_weight.numel()) * int(self._cached_weight.element_size())
        return total


# ---------------------------
# Chunked causal attention
# ---------------------------
class ActualChunkedAttention(nn.Module):
    """
    Streaming / chunked causal self-attention that matches full attention exactly
    (within numerical tolerance) and avoids O(T^2) activation memory.

    All attention math is computed in fp32 for stability, then cast back.
    """

    def __init__(self, d_model: int, n_heads: int, chunk_size: int = 128, use_4bit: bool = True):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.chunk_size = chunk_size

        if use_4bit:
            self.q_proj = ActualReal4BitLinear(d_model, d_model)
            self.k_proj = ActualReal4BitLinear(d_model, d_model)
            self.v_proj = ActualReal4BitLinear(d_model, d_model)
            self.o_proj = ActualReal4BitLinear(d_model, d_model)
        else:
            self.q_proj = nn.Linear(d_model, d_model, bias=False)
            self.k_proj = nn.Linear(d_model, d_model, bias=False)
            self.v_proj = nn.Linear(d_model, d_model, bias=False)
            self.o_proj = nn.Linear(d_model, d_model, bias=False)

        # telemetry
        self.actual_peak_memory = 0  # bytes (inner-step estimate)
        self.chunks_computed = 0

    def chunked_causal_attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """
        q, k, v: [B, H, T, D]. Returns [B, H, T, D].
        Stable streaming softmax in fp32.
        """
        # Promote to fp32 for numerics
        q32, k32, v32 = q.to(torch.float32), k.to(torch.float32), v.to(torch.float32)
        B, H, T, D = q32.shape
        device = q32.device
        out_dtype = q.dtype

        out = torch.zeros_like(q32)
        q_idx_all = torch.arange(T, device=device)
        k_idx_all = torch.arange(T, device=device)
        cs = self.chunk_size
        self.chunks_computed = 0
        max_mem = 0
        inv_sqrt_d = 1.0 / math.sqrt(D)

        for q_start in range(0, T, cs):
            q_end = min(q_start + cs, T)
            q_chunk = q32[:, :, q_start:q_end, :]  # [B,H,Q,D]

            # streaming accumulators
            m = torch.full((B, H, q_end - q_start, 1), -float("inf"), device=device)
            l = torch.zeros((B, H, q_end - q_start, 1), device=device)
            o = torch.zeros_like(q_chunk)

            max_k_pos = q_end
            for k_start in range(0, max_k_pos, cs):
                k_end = min(k_start + cs, max_k_pos)
                k_chunk = k32[:, :, k_start:k_end, :]  # [B,H,K,D]
                v_chunk = v32[:, :, k_start:k_end, :]  # [B,H,K,D]

                s = torch.matmul(q_chunk, k_chunk.transpose(-2, -1)) * inv_sqrt_d  # [B,H,Q,K]

                # causal mask: only allow k <= q
                q_idx = q_idx_all[q_start:q_end].unsqueeze(-1)  # [Q,1]
                k_idx = k_idx_all[k_start:k_end].unsqueeze(0)   # [1,K]
                causal = (q_idx >= k_idx).unsqueeze(0).unsqueeze(0)  # [1,1,Q,K]
                s = s.masked_fill(~causal, -float("inf"))

                # streaming softmax update (fp32)
                s_max = s.max(dim=-1, keepdim=True)[0]  # [B,H,Q,1]
                m_new = torch.maximum(m, s_max)
                alpha = torch.exp(m - m_new)           # [B,H,Q,1]
                p = torch.exp(s - m_new)               # [B,H,Q,K]

                o = o * alpha + torch.matmul(p, v_chunk)             # [B,H,Q,D]
                l = l * alpha + p.sum(dim=-1, keepdim=True)          # [B,H,Q,1]
                m = m_new

                cur = (q_chunk.numel() + k_chunk.numel() + v_chunk.numel() + s.numel() + p.numel()) * 4
                if cur > max_mem:
                    max_mem = cur
                self.chunks_computed += 1

                del k_chunk, v_chunk, s, p

            out[:, :, q_start:q_end, :] = o / l.clamp_min(1e-9)

        self.actual_peak_memory = max_mem
        return out.to(out_dtype)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, T, C = x.shape
        H, D = self.n_heads, self.head_dim

        q = self.q_proj(x).view(B, T, H, D).transpose(1, 2)  # [B,H,T,D]
        k = self.k_proj(x).view(B, T, H, D).transpose(1, 2)
        v = self.v_proj(x).view(B, T, H, D).transpose(1, 2)

        y = self.chunked_causal_attention(q, k, v)  # [B,H,T,D]
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.o_proj(y)


# ---------------------------
# SDPA wrapper with fallback
# ---------------------------
class FlashAttentionFallback(nn.Module):
    """
    Uses PyTorch SDPA (which routes to FlashAttention on supported GPUs),
    falls back to our chunked implementation otherwise.
    """

    def __init__(self, d_model: int, n_heads: int, chunk_size: int = 128, use_4bit: bool = True):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.chunk_size = chunk_size

        if use_4bit:
            self.q_proj = ActualReal4BitLinear(d_model, d_model)
            self.k_proj = ActualReal4BitLinear(d_model, d_model)
            self.v_proj = ActualReal4BitLinear(d_model, d_model)
            self.o_proj = ActualReal4BitLinear(d_model, d_model)
        else:
            self.q_proj = nn.Linear(d_model, d_model, bias=False)
            self.k_proj = nn.Linear(d_model, d_model, bias=False)
            self.v_proj = nn.Linear(d_model, d_model, bias=False)
            self.o_proj = nn.Linear(d_model, d_model, bias=False)

        self.use_sdpa = hasattr(F, "scaled_dot_product_attention")

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, T, C = x.shape
        H, D = self.n_heads, self.head_dim

        q = self.q_proj(x).view(B, T, H, D).transpose(1, 2)  # [B,H,T,D]
        k = self.k_proj(x).view(B, T, H, D).transpose(1, 2)
        v = self.v_proj(x).view(B, T, H, D).transpose(1, 2)

        if self.use_sdpa:
            attn = F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=True)
        else:
            # fallback
            chunked = ActualChunkedAttention(self.d_model, self.n_heads, self.chunk_size, use_4bit=False)
            # share projections so params are identical
            chunked.q_proj = self.q_proj
            chunked.k_proj = self.k_proj
            chunked.v_proj = self.v_proj
            chunked.o_proj = self.o_proj
            return chunked(x, mask)

        y = attn.transpose(1, 2).contiguous().view(B, T, C)
        return self.o_proj(y)


# ---------------------------
# Transformer block / model
# ---------------------------
class ActualOptimizedBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, chunk_size: int, use_4bit: bool, attention_cls):
        super().__init__()
        self.attn = attention_cls(d_model, n_heads, chunk_size, use_4bit)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        if use_4bit:
            self.ff1 = ActualReal4BitLinear(d_model, d_ff)
            self.ff2 = ActualReal4BitLinear(d_ff, d_model)
        else:
            self.ff1 = nn.Linear(d_model, d_ff)
            self.ff2 = nn.Linear(d_ff, d_model)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None):
        x = x + self.attn(self.ln1(x), mask)
        x = x + self.ff2(F.gelu(self.ff1(self.ln2(x))))
        return x


class ActualOptimizedTransformer(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        n_heads: int = 8,
        n_layers: int = 6,
        d_ff: int = 2048,
        max_seq_len: int = 1024,
        chunk_size: int = 128,
        use_4bit: bool = True,
        use_flash_attention: bool = True,
    ):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.chunk_size = chunk_size

        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_seq_len, d_model)

        attention_cls = FlashAttentionFallback if use_flash_attention else ActualChunkedAttention
        self.blocks = nn.ModuleList(
            [ActualOptimizedBlock(d_model, n_heads, d_ff, chunk_size, use_4bit, attention_cls) for _ in range(n_layers)]
        )

        self.ln_f = nn.LayerNorm(d_model)
        self.output = ActualReal4BitLinear(d_model, vocab_size) if use_4bit else nn.Linear(d_model, vocab_size)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        B, T = input_ids.shape
        if T > self.max_seq_len:
            raise ValueError(f"Sequence length {T} > max_seq_len {self.max_seq_len}")
        device = input_ids.device
        pos = torch.arange(T, device=device).unsqueeze(0).expand(B, -1)
        x = self.token_embedding(input_ids) + self.position_embedding(pos)
        for blk in self.blocks:
            x = blk(x)
        x = self.ln_f(x)
        return self.output(x)

    def get_real_model_size(self, include_cache: bool = False) -> dict:
        total = 0
        breakdown = {}
        for name, module in self.named_modules():
            if isinstance(module, ActualReal4BitLinear):
                size = module.get_memory_size(include_cache=include_cache)
                breakdown[name] = f"{size} bytes (4-bit{' + cache' if include_cache else ''})"
                total += size
            elif isinstance(module, nn.Linear) or isinstance(module, nn.Embedding):
                size = sum(p.numel() * p.element_size() for p in module.parameters())
                kind = "fp32" if isinstance(module, nn.Linear) else "embedding"
                breakdown[name] = f"{size} bytes ({kind})"
                total += size
        return {"total_bytes": total, "total_mb": total / (1024 * 1024), "breakdown": breakdown}


# ---------------------------
# Tests / Bench
# ---------------------------
def _device_label_from_module(m: nn.Module) -> str:
    dev = next(m.parameters()).device
    if dev.type == "cuda":
        return "GPU FlashAttention"  # SDPA will route to FA on supported GPUs
    return "CPU SDPA"


@torch.no_grad()
def test_chunked_vs_sdpa_equivalence():
    print("🧪 Testing Chunked vs SDPA Equivalence")
    print("=" * 40)

    devices = ["cpu"]
    if torch.cuda.is_available():
        devices.append("cuda")

    ok_all = True
    for dev in devices:
        print(f"\nTesting on {dev.upper()}:")
        d_model, n_heads, T, B = 128, 4, 96, 2  # a bit larger than 64 to stress numerics
        x = torch.randn(B, T, d_model, device=dev)

        sdpa = FlashAttentionFallback(d_model, n_heads, chunk_size=16, use_4bit=False).to(dev).eval()
        chunk = ActualChunkedAttention(d_model, n_heads, chunk_size=16, use_4bit=False).to(dev).eval()

        # copy weights so paths are identical
        chunk.q_proj.weight.data = sdpa.q_proj.weight.data.clone()
        chunk.k_proj.weight.data = sdpa.k_proj.weight.data.clone()
        chunk.v_proj.weight.data = sdpa.v_proj.weight.data.clone()
        chunk.o_proj.weight.data = sdpa.o_proj.weight.data.clone()

        y_sdpa = sdpa(x)
        y_chunk = chunk(x)

        max_diff = (y_sdpa - y_chunk).abs().max().item()
        mean_diff = (y_sdpa - y_chunk).abs().mean().item()
        tol = 1e-5 if dev == "cuda" else 1e-5  # now both should be very close

        print(f"  Output shape: {y_sdpa.shape}")
        print(f"  Max difference: {max_diff:.2e}")
        print(f"  Mean difference: {mean_diff:.2e}")
        print(f"  Chunked processed: {chunk.chunks_computed} chunks")
        print(f"  Peak inner attn memory (est): {chunk.actual_peak_memory/1024:.1f} KB")

        good = max_diff < tol
        ok_all &= good
        if good:
            print(f"  ✅ Match within tolerance (tol={tol:.1e})")
        else:
            print(f"  ❌ Mismatch (tol={tol:.1e})")
    return ok_all


def _rss_mb() -> float:
    if psutil is None:
        return float("nan")
    return psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)


@torch.no_grad()
def bench_model(seq_len: int, device: str = "cpu"):
    vocab = 5000
    model = ActualOptimizedTransformer(
        vocab_size=vocab,
        d_model=256,
        n_heads=8,
        n_layers=4,
        d_ff=1024,
        chunk_size=128,
        use_4bit=True,
        use_flash_attention=True,
        max_seq_len=max(2048, seq_len),
    ).to(device).eval()

    # Warm-up
    x = torch.randint(0, vocab, (1, seq_len), device=device)
    _ = model(x)
    if device == "cuda":
        torch.cuda.synchronize()

    # Peak GPU memory reset
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    rss_before = _rss_mb()
    t0 = time.time()
    _ = model(x)
    if device == "cuda":
        torch.cuda.synchronize()
    elapsed = time.time() - t0
    rss_after = _rss_mb()

    # attention impl label
    first_attn = model.blocks[0].attn
    if isinstance(first_attn, FlashAttentionFallback) and first_attn.use_sdpa:
        where = _device_label_from_module(model)
        attn_label = f"SDPA ({where})"
    else:
        attn_label = "Chunked fallback"

    # sizes
    size_packed = model.get_real_model_size(include_cache=False)
    size_with_cache = model.get_real_model_size(include_cache=True)

    print(f"    ✅ Inference: {elapsed:.3f}s")
    print(f"    📤 Output shape: {(1, seq_len, vocab)}")
    print(f"    🧠 Attention: {attn_label}")

    if device == "cuda":
        gpu_peak = torch.cuda.max_memory_allocated() / (1024 * 1024)
        print(f"    📈 Peak GPU memory (alloc): {gpu_peak:.1f} MB")
    else:
        print(f"    📈 Process RSS change (approx): {(rss_after - rss_before):.1f} MB")

    print(f"    💾 Model size (packed 4-bit only): {size_packed['total_mb']:.2f} MB")
    print(f"    💾 Model size (+runtime cache):   {size_with_cache['total_mb']:.2f} MB")

    # theoretical chunking savings (activations)
    cs = model.chunk_size
    full = seq_len * seq_len
    chunked = cs * cs
    print(f"    🎯 Theoretical chunked attention savings: {full / chunked:.1f}x")


def test_actual_optimizations():
    print("🚀 TESTING PRODUCTION-READY OPTIMIZATIONS")
    print("=" * 50)
    print("SDPA (GPU->FlashAttention) + numerically-correct chunked fallback + efficient 4-bit caching\n")

    ok = test_chunked_vs_sdpa_equivalence()
    if not ok:
        print("\n⚠️  Equivalence test did not pass on all devices. (Proceeding with bench anyway.)\n")

    tests = [
        {"name": "Short", "seq_len": 256},
        {"name": "Medium", "seq_len": 512},
        {"name": "Long", "seq_len": 1024},
    ]

    for cfg in tests:
        print(f"📏 Testing {cfg['name']} sequence ({cfg['seq_len']} tokens):")
        for dev in (["cpu"] + (["cuda"] if torch.cuda.is_available() else [])):
            print(f"  Device: {dev.upper()}")
            bench_model(cfg["seq_len"], device=dev)
        print("")

    print("🎉 DONE")


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    test_actual_optimizations()
