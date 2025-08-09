#!/usr/bin/env python3
# enhanced_transformer.py

import argparse
import csv
import math
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

# ----------------------------
# Optional memory tracking (RSS; noisy on CPU)
# ----------------------------
try:
    import psutil
    def peak_memory_mb():
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / (1024 * 1024)
except Exception:
    def peak_memory_mb():
        return 0.0


# ----------------------------
# "Int4-style" Linear (int8 storage + per-row float scale)
# ----------------------------
class Int4Linear(nn.Module):
    """
    Storage-efficient linear:
      - weight_int8 stored as a buffer (no grads)
      - per-output-channel scale as trainable Parameter (float)
      - optional float bias
    Dequantization at runtime: W = (weight_int8.float() * scale[:, None])
    """
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        w_int = torch.randint(-8, 8, (out_features, in_features), dtype=torch.int8)
        self.register_buffer("weight_int8", w_int, persistent=True)

        self.scale = nn.Parameter(torch.ones(out_features, dtype=torch.float32))
        self.bias = nn.Parameter(torch.zeros(out_features, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.weight_int8.to(torch.float32) * self.scale.unsqueeze(1)
        if x.dtype != torch.float32:
            x = x.float()
        return F.linear(x, w, self.bias)


# ----------------------------
# Williams-style Attention (cache | grouped | recompute)
# ----------------------------
class WilliamsAttention(nn.Module):
    """
    Multi-head attention with three KV storage policies:

      - cache:     standard full K/V resident (baseline)
      - grouped:   GQA-style sharing (kv_heads < num_heads)
      - recompute: Williams-style space saving:
                   * no full K/V storage
                   * 2D tiling (Q-chunks × K-chunks)
                   * streaming softmax per Q-chunk, causal

    The recompute variant bounds temps to O(Qc*Kc) instead of O(T*Kc).
    """
    def __init__(self, d_model: int, num_heads: int, kv_policy: str = "cache",
                 k_chunk: int | None = None, kv_heads: int | None = None, q_chunk: int | None = None):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.kv_policy = kv_policy
        self.k_chunk = k_chunk
        self.q_chunk = q_chunk
        self.kv_heads = kv_heads if kv_heads is not None else num_heads
        assert self.num_heads % self.kv_heads == 0, "num_heads must be divisible by kv_heads"

        self.q_proj = Int4Linear(d_model, d_model)
        self.k_proj = Int4Linear(d_model, d_model)
        self.v_proj = Int4Linear(d_model, d_model)
        self.o_proj = Int4Linear(d_model, d_model)

    def _sdpa_or_matmul(self, q, k, v, is_causal=True):
        if hasattr(F, "scaled_dot_product_attention"):
            return F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=is_causal)
        else:
            B, H, T, D = q.shape
            scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(D)
            if is_causal:
                idx = torch.arange(T, device=q.device)
                mask = idx.unsqueeze(0) >= idx.unsqueeze(1)  # (T,T) lower-tri
                scores = scores.masked_fill(~mask, float("-inf"))
            probs = F.softmax(scores, dim=-1)
            return probs @ v

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        H = self.num_heads
        D = self.head_dim
        device = x.device

        # Q always once
        q = self.q_proj(x).view(B, T, H, D).transpose(1, 2)  # (B,H,T,D)

        if self.kv_policy == "cache":
            k = self.k_proj(x).view(B, T, H, D).transpose(1, 2)
            v = self.v_proj(x).view(B, T, H, D).transpose(1, 2)
            attn = self._sdpa_or_matmul(q, k, v, is_causal=True)

        elif self.kv_policy == "grouped":
            kvh = self.kv_heads
            g = H // kvh
            k_full = self.k_proj(x).view(B, T, H, D).transpose(1, 2)
            v_full = self.v_proj(x).view(B, T, H, D).transpose(1, 2)
            k_grp = k_full.view(B, kvh, g, T, D).mean(dim=2)  # (B,kvh,T,D)
            v_grp = v_full.view(B, kvh, g, T, D).mean(dim=2)  # (B,kvh,T,D)
            k = k_grp.repeat_interleave(g, dim=1)  # (B,H,T,D)
            v = v_grp.repeat_interleave(g, dim=1)  # (B,H,T,D)
            attn = self._sdpa_or_matmul(q, k, v, is_causal=True)

        elif self.kv_policy == "recompute":
            # 2D tiling: Q-chunks × K-chunks; streaming softmax per Q-chunk
            Qc = self.q_chunk or max(1, int(math.sqrt(max(T, 2))))
            Kc = self.k_chunk or max(1, int(math.sqrt(max(T, 2) * max(1, int(math.log2(max(T, 2)))))))
            scale = 1.0 / math.sqrt(D)

            out = torch.zeros_like(q)

            for qs in range(0, T, Qc):
                qe = min(qs + Qc, T)
                q_tile = q[:, :, qs:qe, :]  # (B,H,Qc,D)

                # streaming accumulators for this Q tile
                m = torch.full((B, H, qe - qs, 1), float("-inf"), device=device)
                l = torch.zeros((B, H, qe - qs, 1), device=device)
                o = torch.zeros_like(q_tile)

                # positions for causal mask
                q_idx = torch.arange(qs, qe, device=device).view(1, 1, -1, 1)

                for ks in range(0, T, Kc):
                    ke = min(ks + Kc, T)

                    # recompute K/V chunk on-the-fly
                    k_chunk = self.k_proj(x[:, ks:ke, :]).view(B, ke - ks, H, D).transpose(1, 2)  # (B,H,Kc,D)
                    v_chunk = self.v_proj(x[:, ks:ke, :]).view(B, ke - ks, H, D).transpose(1, 2)  # (B,H,Kc,D)

                    # scores for this (Qc × Kc) block
                    s = torch.matmul(q_tile, k_chunk.transpose(-2, -1)) * scale  # (B,H,Qc,Kc)

                    # causal within block: key index must be <= query index
                    k_idx = torch.arange(ks, ke, device=device).view(1, 1, 1, -1)
                    mask = (k_idx <= q_idx)  # (1,1,Qc,Kc)
                    s = s.masked_fill(~mask, float("-inf"))

                    # streaming softmax update
                    s_max = s.max(dim=-1, keepdim=True).values     # (B,H,Qc,1)
                    m_new = torch.maximum(m, s_max)
                    alpha = torch.exp(m - m_new)

                    p = torch.exp(s - m_new)                       # (B,H,Qc,Kc)
                    o = o * alpha + torch.matmul(p, v_chunk)       # (B,H,Qc,D)
                    l = l * alpha + p.sum(dim=-1, keepdim=True)    # (B,H,Qc,1)
                    m = m_new

                    # free temps asap
                    del k_chunk, v_chunk, s, s_max, alpha, p, mask, k_idx

                # finalize this Q tile
                out[:, :, qs:qe, :] = o / (l + 1e-8)

                del q_tile, o, l, m, q_idx

            attn = out

        else:
            raise ValueError(f"Unknown kv_policy: {self.kv_policy}")

        y = attn.transpose(1, 2).contiguous().view(B, T, C)
        return self.o_proj(y)


# ----------------------------
# Transformer Block
# ----------------------------
class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, mlp_ratio: int = 4,
                 kv_policy: str = "cache", k_chunk: int | None = None,
                 kv_heads: int | None = None, q_chunk: int | None = None):
        super().__init__()
        self.attn = WilliamsAttention(d_model, num_heads, kv_policy,
                                      k_chunk=k_chunk, kv_heads=kv_heads, q_chunk=q_chunk)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            Int4Linear(d_model, d_model * mlp_ratio),
            nn.GELU(),
            Int4Linear(d_model * mlp_ratio, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


# ----------------------------
# Benchmark
# ----------------------------
def theoretical_block_temp_mb(B, H, Qc, Kc, D, dtype_bytes=4):
    # Rough upper bound for a single (Q,K) tile’s score/prob temporaries + output tile
    # temps ~ B*H*(Qc*Kc  + Qc*D) * dtype_bytes
    elems = B * H * (Qc * Kc + Qc * D)
    return elems * dtype_bytes / (1024 * 1024)

def benchmark(seq_len: int, kv_policy: str, runs: int = 1,
              k_chunk: int | None = None, kv_heads: int | None = None,
              q_chunk: int | None = None, print_result: bool = True):
    torch.manual_seed(0)
    model = TransformerBlock(128, 8, kv_policy=kv_policy, k_chunk=k_chunk, kv_heads=kv_heads, q_chunk=q_chunk)
    x = torch.randn(1, seq_len, 128)

    times, mems = [], []

    for _ in range(runs):
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
        start_mem = peak_memory_mb()
        t0 = time.time()
        with torch.no_grad():
            _ = model(x)
        t1 = time.time()
        end_mem = peak_memory_mb()
        times.append(t1 - t0)
        mems.append(max(0.0, end_mem - start_mem))

    t_avg = sum(times) / runs
    m_avg = sum(mems) / runs

    if print_result:
        extras = ""
        if kv_policy == "recompute":
            # give a quick theoretical temp footprint for reference
            B, H, D = 1, 8, 128
            Qc = q_chunk or max(1, int(math.sqrt(max(seq_len, 2))))
            Kc = k_chunk or max(1, int(math.sqrt(max(seq_len, 2) * max(1, int(math.log2(max(seq_len, 2)))))))
            temp_mb = theoretical_block_temp_mb(B, H, Qc, Kc, D)
            extras = f" | est_tile_tmp≈{temp_mb:.2f} MB (Qc={Qc},Kc={Kc})"
        print(
            f"seq={seq_len:4d} | policy={kv_policy:9s} | k_chunk={k_chunk} | q_chunk={q_chunk} "
            f"| kv_heads={kv_heads} | time={t_avg:.4f}s | Δmem={m_avg:.2f} MB{extras}"
        )
    return t_avg, m_avg


# ----------------------------
# CLI
# ----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq", type=int, default=1024)
    parser.add_argument("--kv-policy", type=str, default="cache", choices=["cache", "grouped", "recompute"])
    parser.add_argument("--chunk", type=int, default=None, help="K-chunk size for kv-policy=recompute")
    parser.add_argument("--q-chunk", type=int, default=None, help="Q-chunk size for kv-policy=recompute")
    parser.add_argument("--kv-heads", type=int, default=None, help="Number of KV heads for kv-policy=grouped")
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--csv", type=str, default=None)
    parser.add_argument("--print", action="store_true")
    args = parser.parse_args()

    t, m = benchmark(
        args.seq, args.kv_policy, args.runs, k_chunk=args.chunk, kv_heads=args.kv_heads,
        q_chunk=args.q_chunk, print_result=args.print
    )

    if args.csv:
        file_exists = os.path.isfile(args.csv)
        with open(args.csv, "a", newline="") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(["seq_len", "kv_policy", "k_chunk", "q_chunk", "kv_heads", "time_s", "delta_mem_mb"])
            writer.writerow([args.seq, args.kv_policy, args.chunk, args.q_chunk, args.kv_heads, t, m])


if __name__ == "__main__":
    main()
