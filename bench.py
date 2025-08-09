#!/usr/bin/env python3
import argparse, time, importlib, inspect, os, sys
import torch

# Optional psutil (for RSS on CPU). If missing, we'll skip RSS.
try:
    import psutil
except Exception:
    psutil = None

# --- Import your library ---
try:
    et = importlib.import_module("enhanced_transformer")
except Exception as e:
    print("Failed to import enhanced_transformer.py in current directory.")
    print("Error:", e)
    sys.exit(1)

def _device():
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

def _rss_mb():
    if not psutil:
        return None
    p = psutil.Process(os.getpid())
    return p.memory_info().rss / (1024 * 1024)

def _reset_cuda_peak():
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

def _peak_cuda_mb():
    if not torch.cuda.is_available():
        return None
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated() / (1024 * 1024)

def _make_block(d_model, heads, kv_policy, k_chunk, q_chunk, kv_heads, device):
    """
    Create TransformerBlock with signature compatibility:
      - If __init__ has 'chunk_size', pass chunk_size=k_chunk
      - Else if it has 'k_chunk'/'q_chunk', pass those
      - Always pass kv_policy if supported; kv_heads if supported
    """
    params = inspect.signature(et.TransformerBlock.__init__).parameters
    kwargs = {}

    if 'kv_policy' in params:
        kwargs['kv_policy'] = kv_policy

    # Newer signature style
    if 'k_chunk' in params or 'q_chunk' in params:
        if 'k_chunk' in params:
            kwargs['k_chunk'] = k_chunk
        if 'q_chunk' in params:
            kwargs['q_chunk'] = q_chunk
    # Older signature style
    elif 'chunk_size' in params:
        kwargs['chunk_size'] = k_chunk  # map K chunk to the single chunk size

    if 'kv_heads' in params and kv_heads is not None:
        kwargs['kv_heads'] = kv_heads

    block = et.TransformerBlock(d_model, heads, **kwargs).to(device)
    block.eval()
    return block

@torch.inference_mode()
def run_once(seq_len, policy, d_model, heads, k_chunk, q_chunk, kv_heads, device):
    torch.manual_seed(0)
    x = torch.randn(1, seq_len, d_model, device=device)

    # Warmup
    block = _make_block(d_model, heads, policy, k_chunk, q_chunk, kv_heads, device)
    for _ in range(2):
        _ = block(x)

    # Measure
    _reset_cuda_peak()
    rss_before = _rss_mb()

    t0 = time.time()
    y = block(x)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    dt = time.time() - t0

    rss_after = _rss_mb()
    peak_cuda = _peak_cuda_mb()
    delta_rss = (rss_after - rss_before) if (rss_after is not None and rss_before is not None) else None

    return dt, peak_cuda, delta_rss, tuple(y.shape)

@torch.inference_mode()
def bench_grid(seq_list, configs, runs, d_model, heads, device, verbose=False):
    rows = []
    for seq in seq_list:
        for cfg in configs:
            policy = cfg['policy']
            kv_heads = cfg.get('kv_heads')
            k_chunk = cfg.get('k_chunk')
            q_chunk = cfg.get('q_chunk')

            times = []
            peaks = []
            rss_list = []
            shape = None
            for _ in range(runs):
                t, peak, rss, shape = run_once(seq, policy, d_model, heads, k_chunk, q_chunk, kv_heads, device)
                times.append(t)
                if peak is not None: peaks.append(peak)
                if rss  is not None: rss_list.append(rss)

            mean_t = sum(times) / len(times)
            mean_peak = (sum(peaks) / len(peaks)) if peaks else None
            mean_rss = (sum(rss_list) / len(rss_list)) if rss_list else None

            row = {
                'seq': seq,
                'policy': policy,
                'k_chunk': k_chunk,
                'q_chunk': q_chunk,
                'kv_heads': kv_heads,
                'time_s': round(mean_t, 4),
                'peak_cuda_MB': round(mean_peak, 2) if mean_peak is not None else None,
                'delta_rss_MB': round(mean_rss, 2) if mean_rss is not None else None,
                'out_shape': shape,
            }
            rows.append(row)

            if verbose:
                print(
                    f"seq={seq} | policy={policy:<9} | "
                    f"k_chunk={str(k_chunk):<5} | q_chunk={str(q_chunk):<5} | "
                    f"kv_heads={str(kv_heads):<4} | time={mean_t:.4f}s"
                    + (f" | peak_cuda={mean_peak:.2f} MB" if mean_peak is not None else "")
                    + (f" | ΔRSS={mean_rss:.2f} MB" if mean_rss is not None else "")
                )
    return rows

def main():
    parser = argparse.ArgumentParser(description="Benchmark Williams-style attention policies")
    parser.add_argument('--seq', type=int, nargs='+', default=[1024, 4096],
                        help="Sequence lengths to test")
    parser.add_argument('--runs', type=int, default=3, help="Runs per point")
    parser.add_argument('--d-model', type=int, default=256)
    parser.add_argument('--heads', type=int, default=8)
    parser.add_argument('--print', action='store_true', dest='do_print', help="Print table rows")
    # override defaults if you want a single config
    parser.add_argument('--policy', type=str, choices=['cache','grouped','recompute'], default=None)
    parser.add_argument('--k-chunk', type=int, default=None)
    parser.add_argument('--q-chunk', type=int, default=None)
    parser.add_argument('--kv-heads', type=int, default=None)
    args = parser.parse_args()

    device = _device()

    # Default trio of configs (matches what you were running)
    default_configs = [
        {'policy': 'cache'},
        {'policy': 'grouped', 'kv_heads': 2},
        {'policy': 'recompute', 'k_chunk': 256, 'q_chunk': 64},
    ]

    if args.policy is not None:
        # Use a single config from CLI
        cfg = {'policy': args.policy}
        if args.k_chunk is not None: cfg['k_chunk'] = args.k_chunk
        if args.q_chunk is not None: cfg['q_chunk'] = args.q_chunk
        if args.kv_heads is not None: cfg['kv_heads'] = args.kv_heads
        configs = [cfg]
    else:
        configs = default_configs

    rows = bench_grid(args.seq, configs, runs=args.runs,
                      d_model=args.d_model, heads=args.heads,
                      device=device, verbose=args.do_print)

    if not args.do_print:
        # Print a compact summary at the end
        print("seq,policy,k_chunk,q_chunk,kv_heads,time_s,peak_cuda_MB,delta_rss_MB,out_shape")
        for r in rows:
            print("{seq},{policy},{k_chunk},{q_chunk},{kv_heads},{time_s},{peak_cuda_MB},{delta_rss_MB},{out_shape}"
                  .format(**{k: (v if v is not None else '') for k,v in r.items()}))

if __name__ == "__main__":
    main()
